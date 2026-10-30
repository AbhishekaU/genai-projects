"""
The three ways a document can be handed to the model. Comparing them is the
"which parser holds up, at what cost" experiment:

    cuad_txt   -- CUAD's own clean text. Not something you'd have in
                  production; it's the upper bound for text-based parsing.
    pypdf      -- the PDF's text layer pulled out with pypdf. Cheap and
                  fast, but loses layout and gets nothing from scanned pages.
    native_pdf -- the raw PDF sent straight to Gemini, which reads the
                  pages itself (multimodal). Sees layout and scans, costs
                  more tokens per page.

Each parser returns the LangChain message content for the document part of
the prompt.
"""

import base64
import unicodedata
from pathlib import Path

from pypdf import PdfReader

PARSERS = ["cuad_txt", "pypdf", "native_pdf"]


def pypdf_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    # Many SEC-filed PDFs put no-break spaces (U+00A0) between every word.
    # They look identical, but the tokenizer handles them badly: the same
    # contract cost 2-3x the input tokens of clean text until this was added.
    # NFKC turns them (and ligatures like "ﬁ") into their plain equivalents.
    return unicodedata.normalize("NFKC", text)


def document_content(parser: str, pdf_path: Path, txt_path: Path | None = None) -> list[dict]:
    if parser == "cuad_txt":
        if txt_path is None:
            raise ValueError("cuad_txt needs the CUAD .txt file (only exists for dataset contracts).")
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        return [{"type": "text", "text": f"<contract>\n{text}\n</contract>"}]

    if parser == "pypdf":
        return [{"type": "text", "text": f"<contract>\n{pypdf_text(pdf_path)}\n</contract>"}]

    if parser == "native_pdf":
        return [{
            "type": "file",
            "base64": base64.b64encode(pdf_path.read_bytes()).decode(),
            "mime_type": "application/pdf",
        }]

    raise ValueError(f"Unknown parser {parser!r}; expected one of {PARSERS}")
