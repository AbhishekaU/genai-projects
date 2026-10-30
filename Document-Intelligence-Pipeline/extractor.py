"""
The extraction step: one Gemini call per contract that returns a
ContractRecord (schema.py), using Gemini's native structured output so the
response is guaranteed to be valid JSON in the right shape.

    extract(pdf_path, parser="pypdf") -> Extraction

Shape is guaranteed by the API; whether the *content* is right (no invented
values, quotes that really exist) is checked separately in validate.py.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from derive import derive_fields
from parsers import document_content
from schema import ContractRecord

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# How long Gemini 3 "thinks" before answering: minimal | low | medium | high.
# Unset = the model's default (high), which is most of the latency and output
# tokens here. Reading terms out of a contract is mostly lookup, so "low" is
# much faster -- whether it costs accuracy is exactly what evaluate.py shows.
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL") or None

# No temperature setting: Gemini 3.x models use fixed sampling defaults and
# ignore it (with a warning). Older models (e.g. gemini-2.5-flash, if you set
# GEMINI_MODEL to one) would benefit from temperature=0 for repeatable evals.
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, thinking_level=THINKING_LEVEL)
structured_llm = llm.with_structured_output(ContractRecord, method="json_schema", include_raw=True)

SYSTEM_PROMPT = """You are a contracts analyst extracting key terms from a commercial contract.

Rules:
- Only report what the contract itself says. If a field isn't stated, return null
  (or present=false for clauses). Never guess or fill in typical values.
- Every evidence quote must be copied verbatim from the contract -- same words, same
  order. Keep it short: the sentence or phrase that proves the value. Never shorten
  a quote with "..." and never join text from different places; pick the single
  contiguous passage that matters most.
- Contracts phrase the same thing in many ways ("shall automatically renew",
  "will be extended for successive periods", "unless either party gives notice of
  non-renewal") -- judge by meaning, not keywords.
- Dates: convert to YYYY-MM-DD. Use YYYY-MM or YYYY if the day or month is blank or
  missing (e.g. "____, 2020" -> "2020").
- Durations: use "<number> <days|months|years>", e.g. "1 year", "90 days".
- Parties: legal entity names as written in the contract, not roles like "Licensee".
"""

# The field definitions from schema.py, repeated in the prompt itself. They're
# also in the JSON schema, but there each one sits next to a "$ref" to a shared
# sub-model -- and JSON Schema readers commonly ignore text beside a $ref. Prompt
# v2's broader cap_on_liability definition changed nothing until it was here.
SYSTEM_PROMPT += "\nField definitions -- follow these, not the everyday meaning of the field name:\n" + "\n".join(
    f"- {name}: {info.description}"
    for name, info in ContractRecord.model_fields.items()
    if info.description
)


@dataclass
class Extraction:
    record: ContractRecord | None
    parser: str
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    error: str | None = None
    derived: list[str] = field(default_factory=list)  # fields computed in code (derive.py)


def extract(pdf_path: Path, parser: str = "pypdf", txt_path: Path | None = None) -> Extraction:
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage([
            {"type": "text", "text": "Extract the contract record from this contract."},
            *document_content(parser, pdf_path, txt_path),
        ]),
    ]

    start = time.perf_counter()
    try:
        result = structured_llm.invoke(messages)
    except Exception as e:  # network/quota errors shouldn't kill a 50-contract eval run
        return Extraction(None, parser, MODEL_NAME, time.perf_counter() - start, 0, 0, error=repr(e))
    latency = time.perf_counter() - start

    usage = result["raw"].usage_metadata or {}
    record = result["parsed"]
    derived = derive_fields(record) if record else []
    return Extraction(
        record=record,
        derived=derived,
        parser=parser,
        model=MODEL_NAME,
        latency_s=latency,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        error=repr(result["parsing_error"]) if result.get("parsing_error") else None,
    )
