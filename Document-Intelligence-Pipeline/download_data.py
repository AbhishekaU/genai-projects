"""
Downloads the CUAD v1 dataset (Contract Understanding Atticus Dataset) and
lays it out in data/ in a form the rest of the project can use.

CUAD is 510 real commercial contracts (filed publicly with the SEC) labelled
by lawyers across 41 clause categories. The part this project uses is
master_clauses.csv, one row per contract, with two columns per category:
    "<Category>"          -- the exact clause text(s) the lawyers highlighted
    "<Category>-Answer"   -- a normalized answer (a date, a state, Yes/No...)
Those labels are what evaluate.py scores the extractor against.

Some CUAD file paths are longer than Windows' 260-character limit, so
instead of unzipping as-is, every contract is written out under a short id:
    data/contracts/c000.pdf, data/contracts/c000.txt, ...
and data/manifest.csv maps each id back to its original CUAD filename.

Run once:  python download_data.py
"""

import csv
import io
import re
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

import requests

CUAD_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
DATA_DIR = Path(__file__).parent / "data"
ZIP_PATH = DATA_DIR / "CUAD_v1.zip"
CONTRACTS_DIR = DATA_DIR / "contracts"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
LABELS_PATH = DATA_DIR / "master_clauses.csv"


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if ZIP_PATH.exists():
        return
    print("Downloading CUAD v1 (~106 MB)...")
    with requests.get(CUAD_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _key(name: str) -> str:
    """Normalizes a filename for matching. The CSV and the zip don't always
    spell the same contract identically ("MACY'S" vs "MACY_S", trailing
    spaces, accents), so compare on lowercase letters and digits only."""
    stem = unicodedata.normalize("NFKD", PurePosixPath(name).stem)
    return re.sub(r"[^0-9a-z]", "", stem.lower())


def _find(key: str, index: dict[str, str]) -> str | None:
    # Exact match first; otherwise a file whose name starts with the CSV's
    # (a few PDFs carry an extra suffix like "_Option Agreement").
    if key in index:
        return index[key]
    return next((v for k, v in index.items() if k.startswith(key)), None)


def extract() -> None:
    CONTRACTS_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        # Index PDFs and TXTs by normalized file stem, which is how
        # master_clauses.csv refers to each contract (its "Filename" column).
        pdfs = {_key(n): n for n in names if n.lower().endswith(".pdf")}
        txts = {_key(n): n for n in names
                if n.startswith("CUAD_v1/full_contract_txt/") and n.endswith(".txt")}

        labels_raw = zf.read("CUAD_v1/master_clauses.csv")
        LABELS_PATH.write_bytes(labels_raw)
        rows = list(csv.DictReader(io.StringIO(labels_raw.decode("utf-8"))))

        manifest = []
        for i, row in enumerate(rows):
            key = _key(row["Filename"])
            pdf_name, txt_name = _find(key, pdfs), _find(key, txts)
            if not pdf_name or not txt_name:
                print(f"  skipping (no pdf/txt found): {row['Filename']!a}")
                continue

            doc_id = f"c{i:03d}"
            (CONTRACTS_DIR / f"{doc_id}.pdf").write_bytes(zf.read(pdf_name))
            (CONTRACTS_DIR / f"{doc_id}.txt").write_bytes(zf.read(txt_name))
            manifest.append({
                "id": doc_id,
                "filename": row["Filename"],
                # e.g. "License_Agreements" -- the contract type folder in CUAD
                "contract_type": PurePosixPath(pdf_name).parent.name,
            })

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "filename", "contract_type"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Done -> {len(manifest)} contracts in {CONTRACTS_DIR}")


if __name__ == "__main__":
    download()
    extract()
