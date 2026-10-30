"""
Loads CUAD contracts and their gold labels from data/ (run
download_data.py first).
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from schema import FIELDS

DATA_DIR = Path(__file__).parent / "data"
CONTRACTS_DIR = DATA_DIR / "contracts"


@dataclass
class Contract:
    id: str
    filename: str
    contract_type: str
    gold: dict[str, str]  # field name -> CUAD answer ("" when the clause isn't there)

    @property
    def pdf_path(self) -> Path:
        return CONTRACTS_DIR / f"{self.id}.pdf"

    @property
    def txt_path(self) -> Path:
        return CONTRACTS_DIR / f"{self.id}.txt"

    def text(self) -> str:
        """CUAD's own clean text of the contract -- the reference the
        evidence quotes are checked against, whichever parser was used."""
        return self.txt_path.read_text(encoding="utf-8", errors="ignore")


def load_contracts() -> list[Contract]:
    if not (DATA_DIR / "manifest.csv").exists():
        raise FileNotFoundError("No dataset found -- run `python download_data.py` first.")

    with open(DATA_DIR / "master_clauses.csv", encoding="utf-8") as f:
        labels = {row["Filename"]: row for row in csv.DictReader(f)}
    with open(DATA_DIR / "manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    return [
        Contract(
            id=m["id"],
            filename=m["filename"],
            contract_type=m["contract_type"],
            gold={spec.name: labels[m["filename"]][spec.cuad_column].strip()
                  for spec in FIELDS if spec.cuad_column},
        )
        for m in manifest
    ]


def sample_contracts(n: int, seed: int = 42) -> list[Contract]:
    """A fixed random sample, so every parser/model is scored on the same contracts."""
    contracts = load_contracts()
    return sorted(random.Random(seed).sample(contracts, min(n, len(contracts))), key=lambda c: c.id)
