"""
Fields worked out in code from what the model extracted, rather than asked
of the model directly.

The first eval showed the model won't do date arithmetic: given "effective
on the date first written above ... for a period of three (3) years" it
returned no expiration date, while CUAD's lawyers wrote down the computed
end date. So the model now extracts the term length (initial_term) and the
date is calculated here -- deterministic arithmetic instead of an LLM guess.

    derive_fields(record) -> list of field names that were filled in
"""

import calendar
import datetime as dt
import re

from matching import parse_pred_date
from schema import ContractRecord, TextField

_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "twenty-five": 25}


def parse_term(s: str) -> tuple[int, str] | None:
    """"3 years" / "three (3) years" -> (3, "year"). None if it isn't a plain duration."""
    t = re.sub(r"[()]", " ", s.lower())
    # Longest words first, so "twenty-five" wins over "twenty".
    words = sorted(_WORD_NUMS, key=len, reverse=True)
    # Contracts write "three (3) years": allow a number word followed by digits.
    m = re.search(r"(\d+|" + "|".join(words) + r")(?:\s+\d+)?\s*(day|week|month|year)", t)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1).isdigit() else _WORD_NUMS[m.group(1)]
    return n, m.group(2)


def add_term(start: str, n: int, unit: str) -> str | None:
    """start is YYYY-MM-DD / YYYY-MM / YYYY; the result keeps the same precision.
    "3 years from 2019-08-01" -> "2022-08-01" (CUAD's convention: no minus-one-day)."""
    parts = parse_pred_date(start)
    if not parts or parts[0] is None:
        return None
    year, month, day = parts

    if unit in ("day", "week"):
        if month is None or day is None:
            return None  # can't count days from a partial date
        days = n * (7 if unit == "week" else 1)
        return (dt.date(year, month, day) + dt.timedelta(days=days)).isoformat()

    months = n * 12 if unit == "year" else n
    if month is None:
        return str(year + months // 12) if months % 12 == 0 else None
    total = (year * 12 + month - 1) + months
    new_year, new_month = divmod(total, 12)
    new_month += 1
    if day is None:
        return f"{new_year:04d}-{new_month:02d}"
    day = min(day, calendar.monthrange(new_year, new_month)[1])  # Jan 31 + 1 month
    return f"{new_year:04d}-{new_month:02d}-{day:02d}"


def derive_fields(record: ContractRecord) -> list[str]:
    derived = []

    term = record.initial_term.value
    if record.expiration_date.value is None and term:
        if "perpetual" in term.lower():
            record.expiration_date = TextField(value="perpetual", evidence=record.initial_term.evidence)
            derived.append("expiration_date")
        else:
            start = record.effective_date.value or record.agreement_date.value
            parsed = parse_term(term)
            end = add_term(start, *parsed) if start and parsed else None
            if end:
                # The quote that justifies a computed date is the term clause.
                record.expiration_date = TextField(value=end, evidence=record.initial_term.evidence)
                derived.append("expiration_date")

    return derived
