"""
Decides whether an extracted value "matches" CUAD's gold answer.

This is the part of the eval that most deserves scrutiny: the model and the
lawyers who labelled CUAD write the same fact in different shapes
("2020-03-20" vs "3/20/20", "12 months" vs "1 year", "Acme Corp." vs
"ACME CORPORATION ("Acme")"). Each compare_* function normalizes both sides
first, so a match means "same fact", not "same string".
"""

import re
import unicodedata

from rapidfuzz import fuzz

# Two names count as the same if their token-sorted similarity is at least this.
NAME_THRESHOLD = 85


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


# ---------------------------------------------------------------- dates ----

def parse_gold_date(s: str) -> tuple[int | None, int | None, int | None] | None:
    """CUAD writes dates as M/D/YY or M/D/YYYY, with [] for unknown parts
    (e.g. "[]/[]/2020"). Returns (year, month, day); None if not a date."""
    m = re.fullmatch(r"\s*(\[\]|\d{1,2})/(\[\]|\d{1,2})/(\[\]|\d{2}|\d{4})\s*", s)
    if not m:
        return None
    month, day, year = (None if p == "[]" else int(p) for p in m.groups())
    if year is not None and year < 100:
        # Two-digit years: CUAD contracts span roughly 1990-2020.
        year += 2000 if year <= 30 else 1900
    return year, month, day


def parse_pred_date(s: str) -> tuple[int | None, int | None, int | None] | None:
    """The model is asked for YYYY-MM-DD, YYYY-MM or YYYY."""
    m = re.fullmatch(r"\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?\s*", s)
    if not m:
        return None
    return tuple(int(p) if p else None for p in m.groups())


def compare_date(pred: str, gold: str) -> bool:
    # A few labels list alternatives: "09/05/2019; perpetual".
    if ";" in gold:
        return any(compare_date(pred, g) for g in gold.split(";"))
    if gold.strip().lower() == "perpetual" or pred.strip().lower() == "perpetual":
        return gold.strip().lower() == pred.strip().lower()
    g, p = parse_gold_date(gold), parse_pred_date(pred)
    if g is None or p is None:
        return _norm_text(pred) == _norm_text(gold)
    # Only the parts the gold label actually knows are compared.
    return all(gv is None or gv == pv for gv, pv in zip(g, p))


# ------------------------------------------------------------ durations ----

_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}


def parse_duration(s: str) -> float | str | None:
    """"successive 1 year" -> 365, "6 months" -> 180, "perpetual" -> "perpetual"."""
    t = _norm_text(s)
    if "perpetual" in t or "indefinite" in t:
        return "perpetual"
    m = re.search(r"(\d+(?:\.\d+)?|" + "|".join(_WORD_NUMS) + r")\s*(day|week|month|year)", t)
    if not m:
        return None
    n = _WORD_NUMS.get(m.group(1)) or float(m.group(1))
    return n * _UNIT_DAYS[m.group(2)]


def compare_duration(pred: str, gold: str) -> bool:
    if ";" in gold:
        return any(compare_duration(pred, g) for g in gold.split(";"))
    p, g = parse_duration(pred), parse_duration(gold)
    if p is None or g is None:
        return _norm_text(pred) == _norm_text(gold)
    if isinstance(p, str) or isinstance(g, str):
        return p == g
    # Allow "12 months" == "1 year" (360 vs 365 days) and similar.
    return abs(p - g) <= max(7, 0.05 * g)


# ---------------------------------------------------------- free text ------

_JURISDICTION_PREFIX = r"^(the )?(laws of )?(the )?(state|province|commonwealth|republic) of "


def compare_text(pred: str, gold: str) -> bool:
    # "Province of Ontario" vs "Ontario, Canada" -> "ontario" vs "ontario canada"
    p = re.sub(_JURISDICTION_PREFIX, "", _norm_text(pred))
    g = re.sub(_JURISDICTION_PREFIX, "", _norm_text(gold))
    # "State of New York" should match "New York"; titles vary in small words.
    return p == g or g in p or p in g or fuzz.token_sort_ratio(p, g) >= NAME_THRESHOLD


# --------------------------------------------------------------- parties ---

def split_gold_parties(gold: str) -> list[str]:
    """CUAD writes parties as 'Acme Corp. ("Acme"); Beta LLC ("Licensee")'.
    Keep only the legal names, dropping the quoted nicknames. Real labels are
    messier than that, so this also handles:
      - a ";" inside the brackets: 'Beta LLC ("VNE"; collectively with VNI, "VERT")'
      - the whole label wrapped in brackets: '(Acme ("Company"); Beta ("MHC"))'
      - a summary entry that isn't a party: 'Acme and Beta (collectively the "Parties")'
    """
    gold = gold.strip()
    if gold.startswith("(") and gold.endswith(")"):
        gold = gold[1:-1]

    # Split on ";" only outside brackets -- unless the label's own brackets
    # don't balance (a handful do), where depth tracking would swallow every ";".
    if gold.count("(") != gold.count(")"):
        parts = gold.split(";")
    else:
        parts, depth, current = [], 0, ""
        for ch in gold:
            depth += (ch == "(") - (ch == ")")
            if ch == ";" and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        parts.append(current)

    names = []
    for part in parts:
        # 'Acme and Beta (individually a "Party", collectively the "Parties")'
        if re.match(r"^[^(]*\(\s*(individually|collectively)", part.strip(), re.IGNORECASE):
            continue
        name = re.sub(r"\([^()]*\)|[\"“”][^\"“”]*[\"“”]", "", part).strip(" ,()")
        if name:
            names.append(name)
    return names


_ENTITY_SUFFIXES = r"\b(inc|incorporated|corp|corporation|co|company|llc|ltd|limited|lp|plc|sa|ag|gmbh|the)\b"


def _norm_party(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(_ENTITY_SUFFIXES, "", _norm_text(s))).strip()


def compare_parties(pred: list[str], gold: str) -> bool:
    """Correct if every gold party is matched by some predicted name."""
    gold_names = [_norm_party(n) for n in split_gold_parties(gold)]
    pred_names = [_norm_party(n) for n in pred]
    return bool(gold_names) and all(
        any(g == p or fuzz.token_set_ratio(g, p) >= NAME_THRESHOLD for p in pred_names)
        for g in gold_names
    )
