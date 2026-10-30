"""
Validation: checks that come after the model's output already has the right
JSON shape (Gemini's structured output takes care of that).

    validate(record, document_text) -> list[Issue]

Two kinds of check:
  * format    -- dates look like dates, durations like durations, a clause
                 marked present actually comes with a quote.
  * grounding -- each evidence quote is really in the document. A value
                 whose quote can't be found is the clearest sign the model
                 made something up instead of reading it.
"""

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from matching import parse_duration, parse_pred_date
from schema import FIELDS, ContractRecord

# partial_ratio of the quote against the document. Not 100, so that harmless
# differences (curly vs straight quotes, a line break mid-sentence, a dropped
# comma) don't count as hallucinations.
GROUNDING_THRESHOLD = 90


@dataclass
class Issue:
    field: str
    kind: str  # "format" | "missing_evidence" | "ungrounded"
    message: str


def _flatten(s: str) -> str:
    # Words and numbers only. PDF text layers are full of layout noise that a
    # correct quote won't reproduce: "governed by and be --------- ---- construed",
    # curly vs straight quotes, hyphenation, stray symbols.
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _is_segment_grounded(segment: str, flat_document: str) -> bool:
    q = _flatten(segment)
    return q in flat_document or fuzz.partial_ratio(q, flat_document) >= GROUNDING_THRESHOLD


def is_grounded(quote: str, flat_document: str) -> bool:
    # A quote shortened with "..." or spread over lines (e.g. a signature block,
    # whose cells PDF parsers emit in any order) is checked piece by piece --
    # each piece must be real text, even if they're not adjacent. Fragments
    # under 3 words ("Date:", "By:") are too short to tell anything.
    segments = [s for s in re.split(r"\.\.\.|…|\n", quote) if len(_flatten(s).split()) >= 3]
    return all(_is_segment_grounded(s, flat_document) for s in segments or [quote])


def validate(record: ContractRecord, document_text: str) -> list[Issue]:
    issues = []
    flat_doc = _flatten(document_text)

    for spec in FIELDS:
        field = getattr(record, spec.name)

        if spec.kind == "flag":
            has_value = field.present
        elif spec.kind == "parties":
            has_value = bool(field.value)
        else:
            has_value = field.value is not None
            if has_value and spec.kind == "date" and field.value.lower() != "perpetual" \
                    and parse_pred_date(field.value) is None:
                issues.append(Issue(spec.name, "format", f"not a YYYY[-MM[-DD]] date: {field.value!r}"))
            if has_value and spec.kind == "duration" and parse_duration(field.value) is None:
                issues.append(Issue(spec.name, "format", f"not a duration: {field.value!r}"))

        if not has_value:
            continue
        if not field.evidence:
            issues.append(Issue(spec.name, "missing_evidence", "value given without a supporting quote"))
        elif not is_grounded(field.evidence, flat_doc):
            issues.append(Issue(spec.name, "ungrounded", f"quote not found in document: {field.evidence[:80]!r}"))

    return issues
