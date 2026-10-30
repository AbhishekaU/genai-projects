"""
The structured record the pipeline produces for every contract.

Every field carries an `evidence` quote copied from the contract. That's what
makes the output checkable: validate.py confirms each quote really appears in
the document, which catches values the model invented rather than read.

FIELDS (bottom of the file) is the single list that ties each schema field to
the CUAD label column it's scored against and to how it should be compared --
evaluate.py and validate.py both loop over it, so adding a field means adding
it to the model and one line there.

Bump PROMPT_VERSION whenever the field descriptions here or the system prompt
in extractor.py change: evaluate.py saves each version's results in its own
folder, so before/after runs can be compared instead of overwriting each other.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

# v1: first eval baseline.
# v2: initial_term added (expiration computed in code), broader cap_on_liability,
#     no "..." in quotes.
# v3: field definitions also put in the system prompt (Gemini ignored them
#     beside a $ref in the JSON schema -- see extractor.py).
# v4: renewal_term matches CUAD's definition (optional renewals count, not just
#     automatic ones) -- v3 made the model actually read the narrower one.
# v5: change_of_control and non_compete narrowed to CUAD's meaning (tuned on
#     --seed 7's errors; --seed 123 is the untouched test). effective_date left
#     alone: CUAD labels "entered into as of <date>" both ways (291 vs 118).
PROMPT_VERSION = 5


class TextField(BaseModel):
    value: str | None = Field(
        description="The extracted value, normalized as the field's description says. "
        "null if the contract doesn't state it."
    )
    evidence: str | None = Field(
        description="A short verbatim quote (under 40 words) from the contract that "
        "supports the value. Copy it exactly. null when value is null."
    )


class PartiesField(BaseModel):
    value: list[str] = Field(
        description="Legal names of every party that signs the contract, e.g. "
        "'Acme Corp.'. Names only -- not defined-term nicknames like 'Licensor'."
    )
    evidence: str | None = Field(
        description="Verbatim quote (under 40 words) where the parties are named."
    )


class ClauseFlag(BaseModel):
    present: bool = Field(description="True if the contract contains this kind of clause.")
    evidence: str | None = Field(
        description="Verbatim quote (under 40 words) of the clause. Required when "
        "present is true, null otherwise."
    )


class ContractRecord(BaseModel):
    # --- Values --------------------------------------------------------------
    document_name: TextField = Field(description="The title of the agreement, e.g. 'DISTRIBUTOR AGREEMENT'.")
    parties: PartiesField
    agreement_date: TextField = Field(
        description="Date the contract was signed, as YYYY-MM-DD. Use YYYY-MM or YYYY "
        "when only part of the date is given."
    )
    effective_date: TextField = Field(
        description="Date the contract takes effect, as YYYY-MM-DD (or YYYY-MM / YYYY). "
        "null if it's only defined relative to something else and no date can be worked out."
    )
    expiration_date: TextField = Field(
        description="Date the initial term ends, ONLY if the contract states it as a "
        "calendar date (YYYY-MM-DD, or YYYY-MM / YYYY), or 'perpetual' if it has no end. "
        "If the end is only given as a length of time, leave this null and fill "
        "initial_term instead -- the date is calculated from that."
    )
    initial_term: TextField = Field(
        description="Length of the initial term, e.g. '3 years', when it's stated as a "
        "duration ('for a period of three (3) years from the Effective Date'). "
        "'perpetual' if it continues indefinitely. null if only an end date is given."
    )
    renewal_term: TextField = Field(
        description="Length of each renewal or extension after the initial term, e.g. "
        "'1 year', '6 months', or 'perpetual'. Counts automatic renewals AND renewals one "
        "party can trigger by giving notice ('may extend for an additional 3 years'). "
        "null only if the contract has no renewal at all."
    )
    notice_period_to_terminate_renewal: TextField = Field(
        description="Notice a party must give to stop the next renewal (or, if renewal "
        "is optional, to trigger it), e.g. '90 days'."
    )
    governing_law: TextField = Field(
        description="State or country whose law governs the contract, e.g. 'New York'."
    )

    # --- Clause flags --------------------------------------------------------
    non_compete: ClauseFlag = Field(
        description="A party is barred from competing with the other party in general, or "
        "from operating in a geography, market or business sector (often lasting after "
        "the contract ends). NOT a non-compete: a promise to only sell/buy the other "
        "party's products during the term (that's exclusivity), or a limit on how "
        "licensed IP, know-how or materials may be used (that's a license restriction)."
    )
    exclusivity: ClauseFlag = Field(description="Exclusive dealing, exclusive license, or a ban on working with others.")
    change_of_control: ClauseFlag = Field(
        description="If a party undergoes a change of control (merger, acquisition, sale "
        "of a controlling stake or of substantially all its assets), the OTHER party gets "
        "a right -- to terminate, or to require its consent or notice. NOT change of "
        "control: permission to assign the contract to an acquirer or successor without "
        "consent -- that's an exception to anti-assignment, not a change-of-control right."
    )
    anti_assignment: ClauseFlag = Field(description="Consent or notice is needed to assign the contract to a third party.")
    termination_for_convenience: ClauseFlag = Field(description="A party may terminate without cause, just by giving notice.")
    cap_on_liability: ClauseFlag = Field(
        description="Any limit on a party's liability. This includes a maximum amount "
        "recoverable, a time limit for bringing claims, AND clauses excluding types of "
        "damages ('in no event shall either party be liable for consequential, "
        "indirect, special or punitive damages') -- those count as a cap too."
    )
    uncapped_liability: ClauseFlag = Field(description="Liability that is explicitly not capped (e.g. carve-outs from the cap).")
    audit_rights: ClauseFlag = Field(description="A party may audit or inspect the other's books, records or premises.")
    insurance: ClauseFlag = Field(description="A party must maintain insurance.")


@dataclass(frozen=True)
class FieldSpec:
    name: str          # attribute on ContractRecord
    cuad_column: str | None  # "-Answer" column holding the gold label; None = not scored
    kind: Literal["text", "parties", "date", "duration", "flag"]


FIELDS = [
    FieldSpec("document_name", "Document Name-Answer", "text"),
    FieldSpec("parties", "Parties-Answer", "parties"),
    FieldSpec("agreement_date", "Agreement Date-Answer", "date"),
    FieldSpec("effective_date", "Effective Date-Answer", "date"),
    FieldSpec("expiration_date", "Expiration Date-Answer", "date"),
    # No CUAD label of its own -- it feeds expiration_date (see derive.py) and
    # its quote is still grounding-checked.
    FieldSpec("initial_term", None, "duration"),
    FieldSpec("renewal_term", "Renewal Term-Answer", "duration"),
    # CUAD's header really does have a space after the hyphen here.
    FieldSpec("notice_period_to_terminate_renewal", "Notice Period To Terminate Renewal- Answer", "duration"),
    FieldSpec("governing_law", "Governing Law-Answer", "text"),
    FieldSpec("non_compete", "Non-Compete-Answer", "flag"),
    FieldSpec("exclusivity", "Exclusivity-Answer", "flag"),
    FieldSpec("change_of_control", "Change Of Control-Answer", "flag"),
    FieldSpec("anti_assignment", "Anti-Assignment-Answer", "flag"),
    FieldSpec("termination_for_convenience", "Termination For Convenience-Answer", "flag"),
    FieldSpec("cap_on_liability", "Cap On Liability-Answer", "flag"),
    FieldSpec("uncapped_liability", "Uncapped Liability-Answer", "flag"),
    FieldSpec("audit_rights", "Audit Rights-Answer", "flag"),
    FieldSpec("insurance", "Insurance-Answer", "flag"),
]
