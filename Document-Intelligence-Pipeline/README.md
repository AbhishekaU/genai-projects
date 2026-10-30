# Document Intelligence Pipeline

Turns messy commercial contracts (PDFs) into clean, validated JSON records -- and
measures how well it does that against lawyer-labelled ground truth.

Built on **CUAD** (510 real SEC-filed contracts, labelled by lawyers across 41
clause categories) with **Gemini** via LangChain, plus a Streamlit UI.

```
PDF ──► parser ──► Gemini (structured output) ──► ContractRecord ──► validation ──► JSON
        │                                          17 fields, each      │
        ├─ cuad_txt   (clean text, upper bound)    with an evidence     ├─ format checks
        ├─ pypdf      (PDF text layer)             quote                └─ grounding: is every
        └─ native_pdf (Gemini reads the PDF)                               quote really in the doc?
                                                         │
                                                  evaluate.py ──► precision / recall per field
                                                                   vs CUAD labels, + latency, tokens
```

## What it extracts

| Kind | Fields |
|---|---|
| Key terms (values) | document name, parties, agreement / effective / expiration date, renewal term, notice period to terminate renewal, governing law |
| Clause flags (yes/no) | non-compete, exclusivity, change of control, anti-assignment, termination for convenience, cap on liability, uncapped liability, audit rights, insurance |

Every field carries a verbatim `evidence` quote, so each value can be traced back
to the text -- and checked automatically.

## Files

| File | Role |
|---|---|
| `download_data.py` | Downloads CUAD, writes `data/contracts/c000.pdf/.txt` + `manifest.csv` (short ids dodge Windows' path-length limit) |
| `schema.py` | The `ContractRecord` Pydantic model, and `FIELDS`: which CUAD column each field is scored against |
| `parsers.py` | The three ways a document reaches the model (`cuad_txt`, `pypdf`, `native_pdf`) |
| `extractor.py` | One Gemini call per contract → `ContractRecord`, with latency and token usage |
| `validate.py` | Format checks + grounding check (catches invented values) |
| `matching.py` | Decides if a prediction matches the gold label: normalizes dates (`3/20/20` = `2020-03-20`), durations (`12 months` = `1 year`), names, jurisdictions |
| `evaluate.py` | Runs a sample, caches predictions, reports precision/recall/F1 per field |
| `app.py` | Streamlit UI: upload a PDF, get the record + validation report + JSON download |

## Setup

```bash
cd Document-Intelligence-Pipeline
python -m venv venv
venv\Scripts\activate            # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

copy .env.example .env           # then put your Gemini API key in .env
python download_data.py          # ~106 MB download, one time
```

## Run

```bash
# Score one parser on 25 contracts (same 25 every time, via --seed)
python evaluate.py --parser pypdf --n 25
python evaluate.py --parser native_pdf --n 25
python evaluate.py --parser cuad_txt --n 25

# Side-by-side comparison of every parser you've run
python evaluate.py --compare

# Web UI
streamlit run app.py
```

**Speed:** each call takes ~20-40s, mostly Gemini "thinking". `--workers 4`
(the default) runs contracts in parallel; drop it to 2 if you see rate-limit
retries. Set `GEMINI_THINKING_LEVEL=low` in `.env` for much faster calls --
those runs are saved as `pypdf_low` etc., so `--compare` shows whether less
thinking cost any accuracy.

Predictions are cached in `results/<run>/`, so re-running after changing
`matching.py` re-scores without calling the model again. Every individual
mistake is listed under `errors` in `results/summary_<parser>.json` -- read
through those; that's where the insights are.

## How it's scored

- **Precision** -- of the fields the system filled in, how many were right.
- **Recall** -- of the fields the contract actually has, how many it got right.
- A filled-in *wrong* value counts against both.
- **Grounding rate** -- share of evidence quotes found in the contract text
  (fuzzy match ≥ 90, so curly quotes and line breaks don't count as hallucinations).
- **Cost side** -- average latency and input/output tokens per contract, per parser.

## Findings so far

25 CUAD contracts (`--seed 42`), `pypdf` parser, `gemini-3.8-flash`.

| Run | What changed | Values F1 | Flags F1 | Latency | Tokens out |
|---|---|---|---|---|---|
| `pypdf` | baseline, default (high) thinking | 93% | 86% | 25.3s | 2,592 |
| `pypdf_low` | thinking level `low` | 91% | 88% | 8.3s | 830 |
| `pypdf_low_v2` | term length extracted, expiration date computed in code | 91% | 87% | 8.3s | 882 |
| `pypdf_low_v3` | field definitions moved into the system prompt | 92% | 91% | 8.2s | 882 |
| `pypdf_low_v4` | renewal definition aligned with CUAD's | 91% | 92% | 8.9s | 871 |

**Final test** -- prompt v5 on `--seed 123`, a sample never used for tuning
(5 of its 25 contracts also appear in the tuning samples; 20 are unseen):

| Run | Values F1 | Flags F1 | Grounded | Latency | Tokens out |
|---|---|---|---|---|---|
| `pypdf_low_v5_seed123` | **88%** | **91%** | 99% | 7.7s | 809 |

Tuning history after v4: v4 on `--seed 7` (held out at the time) scored
91% / 87%; v5 fixed the clause definitions behind that gap (seed 7: 92% / 90%),
then was tested once on seed 123 and left alone.

What each step taught:

- **Thinking level is a free win here.** `low` cut latency and output tokens
  ~3x with overall F1 within ±2 points. Extracting terms is mostly lookup.
- **Don't make the LLM do arithmetic.** Given "for a period of three (3) years"
  it returned no expiration date. Extracting the term and adding it to the
  effective date in code (`derive.py`) took expiration recall from 46% to 77-85%.
- **The model wasn't reading the field definitions.** They were in the JSON
  schema next to a `$ref`, where they get ignored. A broader "cap on liability"
  definition (CUAD counts "no consequential damages" clauses as a cap) changed
  nothing until it went into the system prompt; then recall went 50% -> 100%.
- **Once definitions are read, they have to be right.** v3 exposed that my
  renewal definition ("automatic renewals only") was narrower than CUAD's;
  renewal recall fell 83% -> 33% until v4 fixed the wording. Dataset
  vocabulary has to match the label guidelines, not everyday meaning.
- **Grounding: 100%, zero invented quotes.** The first run flagged 10 quotes,
  but all were PDF layout noise (`be --------- ---- construed`), `...`
  shortening, or reordered signature blocks -- the checker was too strict,
  not the model.
- **Label gaps are real.** ~6-8 errors per run are values the model found
  with a verified quote that CUAD left blank (e.g. governing law "State of
  Florida" on c327). These are `label_gap?` in the error list; reported
  precision is a lower bound.
- **Remaining weak spots:** change of control (precision 50-62%) and
  uncapped liability -- clause *interpretation*, not lookup.
- **Held-out check** (v4 on 25 *different* contracts, `--seed 7`): values F1
  **91%** -- same as on the tuning set, so the value fixes generalize. Flags F1
  **87%** vs 92% -- clause interpretation overfit somewhat. Most held-out flag
  errors: "may assign to an acquirer without consent" read as change of control
  (it's an anti-assignment carve-out), and restrictions on selling competing
  products read as non-compete.
- The held-out run also exposed more **matcher** bugs: CUAD party labels
  include a "X and Y (collectively the Parties)" summary entry, semicolons inside
  brackets, and outer brackets -- 3 of 5 "wrong" parties were scoring errors.
- **Not every error is worth a prompt fix.** CUAD labels "entered into as of
  <date>" as the effective date in 291 contracts and leaves it blank in 118 --
  there's no rule to learn, so v5 left `effective_date` alone rather than fit
  four examples.
- **v5 (clause definitions)** -- change of control narrowed to "the other party
  gets a right when one party is acquired" (permission to assign to an acquirer
  is an anti-assignment carve-out), non-compete separated from exclusivity and
  licence restrictions. On the untouched test: change of control 100% F1,
  flags 91% overall -- the fix generalized.
- **What the test sample says about values (88%):** 13 of the errors are
  `label_gap?` -- mostly expiration dates `derive.py` computed ("Agreement
  Term: 1 Year") where CUAD didn't, and renewals/notice periods CUAD didn't
  label. CUAD is inconsistent about computing dates, so some of these are
  scoring artefacts. Real misses: the v4 renewal definition is now slightly
  too broad (it picked up "may be extended *by mutual agreement*"), and
  contracts with two candidate dates (preamble vs signature).
- **Grounding checks the quote, not the value.** On c436 the quote is real
  ("Date: ______, 1997") but the model still produced a full date from it. A
  next step: check the value can be read from its own quote.

### Parser comparison (prompt v5, test sample `--seed 123`)

| Parser | Values F1 | Flags F1 | Latency | Tokens in | Tokens out |
|---|---|---|---|---|---|
| `cuad_txt` (clean text, upper bound) | 88% | 92% | 6.8s | 7,634 | 825 |
| `pypdf` before whitespace fix | 88% | 91% | 7.7s | 10,868 | 809 |
| `pypdf` (NFKC-normalized) | 88% | 92% | **6.7s** | 8,051 | 818 |
| `native_pdf` (Gemini reads the PDF) | 87% | 92% | 7.7s | **6,514** | 697 |

- **On these contracts the parser doesn't change accuracy** -- all within a
  point of the clean-text upper bound. CUAD's PDFs are digitally generated
  with good text layers, so text extraction loses almost nothing.
- **A hidden cost bug:** pypdf emitted a no-break space (U+00A0) between every
  word on many SEC filings -- invisible, same character count, but 2-3x the
  tokens on affected contracts (c393: 42k vs 15k). One line of Unicode
  normalization cut pypdf input tokens 26% at identical accuracy.
- **Trade-off:** `native_pdf` uses the fewest input tokens (~19% fewer than
  fixed `pypdf`) and is the only route that would work on scanned documents;
  `pypdf` is ~1s faster per contract. For digital PDFs either is fine; for a
  mixed inbox with scans, `native_pdf`.

## Resume / interview talking points

Fill these in from your eval numbers:

1. **Parser trade-off** -- F1 and cost of `pypdf` vs `native_pdf` vs `cuad_txt`; which you'd ship and why.
2. **Hardest fields** -- which fields had the lowest precision/recall, and why (e.g. relative expiration dates, implied clauses).
3. **Hallucination control** -- evidence quotes + grounding check; what grounding rate you achieved.
4. **Prompt / context engineering** -- what changes to the system prompt or field descriptions moved the numbers.
5. **Eval design** -- normalizing dates/durations/names before comparing, and handling label noise.

## Ideas to extend

- Add a real OCR route (e.g. Tesseract) and compare on scanned contracts.
- Try a second model (set `GEMINI_MODEL`) and plot F1 vs cost per contract.
- Retry-on-validation-failure: re-ask the model only for fields that failed grounding.
- Break results down by `contract_type` (in `data/manifest.csv`).

## Walkthrough video

`video/` builds a ~2-minute narrated slide video of this project (1920x1080,
burned-in subtitles) -- every number on the slides is read from `results/`.

```bash
python video/capture_demo.py     # screenshot of the app for the demo slide (1 model call)
python video/make_video.py       # voice (Gemini TTS) + render slides + assemble with ffmpeg
python video/voice.py --check    # transcribe each clip back and compare with the script
```

Needs ffmpeg on PATH and Google Chrome installed (Playwright drives the system
Chrome, so no browser download). Edit the script in `video/narration.py`; only
changed slides are re-voiced. Output: `video/build/document-intelligence-pipeline.mp4`.
