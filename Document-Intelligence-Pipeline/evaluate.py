"""
Runs the extractor over a sample of CUAD contracts and scores it against
the lawyers' labels.

    python evaluate.py --parser pypdf --n 25
    python evaluate.py --parser native_pdf --n 25
    python evaluate.py --compare            # table of every parser already run

Metrics, per field:
  precision -- of the fields the system filled in, how many were right
  recall    -- of the fields the contract actually has, how many it got right
A filled-in but wrong value counts against both (it's a false positive
*and* the real answer was missed).

Also reported: grounding rate (share of evidence quotes found in the text),
format issues, average latency, and tokens per contract -- the cost side of
the parser trade-off.

Errors are split by type, because not every miss is the model's fault:
  missed      -- the contract has it (per CUAD), the system returned nothing
  wrong       -- both have a value, and they don't match
  extra       -- the system found something CUAD has no label for
  label_gap?  -- an "extra" value (not a yes/no clause) whose quote is verified
                 in the text: likely a gap in CUAD's labels, not a model error

Predictions are cached in results/<run>/<id>.json, so re-running only calls
the model for contracts it hasn't seen; delete the folder to redo. <run> is
the parser + thinking level + prompt version, e.g. pypdf_low_v2.
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from dataset import Contract, sample_contracts
from matching import compare_date, compare_duration, compare_parties, compare_text
from parsers import PARSERS
from schema import FIELDS, PROMPT_VERSION, ContractRecord, FieldSpec
from validate import validate

load_dotenv()  # for GEMINI_THINKING_LEVEL, which decides the results folder

RESULTS_DIR = Path(__file__).parent / "results"


# --------------------------------------------------------------- running ---

def run_tag(parser: str, prompt_version: int = PROMPT_VERSION) -> str:
    """Results folder name: the parser, plus the thinking level if one is set,
    plus the prompt version from v2 on (v1 runs predate versioning), so runs
    with different settings are cached and compared separately."""
    tag = parser
    if level := os.getenv("GEMINI_THINKING_LEVEL"):
        tag += f"_{level}"
    if prompt_version > 1:
        tag += f"_v{prompt_version}"
    return tag


def run_extractions(contracts: list[Contract], tag: str, parser: str, workers: int) -> dict[str, dict]:
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    preds, todo = {}, []

    for c in contracts:
        cache = out_dir / f"{c.id}.json"
        if cache.exists():
            preds[c.id] = json.loads(cache.read_text(encoding="utf-8"))
        else:
            todo.append(c)
    if not todo:
        return preds

    # Imported only once a model call is needed, so re-scoring cached
    # results (e.g. after changing matching.py) works without an API key.
    from extractor import extract

    print(f"{len(preds)} cached, {len(todo)} to extract with {workers} parallel worker(s)...")
    # Each call is ~20-40s of waiting on the API, so running a few at once
    # cuts wall time roughly by the worker count. Too many and the free tier
    # starts rate-limiting (calls then slow down as the client retries).
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract, c.pdf_path, parser, c.txt_path): c for c in todo}
        for done, future in enumerate(as_completed(futures), 1):
            c, ex = futures[future], future.result()
            prefix = f"[{done}/{len(todo)}] {c.id} ({c.contract_type})"
            if ex.error:
                print(f"{prefix} failed after {ex.latency_s:.0f}s: {ex.error[:200]}", flush=True)
                continue  # not cached, so the next run retries it
            print(f"{prefix} {ex.latency_s:.0f}s", flush=True)

            pred = {
                "record": ex.record.model_dump(),
                "model": ex.model,
                "latency_s": round(ex.latency_s, 2),
                "input_tokens": ex.input_tokens,
                "output_tokens": ex.output_tokens,
                "derived": ex.derived,
            }
            (out_dir / f"{c.id}.json").write_text(json.dumps(pred, indent=2), encoding="utf-8")
            preds[c.id] = pred
    return preds


# --------------------------------------------------------------- scoring ---

def score_field(spec: FieldSpec, pred_field: dict, gold: str) -> tuple[bool, bool, bool]:
    """Returns (predicted_something, gold_has_something, correct)."""
    if spec.kind == "flag":
        pred_yes, gold_yes = pred_field["present"], gold == "Yes"
        return pred_yes, gold_yes, pred_yes and gold_yes

    value = pred_field["value"]
    has_pred = bool(value)
    has_gold = bool(gold)
    if not (has_pred and has_gold):
        return has_pred, has_gold, False

    compare = {"text": compare_text, "date": compare_date,
               "duration": compare_duration, "parties": compare_parties}[spec.kind]
    return True, True, compare(value, gold)


def score(contracts: list[Contract], preds: dict[str, dict]) -> dict:
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    errors = []  # every mistake, for reading through afterwards
    issue_counts = defaultdict(int)
    grounded = total_quotes = 0

    scored = [c for c in contracts if c.id in preds]
    for c in scored:
        record = preds[c.id]["record"]
        # v1 records predate initial_term.
        record.setdefault("initial_term", {"value": None, "evidence": None})
        issues = validate(ContractRecord.model_validate(record), c.text())
        ungrounded = {i.field for i in issues if i.kind == "ungrounded"}

        for spec in FIELDS:
            if spec.cuad_column is None:
                continue  # helper field with no label (e.g. initial_term)
            has_pred, has_gold, correct = score_field(spec, record[spec.name], c.gold[spec.name])
            n = counts[spec.name]
            if correct:
                n["tp"] += 1
                continue
            n["fp"] += has_pred
            n["fn"] += has_gold
            if not (has_pred or has_gold):
                continue
            if has_pred and has_gold:
                kind = "wrong"
            elif has_gold:
                kind = "missed"
            else:
                # A verified quote shows a value really is in the text. For a
                # yes/no clause it only shows the sentence exists, not that it
                # is that kind of clause -- so flags never count as label gaps.
                grounded_value = spec.kind != "flag" and spec.name not in ungrounded
                kind = "label_gap?" if grounded_value else "extra"
            errors.append({"id": c.id, "field": spec.name, "type": kind, "gold": c.gold[spec.name],
                           "pred": record[spec.name].get("value", record[spec.name].get("present")),
                           "evidence": record[spec.name].get("evidence")})

        for issue in issues:
            issue_counts[issue.kind] += 1
        n_quotes = sum(
            1 for spec in FIELDS
            if record[spec.name].get("evidence")
            and (record[spec.name].get("present") or record[spec.name].get("value"))
        )
        total_quotes += n_quotes
        grounded += n_quotes - sum(1 for i in issues if i.kind == "ungrounded")

    def prf(n):
        p = n["tp"] / (n["tp"] + n["fp"]) if n["tp"] + n["fp"] else None
        r = n["tp"] / (n["tp"] + n["fn"]) if n["tp"] + n["fn"] else None
        f1 = 2 * p * r / (p + r) if p and r else None
        return {"precision": p, "recall": r, "f1": f1, **n}

    scored_fields = [s for s in FIELDS if s.cuad_column]
    per_field = {spec.name: prf(counts[spec.name]) for spec in scored_fields}
    by_kind = {}
    for kind in ("values", "flags"):
        names = [s.name for s in scored_fields if (s.kind == "flag") == (kind == "flags")]
        by_kind[kind] = prf({k: sum(counts[f][k] for f in names) for k in ("tp", "fp", "fn")})

    p = [preds[c.id] for c in scored]
    return {
        "n_contracts": len(scored),
        "model": p[0]["model"] if p else None,
        "per_field": per_field,
        "overall": by_kind,
        "grounding_rate": grounded / total_quotes if total_quotes else None,
        "issues": dict(issue_counts),
        "avg_latency_s": sum(x["latency_s"] for x in p) / len(p) if p else None,
        "avg_input_tokens": sum(x["input_tokens"] for x in p) / len(p) if p else None,
        "avg_output_tokens": sum(x["output_tokens"] for x in p) / len(p) if p else None,
        "error_types": {t: sum(e["type"] == t for e in errors)
                        for t in ("missed", "wrong", "extra", "label_gap?")},
        "errors": errors,
    }


# --------------------------------------------------------------- reports ---

def _pct(x):
    return "  -  " if x is None else f"{x:5.0%}"


def print_report(parser: str, s: dict) -> None:
    print(f"\n=== {parser}  |  {s['model']}  |  {s['n_contracts']} contracts ===")
    print(f"{'field':38} {'prec':>5} {'rec':>5} {'f1':>5}   tp  fp  fn")
    for name, m in s["per_field"].items():
        print(f"{name:38} {_pct(m['precision'])} {_pct(m['recall'])} {_pct(m['f1'])}  "
              f"{m['tp']:3d} {m['fp']:3d} {m['fn']:3d}")
    for kind, m in s["overall"].items():
        print(f"{'ALL ' + kind.upper():38} {_pct(m['precision'])} {_pct(m['recall'])} {_pct(m['f1'])}")
    print(f"\ngrounding rate: {_pct(s['grounding_rate'])}   issues: {s['issues'] or 'none'}")
    print(f"errors by type: {s['error_types']}")
    print(f"avg latency: {s['avg_latency_s']:.1f}s   avg tokens in/out: "
          f"{s['avg_input_tokens']:,.0f} / {s['avg_output_tokens']:,.0f}")


def compare_parsers() -> None:
    # Every run so far -- e.g. summary_pypdf.json, summary_pypdf_low.json.
    rows = [(path.stem.removeprefix("summary_"), json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(RESULTS_DIR.glob("summary_*.json"))]
    if not rows:
        print("Nothing to compare yet -- run evaluate.py with --parser first.")
        return
    width = max(len(tag) for tag, _ in rows)
    print(f"\n{'run':{width}} {'n':>3} {'values F1':>10} {'flags F1':>9} {'grounded':>9} "
          f"{'latency':>8} {'tokens in':>10} {'tokens out':>11}")
    for tag, s in rows:
        print(f"{tag:{width}} {s['n_contracts']:3d} {_pct(s['overall']['values']['f1']):>10} "
              f"{_pct(s['overall']['flags']['f1']):>9} {_pct(s['grounding_rate']):>9} "
              f"{s['avg_latency_s']:7.1f}s {s['avg_input_tokens']:10,.0f} {s['avg_output_tokens']:11,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser", choices=PARSERS, default="pypdf")
    ap.add_argument("--n", type=int, default=25, help="number of contracts to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4,
                    help="contracts extracted in parallel (lower it if you hit rate limits)")
    ap.add_argument("--prompt-version", type=int, default=PROMPT_VERSION,
                    help="re-score an older prompt version's cached results (no model calls)")
    ap.add_argument("--compare", action="store_true", help="compare parsers already evaluated")
    args = ap.parse_args()

    if args.compare:
        compare_parsers()
        return

    tag = run_tag(args.parser, args.prompt_version)
    if args.seed != 42:
        # A different sample is a held-out check: keep it apart from the
        # 25 contracts the prompt was tuned on.
        tag += f"_seed{args.seed}"
    contracts = sample_contracts(args.n, args.seed)
    if args.prompt_version != PROMPT_VERSION:
        # Old prompt: only its cached predictions can be scored, never re-extracted.
        out_dir = RESULTS_DIR / tag
        preds = {c.id: json.loads((out_dir / f"{c.id}.json").read_text(encoding="utf-8"))
                 for c in contracts if (out_dir / f"{c.id}.json").exists()}
    else:
        preds = run_extractions(contracts, tag, args.parser, args.workers)
    summary = score(contracts, preds)
    print_report(tag, summary)

    path = RESULTS_DIR / f"summary_{tag}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFull results (including every error) -> {path}")


if __name__ == "__main__":
    main()
