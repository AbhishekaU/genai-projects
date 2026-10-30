"""
Renders the 12 video slides to 1920x1080 PNGs (video/build/slides/<id>.png).

Each slide is an HTML page drawn by system Chrome through Playwright. Every
number on a slide is read from results/summary_*.json, so the video can't
drift from the eval -- re-run evaluate.py, re-run this, and the slides follow.

    python video/render_slides.py

The bottom ~200px of every slide is left empty: make_video.py draws the
subtitles there.
"""

import html
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from narration import SLIDES  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
BUILD = Path(__file__).parent / "build"
OUT_DIR = BUILD / "slides"

AUTHOR = "Abhisheka U"
GITHUB = "github.com/AbhishekaU/genai-projects"


def summary(tag: str) -> dict:
    return json.loads((RESULTS / f"summary_{tag}.json").read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{round(x * 100)}%"


# The runs the video talks about.
BASE_HIGH = summary("pypdf")                  # v1, default (high) thinking
TUNING = [                                     # same 25 contracts, --seed 42
    ("v1", "baseline", summary("pypdf_low")),
    ("v2", "term computed in code", summary("pypdf_low_v2")),
    ("v3", "definitions in prompt", summary("pypdf_low_v3")),
    ("v4", "renewal definition fixed", summary("pypdf_low_v4")),
]
TEST = {                                       # v5 on unseen contracts, --seed 123
    "cuad_txt": summary("cuad_txt_low_v5_seed123"),
    "pypdf_nbsp": summary("pypdf-nbsp_low_v5_seed123"),
    "pypdf": summary("pypdf_low_v5_seed123"),
    "native_pdf": summary("native_pdf_low_v5_seed123"),
}
FINAL = TEST["pypdf"]

# --------------------------------------------------------------- styling ---

CSS = """
:root {
  --page: #0d0d0d; --card: #1a1a19; --card-2: #232321;
  --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --blue: #3987e5; --orange: #d95926; --good: #0ca30c; --serious: #ec835a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { width: 1920px; height: 1080px; background: var(--page); color: var(--ink);
  font-family: "Segoe UI", system-ui, sans-serif; overflow: hidden; }
.slide { position: relative; width: 1920px; height: 1080px; padding: 64px 120px 0; }
.top { display: flex; justify-content: space-between; font-size: 22px; color: var(--muted);
  letter-spacing: 0.02em; margin-bottom: 44px; }
.top b { color: var(--ink-2); font-weight: 600; }
.kicker { font-size: 26px; font-weight: 600; color: var(--blue); letter-spacing: 0.06em;
  text-transform: uppercase; margin-bottom: 14px; }
h1 { font-size: 64px; font-weight: 700; line-height: 1.1; letter-spacing: -0.01em; }
h1 .dim { color: var(--ink-2); font-weight: 600; }
.lede { font-size: 30px; color: var(--ink-2); margin-top: 18px; line-height: 1.4; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 20px; padding: 32px 36px; }
.row { display: flex; gap: 28px; align-items: stretch; }
.label { font-size: 22px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.big { font-size: 84px; font-weight: 700; line-height: 1; }
.mid { font-size: 30px; line-height: 1.45; color: var(--ink-2); }
.mono { font-family: "Cascadia Mono", Consolas, monospace; }
.quote { font-family: Georgia, "Times New Roman", serif; font-size: 27px; line-height: 1.6; color: var(--ink-2); }
mark { background: rgba(57,135,229,0.22); color: var(--ink); border-bottom: 3px solid var(--blue); padding: 0 2px; }
.arrow { font-size: 56px; color: var(--muted); align-self: center; }
.pill { display: inline-block; font-size: 22px; padding: 6px 16px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--ink-2); background: var(--card-2); }
.good { color: var(--good); }
.from { color: var(--muted); text-decoration: line-through; text-decoration-thickness: 3px; }
table { border-collapse: collapse; width: 100%; font-size: 28px; font-variant-numeric: tabular-nums; }
th { white-space: nowrap; text-align: left; font-size: 21px; font-weight: 600; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.05em; padding: 0 20px 14px; border-bottom: 1px solid var(--axis); }
td { padding: 16px 20px; border-bottom: 1px solid var(--grid); color: var(--ink-2); }
td:first-child { color: var(--ink); }
th.n, td.n { text-align: right; }
"""


def page(index: int, body: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body><div class="slide">
<div class="top"><span><b>Document Intelligence Pipeline</b> &nbsp;·&nbsp; Project 1 of 6</span>
<span>{AUTHOR} &nbsp;·&nbsp; {index:02d} / {len(SLIDES):02d}</span></div>
{body}
</div></body></html>"""


# ---------------------------------------------------------------- charts ---

def line_chart() -> str:
    """Clause and value F1 across prompt versions v1-v4 on the tuning sample.
    One y-axis (F1, %); two series, direct-labelled at the line ends + legend."""
    w, h = 1060, 470
    left, right, top, bottom = 80, 250, 30, 90
    y_min, y_max = 0.80, 1.00
    xs = [left + i * (w - left - right) / (len(TUNING) - 1) for i in range(len(TUNING))]

    def y(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * (h - top - bottom)

    parts = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
             'font-family="Segoe UI, system-ui, sans-serif">']
    for v in (0.80, 0.85, 0.90, 0.95, 1.00):
        parts.append(f'<line x1="{left}" x2="{w - right + 30}" y1="{y(v)}" y2="{y(v)}" stroke="#2c2c2a" stroke-width="1"/>')
        parts.append(f'<text x="{left - 16}" y="{y(v) + 8}" fill="#898781" font-size="22" text-anchor="end">{round(v * 100)}%</text>')
    for x, (ver, what, _) in zip(xs, TUNING):
        parts.append(f'<text x="{x}" y="{h - 52}" fill="#c3c2b7" font-size="24" font-weight="600" text-anchor="middle">{ver}</text>')
        parts.append(f'<text x="{x}" y="{h - 20}" fill="#898781" font-size="19" text-anchor="middle">{what}</text>')

    series = [("Clauses", "flags", "#3987e5"), ("Key terms", "values", "#d95926")]
    end_labels = []
    for name, key, color in series:
        vals = [s["overall"][key]["f1"] for _, _, s in TUNING]
        pts = " ".join(f"{x},{y(v)}" for x, v in zip(xs, vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" '
                     'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, v in zip(xs, vals):
            # 2px surface ring keeps the markers readable where the lines cross.
            parts.append(f'<circle cx="{x}" cy="{y(v)}" r="8" fill="{color}" stroke="#1a1a19" stroke-width="3"/>')
        end_labels.append([y(vals[-1]) + 9, f"{name} {pct(vals[-1])}"])

    # The two series can end a point apart: push their labels at least 34px apart.
    end_labels.sort()
    for upper, lower in zip(end_labels, end_labels[1:]):
        if lower[0] - upper[0] < 34:
            mid = (upper[0] + lower[0]) / 2
            upper[0], lower[0] = mid - 17, mid + 17
    for ty, text in end_labels:
        parts.append(f'<text x="{xs[-1] + 22}" y="{ty}" fill="#f0efec" font-size="26" font-weight="600">{text}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_pair(title: str, before: float, after: float, fmt) -> str:
    """Two bars, before (neutral) and after (blue), from a zero baseline."""
    w, h, base = 520, 380, 300
    top_pad = 50
    scale = (base - top_pad) / max(before, after)
    bw, gap = 150, 70
    x0 = (w - 2 * bw - gap) / 2
    parts = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, system-ui, sans-serif">']
    for i, (label, v, color) in enumerate([("default", before, "#5d5c58"), ("low", after, "#3987e5")]):
        x = x0 + i * (bw + gap)
        bh = v * scale
        # Rounded top only (4px), anchored square on the baseline.
        parts.append(f'<path d="M{x},{base} V{base - bh + 4} Q{x},{base - bh} {x + 4},{base - bh} '
                     f'H{x + bw - 4} Q{x + bw},{base - bh} {x + bw},{base - bh + 4} V{base} Z" fill="{color}"/>')
        parts.append(f'<text x="{x + bw / 2}" y="{base - bh - 14}" fill="#f0efec" font-size="34" font-weight="700" '
                     f'text-anchor="middle">{fmt(v)}</text>')
        parts.append(f'<text x="{x + bw / 2}" y="{base + 40}" fill="#c3c2b7" font-size="24" text-anchor="middle">{label}</text>')
    parts.append(f'<line x1="20" x2="{w - 20}" y1="{base}" y2="{base}" stroke="#383835" stroke-width="2"/>')
    parts.append("</svg>")
    return f'<div class="card" style="flex:1"><div class="label">{title}</div>{"".join(parts)}</div>'


# ---------------------------------------------------------------- slides ---

def contract_excerpt(contract_id: str, start: int, length: int) -> str:
    text = (ROOT / "data" / "contracts" / f"{contract_id}.txt").read_text(encoding="utf-8", errors="ignore")
    return " ".join(text[start:start + length].split())


def s01() -> str:
    return """
<div style="margin-top:120px">
  <div class="kicker">AI engineering portfolio · project 1 of 6</div>
  <h1 style="font-size:92px">Reading contracts like a lawyer<br><span class="dim">and measuring how well it does.</span></h1>
  <div class="lede" style="font-size:34px;margin-top:36px">Document intelligence pipeline &nbsp;·&nbsp; Gemini &nbsp;·&nbsp; LangChain &nbsp;·&nbsp; 510 lawyer-labelled contracts</div>
  <div style="margin-top:70px;font-size:40px;font-weight:600">Abhisheka U</div>
</div>"""


def s02() -> str:
    rec = json.loads((RESULTS / "pypdf_low_v4" / "c015.json").read_text(encoding="utf-8"))["record"]
    shown = {
        "document_name": rec["document_name"]["value"],
        "parties": rec["parties"]["value"],
        "effective_date": rec["effective_date"]["value"],
        "expiration_date": rec["expiration_date"]["value"],
        "governing_law": rec["governing_law"]["value"],
        "renewal_term": rec["renewal_term"]["value"],
        "exclusivity": rec["exclusivity"]["present"],
        "cap_on_liability": rec["cap_on_liability"]["present"],
    }
    lines = []
    for k, v in shown.items():
        val = html.escape(json.dumps(v))
        color = "#3987e5" if isinstance(v, bool) else "#f0efec"
        lines.append(f'&nbsp;&nbsp;<span style="color:#c3c2b7">"{k}"</span>: <span style="color:{color}">{val}</span>,')
    lines.append('&nbsp;&nbsp;<span style="color:#898781">… 9 more, each with a source quote</span>')
    excerpt = html.escape(contract_excerpt("c015", 0, 520))
    return f"""
<div class="kicker">The problem</div>
<h1>Contract PDF in. <span class="dim">Structured, checkable data out.</span></h1>
<div class="row" style="margin-top:44px;height:560px">
  <div class="card" style="flex:1;background:#f4f2ec;color:#2b2b2b;overflow:hidden">
    <div class="label" style="color:#7a776f;margin-bottom:14px">c015.pdf · distributor agreement</div>
    <div class="quote" style="color:#2b2b2b;font-size:24px">{excerpt} …</div>
  </div>
  <div class="arrow">→</div>
  <div class="card mono" style="flex:1.15;font-size:25px;line-height:1.75">{{<br>{"<br>".join(lines)}<br>}}</div>
</div>"""


def s03() -> str:
    def box(title, sub, accent=False):
        border = "border-color:#3987e5" if accent else ""
        return (f'<div class="card" style="flex:1;padding:26px 24px;{border}">'
                f'<div style="font-size:30px;font-weight:700">{title}</div>'
                f'<div style="font-size:22px;color:#c3c2b7;margin-top:10px;line-height:1.4">{sub}</div></div>')
    arrow = '<div class="arrow" style="font-size:40px">→</div>'
    flow = arrow.join([
        box("PDF", "contract as filed"),
        box("Parser", "pypdf text, or the PDF itself"),
        box("Gemini", "structured output: 17 fields, each with a quote", accent=True),
        box("Derive", "dates computed in code"),
        box("Validate", "formats + is every quote really in the text?"),
        box("JSON", "validated record"),
    ])
    return f"""
<div class="kicker">Architecture</div>
<h1>Every value has to prove where it came from.</h1>
<div class="row" style="margin-top:60px;gap:14px">{flow}</div>
<div class="card" style="margin-top:44px;display:flex;gap:40px;align-items:center">
  <div class="label" style="white-space:nowrap">Eval loop</div>
  <div class="mid">Score against lawyers' labels &nbsp;→&nbsp; precision / recall per field &nbsp;→&nbsp; read every error &nbsp;→&nbsp; fix the cause &nbsp;→&nbsp; re-test on unseen contracts</div>
</div>"""


def s04() -> str:
    def tile(big, label, sub):
        return (f'<div class="card" style="flex:1"><div class="big">{big}</div>'
                f'<div style="font-size:28px;font-weight:600;margin-top:18px">{label}</div>'
                f'<div class="mid" style="font-size:24px;margin-top:8px">{sub}</div></div>')
    return f"""
<div class="kicker">The benchmark</div>
<h1>CUAD: real contracts, <span class="dim">labelled by lawyers.</span></h1>
<div class="row" style="margin-top:56px">
  {tile("510", "SEC-filed contracts", "licences, distribution, JV, supply…")}
  {tile("17", "fields scored", "8 key terms + 9 clause types")}
  {tile("3 × 25", "contract samples", "2 for tuning, 1 never seen until the final test")}
</div>
<div class="row" style="margin-top:28px">
  <div class="card" style="flex:1"><div class="label">Precision</div><div class="mid" style="margin-top:10px">Of the fields it filled in, how many were right?</div></div>
  <div class="card" style="flex:1"><div class="label">Recall</div><div class="mid" style="margin-top:10px">Of the fields the contract has, how many did it find?</div></div>
</div>"""


def s05() -> str:
    return f"""
<div class="kicker">Iterating on evidence</div>
<h1>Five prompt versions, <span class="dim">each one driven by the errors.</span></h1>
<div class="row" style="margin-top:36px;align-items:flex-start">
  <div class="card" style="padding:28px 24px 10px">
    <div class="label" style="margin-left:56px">F1 by prompt version · tuning sample (25 contracts)</div>
    <div style="display:flex;gap:28px;margin:12px 0 0 56px;font-size:22px;color:#c3c2b7">
      <span><span style="display:inline-block;width:28px;height:4px;background:#3987e5;vertical-align:middle;margin-right:8px"></span>Clauses</span>
      <span><span style="display:inline-block;width:28px;height:4px;background:#d95926;vertical-align:middle;margin-right:8px"></span>Key terms</span>
    </div>
    {line_chart()}
  </div>
  <div class="card" style="flex:1;border-color:#3987e5">
    <div class="label">Final test · v5</div>
    <div class="mid" style="font-size:24px;margin-top:6px">25 contracts it had never seen</div>
    <div class="big" style="margin-top:36px">{pct(FINAL["overall"]["flags"]["f1"])}</div>
    <div style="font-size:26px;color:#c3c2b7;margin-top:6px">clauses F1</div>
    <div class="big" style="margin-top:30px">{pct(FINAL["overall"]["values"]["f1"])}</div>
    <div style="font-size:26px;color:#c3c2b7;margin-top:6px">key terms F1</div>
  </div>
</div>"""


def s06() -> str:
    low = TUNING[0][2]
    return f"""
<div class="kicker">Lesson 1 · thinking level</div>
<h1>3× faster, <span class="dim">at almost the same accuracy.</span></h1>
<div class="row" style="margin-top:44px">
  {bar_pair("Seconds per contract", BASE_HIGH["avg_latency_s"], low["avg_latency_s"], lambda v: f"{v:.1f}s")}
  {bar_pair("Output tokens per contract", BASE_HIGH["avg_output_tokens"], low["avg_output_tokens"], lambda v: f"{v:,.0f}")}
  <div class="card" style="flex:0.8">
    <div class="label">Accuracy (F1)</div>
    <div class="mid" style="margin-top:28px">Key terms<br><span style="font-size:44px;color:#f0efec;font-weight:700">{pct(BASE_HIGH["overall"]["values"]["f1"])} → {pct(low["overall"]["values"]["f1"])}</span></div>
    <div class="mid" style="margin-top:28px">Clauses<br><span style="font-size:44px;color:#f0efec;font-weight:700">{pct(BASE_HIGH["overall"]["flags"]["f1"])} → {pct(low["overall"]["flags"]["f1"])}</span></div>
    <div class="mid" style="font-size:22px;margin-top:28px">Reading terms is lookup, not reasoning.</div>
  </div>
</div>"""


def s07() -> str:
    before = TUNING[0][2]["per_field"]["expiration_date"]["recall"]
    after = TUNING[1][2]["per_field"]["expiration_date"]["recall"]
    return f"""
<div class="kicker">Lesson 2 · don't make the LLM do arithmetic</div>
<h1>The model reads. <span class="dim">Code calculates.</span></h1>
<div class="row" style="margin-top:44px;align-items:center">
  <div class="card" style="flex:1.2">
    <div class="quote">"…shall continue in effect for a period of <mark>Three (3) years</mark>."</div>
    <div class="mono" style="font-size:26px;margin-top:30px;line-height:1.9;color:#c3c2b7">
      initial_term &nbsp;&nbsp;= <span style="color:#f0efec">"3 years"</span> &nbsp;<span class="pill">model</span><br>
      effective_date = <span style="color:#f0efec">"2019-08-01"</span> &nbsp;<span class="pill">model</span><br>
      expiration &nbsp;&nbsp;&nbsp;= <span style="color:#3987e5">2019-08-01 + 3 years</span><br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:#f0efec;font-weight:700">"2022-08-01"</span> &nbsp;<span class="pill">code</span>
    </div>
  </div>
  <div class="card" style="flex:0.8;text-align:center">
    <div class="label">Expiration-date recall</div>
    <div style="font-size:120px;font-weight:700;margin-top:20px"><span class="from" style="font-size:72px">{pct(before)}</span> {pct(after)}</div>
    <div class="mid" style="font-size:22px;margin-top:10px">Same 25 contracts, before → after</div>
  </div>
</div>"""


def s08() -> str:
    v2 = TUNING[1][2]["per_field"]["cap_on_liability"]["recall"]
    v3 = TUNING[2][2]["per_field"]["cap_on_liability"]["recall"]
    return f"""
<div class="kicker">Lesson 3 · check the model can see your instructions</div>
<h1>My field definitions <span class="dim">were never being read.</span></h1>
<div class="row" style="margin-top:44px">
  <div class="card mono" style="flex:1;font-size:23px;line-height:1.7">
    <div class="label" style="font-family:'Segoe UI';margin-bottom:16px">Before · JSON schema</div>
    "cap_on_liability": {{<br>
    &nbsp;&nbsp;"$ref": "#/$defs/ClauseFlag",<br>
    &nbsp;&nbsp;<span style="color:#5d5c58">"description": "…excluding consequential damages counts as a cap…"</span><br>}}
    <div style="font-family:'Segoe UI';font-size:22px;color:#ec835a;margin-top:18px">⚠ text beside a $ref is ignored</div>
  </div>
  <div class="arrow">→</div>
  <div class="card mono" style="flex:1;font-size:23px;line-height:1.7">
    <div class="label" style="font-family:'Segoe UI';margin-bottom:16px">After · system prompt</div>
    Field definitions:<br>
    - cap_on_liability: any limit on liability, <span style="color:#f0efec">including clauses excluding consequential, indirect or punitive damages</span>
  </div>
  <div class="card" style="flex:0.7;text-align:center">
    <div class="label">Cap-on-liability recall</div>
    <div style="font-size:110px;font-weight:700;margin-top:24px;line-height:1.05"><span class="from" style="font-size:64px">{pct(v2)}</span><br>{pct(v3)}</div>
  </div>
</div>"""


def s09() -> str:
    rates = [s["grounding_rate"] for s in [*(t[2] for t in TUNING), *TEST.values()]]
    excerpt = html.escape(contract_excerpt("c015", 0, 10**6))
    clause = "This Agreement is a contract under the laws of the State of Florida"
    i = excerpt.find(clause)
    snippet = excerpt[max(0, i - 160):i] + f"<mark>{clause}</mark>" + excerpt[i + len(clause):i + len(clause) + 140]
    return f"""
<div class="kicker">Hallucination control</div>
<h1>Every quote is checked <span class="dim">against the source.</span></h1>
<div class="row" style="margin-top:44px">
  <div class="card" style="flex:1.4">
    <div class="label" style="margin-bottom:16px">governing_law = "Florida" · evidence found in c015</div>
    <div class="quote">…{snippet}…</div>
    <div style="font-size:26px;margin-top:26px" class="good">✓ grounded</div>
  </div>
  <div class="card" style="flex:0.8">
    <div class="big">{pct(min(rates))}+</div>
    <div class="mid" style="margin-top:12px">of evidence quotes verified, in every run</div>
    <div class="big" style="margin-top:44px">{FINAL["error_types"]["label_gap?"]}</div>
    <div class="mid" style="margin-top:12px">"errors" on the test set were values with a verified quote that CUAD left unlabelled</div>
  </div>
</div>"""


def s10() -> str:
    rows = [
        ("Clean text (CUAD)", "upper bound", TEST["cuad_txt"]),
        ("pypdf text", "before fix", TEST["pypdf_nbsp"]),
        ("pypdf text", "normalized", TEST["pypdf"]),
        ("Gemini reads the PDF", "multimodal", TEST["native_pdf"]),
    ]
    body = "".join(
        f'<tr><td>{name} <span class="pill" style="font-size:18px;margin-left:8px">{tag}</span></td>'
        f'<td class="n">{pct(s["overall"]["values"]["f1"])}</td><td class="n">{pct(s["overall"]["flags"]["f1"])}</td>'
        f'<td class="n">{s["avg_latency_s"]:.1f}s</td><td class="n" style="color:#f0efec">{s["avg_input_tokens"]:,.0f}</td></tr>'
        for name, tag, s in rows
    )
    saved = 1 - TEST["pypdf"]["avg_input_tokens"] / TEST["pypdf_nbsp"]["avg_input_tokens"]
    return f"""
<div class="kicker">Parser comparison · test sample</div>
<h1>Same accuracy. <span class="dim">Very different token bills.</span></h1>
<div class="row" style="margin-top:40px">
  <div class="card" style="flex:1.5">
    <table><tr><th>Parser</th><th class="n">Key terms</th><th class="n">Clauses</th><th class="n">Latency</th><th class="n">Tokens in</th></tr>{body}</table>
  </div>
  <div class="card" style="flex:0.75;border-color:#d95926">
    <div class="label">The hidden bug</div>
    <div class="mid" style="margin-top:14px">pypdf put a <span class="mono" style="color:#f0efec">U+00A0</span> no-break space between every word.</div>
    <div class="mid" style="margin-top:12px">Invisible, and 2–3× the tokens.</div>
    <div class="big" style="margin-top:26px">−{round(saved * 100)}%</div>
    <div class="mid" style="font-size:24px;margin-top:8px">input tokens after one line of Unicode normalization</div>
  </div>
</div>"""


def s11() -> str:
    # Crop to the part of the app that matters (title, metrics, table), so the
    # screenshot stays clear of the subtitle band at the bottom of the slide.
    from PIL import Image
    cropped = BUILD / "demo_cropped.png"
    Image.open(BUILD / "demo.png").crop((0, 70, 1600, 770)).save(cropped)
    return f"""
<div class="kicker">Demo</div>
<h1>Upload a contract. <span class="dim">Get validated JSON.</span></h1>
<div style="margin-top:34px;display:flex;justify-content:center">
  <img src="{cropped.as_uri()}" style="width:1240px;border-radius:16px;border:1px solid rgba(255,255,255,0.18);box-shadow:0 30px 80px rgba(0,0,0,0.6)">
</div>"""


def s12() -> str:
    def tile(big, label):
        return (f'<div class="card" style="flex:1;text-align:center"><div class="big">{big}</div>'
                f'<div class="mid" style="font-size:26px;margin-top:14px">{label}</div></div>')
    rates = [s["grounding_rate"] for s in TEST.values()]
    return f"""
<div class="kicker">Results · 25 unseen contracts</div>
<h1>Measured, not guessed.</h1>
<div class="row" style="margin-top:48px">
  {tile(pct(FINAL["overall"]["values"]["f1"]), "key terms F1")}
  {tile(pct(FINAL["overall"]["flags"]["f1"]), "clauses F1")}
  {tile(pct(min(rates)) + "+", "quotes verified")}
  {tile(f'{FINAL["avg_latency_s"]:.1f}s', "per contract")}
</div>
<div class="card" style="margin-top:36px;display:flex;justify-content:space-between;align-items:center">
  <div><div class="label">Code + full evaluation</div><div style="font-size:40px;font-weight:600;margin-top:8px">{GITHUB}</div></div>
  <div style="text-align:right"><div class="label">Next · project 2</div><div style="font-size:32px;margin-top:8px;color:#c3c2b7">Enterprise search with RAG</div></div>
</div>
<div style="margin-top:30px;font-size:30px;font-weight:600">{AUTHOR}</div>"""


RENDERERS = [s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12]


def render_all() -> None:
    assert len(RENDERERS) == len(SLIDES), "one renderer per slide in narration.py"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_dir = BUILD / "html"
    html_dir.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page_ = browser.new_page(viewport={"width": 1920, "height": 1080})
        for i, (slide, render) in enumerate(zip(SLIDES, RENDERERS), 1):
            # Written to disk (not set_content) so the demo slide's file:// image loads.
            path = html_dir / f"{slide.id}.html"
            path.write_text(page(i, render()), encoding="utf-8")
            page_.goto(path.as_uri())
            page_.wait_for_load_state("networkidle")
            page_.screenshot(path=OUT_DIR / f"{slide.id}.png")
            print(f"rendered {slide.id}")
        browser.close()


if __name__ == "__main__":
    render_all()
