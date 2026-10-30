"""
Streamlit web UI for the Document Intelligence Pipeline.

Upload a contract PDF, pick how it's parsed, and get back the structured
record plus the validation report. Like the Event Planner's app.py, this
file holds no extraction logic of its own -- it only calls extract() and
validate(), so the web page and evaluate.py run exactly the same pipeline.
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

from extractor import MODEL_NAME, extract
from parsers import pypdf_text
from schema import FIELDS
from validate import validate

st.set_page_config(
    page_title="Contract intelligence",
    page_icon=":material/contract:",
    layout="wide",
)

if "result" not in st.session_state:
    st.session_state.result = None  # (Extraction, list[Issue], filename)

st.title(":material/contract: Contract intelligence")
st.caption(
    "Turns a commercial contract into a structured, validated JSON record. "
    f"Model: `{MODEL_NAME}`."
)

with st.sidebar:
    st.subheader("Input")
    uploaded = st.file_uploader("Contract PDF", type=["pdf"])
    parser = st.segmented_control(
        "Parser",
        ["pypdf", "native_pdf"],
        default="pypdf",
        help="pypdf: extract the PDF's text layer, then send text (cheaper). "
        "native_pdf: send the PDF itself and let Gemini read the pages (handles scans).",
    )
    run = st.button(
        "Extract",
        icon=":material/play_arrow:",
        type="primary",
        width="stretch",
        disabled=uploaded is None or parser is None,
    )

if run:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "contract.pdf"
        pdf_path.write_bytes(uploaded.getvalue())
        with st.spinner("Reading the contract..."):
            ex = extract(pdf_path, parser=parser)
            # Quotes are checked against the PDF's text layer. For a scanned
            # PDF that's empty, so every quote would show as ungrounded.
            issues = validate(ex.record, pypdf_text(pdf_path)) if ex.record else []
    st.session_state.result = (ex, issues, uploaded.name)

if st.session_state.result is None:
    st.info("Upload a contract PDF in the sidebar and click **Extract**.", icon=":material/upload_file:")
    st.stop()

ex, issues, filename = st.session_state.result
if ex.error or ex.record is None:
    st.error(ex.error or "The model returned no record.", title="Extraction failed")
    st.stop()

record = ex.record.model_dump()
issues_by_field = {}
for issue in issues:
    issues_by_field.setdefault(issue.field, []).append(issue)

with st.container(horizontal=True):
    st.metric("Latency", f"{ex.latency_s:.1f}s", border=True)
    st.metric("Tokens in", f"{ex.input_tokens:,}", border=True)
    st.metric("Tokens out", f"{ex.output_tokens:,}", border=True)
    st.metric("Validation issues", len(issues), border=True)

if issues:
    st.warning(
        "\n".join(f"- **{i.field}** ({i.kind}): {i.message}" for i in issues),
        title="Some values need a second look",
    )


def status(name: str) -> str:
    # Plain characters: dataframe cells don't render :material/...: icon shortcodes.
    return "⚠ check" if name in issues_by_field else "✓ grounded"


values_tab, clauses_tab, json_tab = st.tabs(["Key terms", "Clauses", "JSON"])

with values_tab:
    rows = []
    for spec in FIELDS:
        if spec.kind == "flag":
            continue
        field = record[spec.name]
        value = ", ".join(field["value"]) if spec.kind == "parties" else field["value"]
        rows.append({
            "Field": spec.name.replace("_", " "),
            "Value": value or "—",
            "Evidence": field["evidence"] or "",
            "Check": status(spec.name) if value else "",
        })
    st.dataframe(rows, hide_index=True, alt="Extracted key terms with supporting quotes")

with clauses_tab:
    rows = [
        {
            "Clause": spec.name.replace("_", " "),
            "Present": record[spec.name]["present"],
            "Evidence": record[spec.name]["evidence"] or "",
            "Check": status(spec.name) if record[spec.name]["present"] else "",
        }
        for spec in FIELDS if spec.kind == "flag"
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={"Present": st.column_config.CheckboxColumn()},
        alt="Clause types found in the contract",
    )

with json_tab:
    st.json(record)
    st.download_button(
        "Download JSON",
        data=json.dumps(record, indent=2),
        file_name=f"{Path(filename).stem}.json",
        mime="application/json",
        icon=":material/download:",
    )
