"""
Streamlit web UI for the Smart Event Planning Assistant.

This file knows nothing about LangChain -- it only calls run_event_planner()
from orchestrator.py and displays whatever comes back. That separation is
deliberate: the whole multi-agent system behaves identically whether it's
driven from this web page or from orchestrator.py's own terminal test.

Visual styling comes from .streamlit/config.toml (colors, fonts, radius) --
not custom CSS -- so it stays clean and won't break on Streamlit upgrades.
"""

import streamlit as st

from orchestrator import run_event_planner

st.set_page_config(
    page_title="Smart Event Planning Assistant",
    page_icon="🎉",
    layout="centered",
)

# Session state initialized once, in one place, so it's easy to see everything
# this page remembers across reruns.
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "event_summary" not in st.session_state:
    st.session_state.event_summary = None

st.title(":material/celebration: Smart event planning assistant")
st.caption(
    "A LangChain multi-agent system: a Coordinator routes work to "
    "Venue, Budget, Vendor, and Schedule agents, then merges their "
    "output into one event plan."
)

# The sidebar holds the input form -- it's the "settings" that drive what
# the main area shows, so the main area stays free for the actual plan.
with st.sidebar:
    st.subheader("Event details")

    with st.form("event_form"):
        event_type = st.text_input(
            "Event type", placeholder="e.g. birthday, wedding, corporate offsite"
        )
        city = st.text_input("City", placeholder="e.g. Bangalore")
        attendees = st.number_input("Number of attendees", min_value=1, value=30)
        budget = st.number_input("Budget (Rs.)", min_value=0, value=50000, step=5000)
        preferences = st.text_area(
            "Preferences",
            placeholder="e.g. outdoor venue, vegetarian catering, needs a DJ",
        )
        submitted = st.form_submit_button(
            "Plan my event", icon=":material/auto_awesome:", type="primary"
        )

    st.caption("Built with LangChain + Gemini")

if submitted:
    if not event_type or not city:
        st.error("Please fill in at least the event type and city.")
    else:
        user_request = (
            f"Event type: {event_type}\n"
            f"City: {city}\n"
            f"Attendees: {attendees}\n"
            f"Budget: Rs. {budget:,.0f}\n"
            f"Preferences: {preferences or 'none specified'}"
        )

        with st.status("Planning your event...", expanded=True) as status:

            def on_step(step, specialist_name, result):
                # Rendered inside the status box above, since we're still
                # inside its "with" block while the orchestrator runs.
                st.badge(
                    f"{specialist_name.title()} agent finished",
                    icon=":material/check_circle:",
                    color="green",
                )

            final_state = run_event_planner(user_request, on_step=on_step)
            status.update(label="Event plan ready", state="complete")

        st.session_state.final_state = final_state
        st.session_state.event_summary = {
            "city": city,
            "attendees": attendees,
            "budget": budget,
        }

final_state = st.session_state.final_state

if final_state is None:
    st.info(
        "Fill in the event details in the sidebar and click **Plan my event** "
        "to get started.",
        icon=":material/arrow_back:",
    )
else:
    summary = st.session_state.event_summary

    cols = st.columns(3)
    cols[0].metric("City", summary["city"])
    cols[1].metric("Attendees", summary["attendees"])
    cols[2].metric("Budget", f"Rs. {summary['budget']:,.0f}")

    with st.container(border=True):
        st.markdown(final_state.final_report)

    st.download_button(
        "Download plan",
        data=final_state.final_report,
        file_name="event_plan.md",
        icon=":material/download:",
    )

    with st.expander("See what each agent did", icon=":material/history:"):
        for note in final_state.history:
            st.caption(note)
