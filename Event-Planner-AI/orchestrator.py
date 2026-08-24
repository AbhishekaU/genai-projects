"""
The orchestration loop for the Smart Event Planning Assistant.

This is the hand-written equivalent of what a LangGraph StateGraph would do
automatically for you: repeatedly ask the Coordinator what's next, run that
specialist, update the shared state, and stop once the Coordinator says
"done" (or a safety cap is hit).
"""

from agents import run_budget_agent, run_schedule_agent, run_vendor_agent, run_venue_agent
from state import EventState
from supervisor import decide_next, generate_final_report

# Safety cap: if the Coordinator somehow never says "done" (e.g. it keeps
# re-picking the same specialist), the loop still has to stop eventually.
MAX_STEPS = 8

# Maps the Coordinator's routing word to the function that actually does the work.
SPECIALISTS = {
    "venue": run_venue_agent,
    "vendor": run_vendor_agent,
    "budget": run_budget_agent,
    "schedule": run_schedule_agent,
}

# Maps the same routing word to which EventState field that specialist owns.
STATE_FIELD = {
    "venue": "venue_info",
    "vendor": "vendor_info",
    "budget": "budget_info",
    "schedule": "schedule_info",
}


def run_event_planner(user_request: str, on_step=None) -> EventState:
    """
    Run the full multi-agent event-planning workflow for one user request.

    on_step, if given, is called as on_step(step_number, specialist_name, result)
    right after each specialist finishes -- this is how app.py will show live
    progress in the Streamlit UI without this file needing to know anything
    about Streamlit.
    """
    state = EventState(user_request=user_request)

    for step in range(1, MAX_STEPS + 1):
        decision = decide_next(state)

        if decision.next == "done":
            state.record(f"Step {step}: Coordinator finished -- {decision.reason}")
            break

        specialist_name = decision.next
        run_specialist = SPECIALISTS[specialist_name]
        field_name = STATE_FIELD[specialist_name]

        result = run_specialist(state.known_info())
        setattr(state, field_name, result)
        state.record(f"Step {step}: {specialist_name} agent ran -- {decision.reason}")

        if on_step:
            on_step(step, specialist_name, result)

    else:
        # The for-loop finished all MAX_STEPS without hitting the "break" above,
        # meaning the Coordinator never said "done". Python's for/else runs this
        # block only in that case -- it's skipped entirely if we broke out early.
        state.record(f"Stopped after {MAX_STEPS} steps without an explicit 'done'.")

    state.final_report = generate_final_report(state)
    return state


if __name__ == "__main__":
    # Quick terminal smoke test while app.py doesn't exist yet.
    # app.py (next file) is the real, intended entry point.
    query = input("Describe the event you want to plan: ")
    final_state = run_event_planner(query)

    print("\n" + "=" * 80)
    print("FINAL EVENT PLAN")
    print("=" * 80)
    print(final_state.final_report)
    print("\n--- Trace ---")
    for note in final_state.history:
        print("-", note)
