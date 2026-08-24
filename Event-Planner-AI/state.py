"""
Shared state for the Smart Event Planning Assistant.

This is a plain Python object (no LangChain/LangGraph machinery) passed by hand
between the Coordinator and the specialist agents. Every specialist reads what it
needs from an EventState and writes back into its own field only.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EventState:
    # The original ask. Set once, never overwritten.
    user_request: str

    # Audit trail of what has run so far, e.g. "Venue Agent ran".
    # Used for debugging, for the Coordinator's own reasoning, and later
    # for showing a live trace in the Streamlit UI.
    history: List[str] = field(default_factory=list)

    # Each specialist owns exactly one of these fields. They start out
    # unset (None) and the Coordinator uses that to know who still needs to run.
    venue_info: Optional[str] = None
    budget_info: Optional[str] = None
    vendor_info: Optional[str] = None
    schedule_info: Optional[str] = None

    # Filled in only once the Coordinator decides the plan is complete.
    final_report: Optional[str] = None

    def record(self, note: str) -> None:
        """Append a note to the history trail."""
        self.history.append(note)

    def missing_fields(self) -> List[str]:
        """Which specialist outputs haven't been produced yet."""
        missing = []
        if self.venue_info is None:
            missing.append("venue")
        if self.budget_info is None:
            missing.append("budget")
        if self.vendor_info is None:
            missing.append("vendor")
        if self.schedule_info is None:
            missing.append("schedule")
        return missing

    def known_info(self) -> str:
        """
        Render everything gathered so far as one text block, so it can be
        dropped straight into a prompt for the Coordinator or a specialist.
        """
        lines = [f"User request: {self.user_request}"]

        if self.venue_info:
            lines.append(f"\nVenue info so far:\n{self.venue_info}")
        if self.budget_info:
            lines.append(f"\nBudget info so far:\n{self.budget_info}")
        if self.vendor_info:
            lines.append(f"\nVendor info so far:\n{self.vendor_info}")
        if self.schedule_info:
            lines.append(f"\nSchedule info so far:\n{self.schedule_info}")

        if len(lines) == 1:
            lines.append("\n(Nothing gathered yet.)")

        return "\n".join(lines)
