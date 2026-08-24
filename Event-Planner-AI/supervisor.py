"""
The Coordinator for the Smart Event Planning Assistant.

The Coordinator never does planning work itself. Its only two jobs are:
1. Look at what's known so far and decide which specialist should run next
   (or whether enough is known to stop).
2. Once done, combine everyone's work into one final report.

Unlike multi-agent-demo/workflows/supervisor_flow.py, which asked the LLM for
a plain-text answer and then guessed at its meaning with string matching
("travel" in decision.lower()), this Coordinator uses LangChain's structured
output: the LLM is forced to fill in a Pydantic model, so there is nothing to
mis-parse.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import EventState
from utils import extract_text

load_dotenv()

# Deterministic on purpose -- routing should be consistent given the same state.
coordinator_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    temperature=0,
)


class RoutingDecision(BaseModel):
    """The Coordinator's decision about what should happen next."""

    next: Literal["venue", "budget", "vendor", "schedule", "done"] = Field(
        description=(
            "Which specialist should run next, or 'done' if enough "
            "information has been gathered to write the final event plan."
        )
    )
    reason: str = Field(description="One short sentence explaining why.")


ROUTER_PROMPT = PromptTemplate.from_template("""
You are the Coordinator of an event planning system. You do not do any
planning work yourself -- you only decide which specialist should act next.

Specialists available:
- venue: finds suitable venues
- budget: builds a cost breakdown
- vendor: recommends catering, decoration, photography, entertainment
- schedule: builds the event day timeline

Rules:
- Only pick a specialist whose information is still missing.
- Schedule works best once the venue is known -- prefer venue before schedule
  if both are still missing.
- If a specialist genuinely isn't relevant to this request, you may treat its
  info as "not needed" and stop routing to it.
- Once everything relevant has been gathered, respond with "done".

Specialists still missing output: {missing}

What's known so far:
{known_info}

Decide what happens next.
""")

_router_chain = ROUTER_PROMPT | coordinator_llm.with_structured_output(RoutingDecision)


def decide_next(state: EventState) -> RoutingDecision:
    """Ask the Coordinator what should happen next, given the current state."""
    return _router_chain.invoke({
        "missing": ", ".join(state.missing_fields()) or "(none)",
        "known_info": state.known_info(),
    })


FINAL_REPORT_PROMPT = PromptTemplate.from_template("""
You are the Coordinator of an event planning system, writing the final event
plan for the customer by combining everyone's work below into one clear,
well-organized report. Do not invent new information -- only organize and
lightly summarize what's given.

{known_info}

Write the final Event Plan with these sections: Venue Recommendation,
Budget Summary, Vendor Suggestions, Event Schedule, Final Recommendations.
Skip a section if nothing was gathered for it.
""")

_final_report_chain = FINAL_REPORT_PROMPT | coordinator_llm


def generate_final_report(state: EventState) -> str:
    """Merge everything gathered so far into one final report."""
    response = _final_report_chain.invoke({"known_info": state.known_info()})
    return extract_text(response.content)
