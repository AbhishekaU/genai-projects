"""
The four specialist agents for the Smart Event Planning Assistant:
Venue, Vendor, Budget, and Schedule.

Built with LangChain's create_agent -- the current (LangChain 1.x) agent
constructor. The older create_react_agent + AgentExecutor combo (used in
Langchain/Langchainnew.py) was moved out of core langchain into a separate
legacy package (langchain_classic) as of LangChain 1.0; create_agent is now
the standard way to build a tool-using agent, and it's simpler too -- no
hand-written ReAct prompt template needed, just a model, some tools, and a
plain system prompt.

Every agent is exposed as a plain function with the same shape:
    run_x_agent(context: str) -> str
so orchestrator.py can call any of them the same way.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import TOOLS_BUDGET, TOOLS_VENDOR, TOOLS_VENUE
from utils import extract_text

load_dotenv()

# =====================================================
# MODEL
# =====================================================
# Slightly creative (temperature=0.3) since these agents are writing
# recommendations in natural language, not making a strict routing decision.

specialist_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    temperature=0.3,
)


def _run_agent(executor, context: str) -> str:
    """
    Shared helper: send `context` in as a user message and return the
    agent's final reply text. create_agent expects/returns the same
    {"messages": [...]} shape regardless of which agent it is.
    """
    result = executor.invoke({"messages": [{"role": "user", "content": context}]})
    return extract_text(result["messages"][-1].content)


# =====================================================
# VENUE AGENT
# =====================================================

VENUE_ROLE = """
You are the Venue Agent for an event planning system.

Your job: given details about an event (type, city, budget, number of
attendees), use the search_venues tool to find 2-3 suitable venues and
recommend them with a short reason each.

Only handle venues. Do not discuss budget breakdowns or vendors.
"""

venue_agent = create_agent(model=specialist_llm, tools=TOOLS_VENUE, system_prompt=VENUE_ROLE)


def run_venue_agent(context: str) -> str:
    return _run_agent(venue_agent, context)


# =====================================================
# VENDOR AGENT
# =====================================================

VENDOR_ROLE = """
You are the Vendor Agent for an event planning system.

Your job: given details about an event, use the search_vendors tool to find
vendors for the categories that are relevant (catering, decoration,
photography, entertainment) and recommend them briefly.

Only handle vendors. Do not discuss venue or budget.
"""

vendor_agent = create_agent(model=specialist_llm, tools=TOOLS_VENDOR, system_prompt=VENDOR_ROLE)


def run_vendor_agent(context: str) -> str:
    return _run_agent(vendor_agent, context)


# =====================================================
# BUDGET AGENT
# =====================================================

BUDGET_ROLE = """
You are the Budget Agent for an event planning system.

Your job: given details about an event -- including any venue or vendor
costs already known -- build a cost breakdown and total using the
calculate_budget tool. Always use the tool for arithmetic, never add numbers
yourself. If the budget looks tight, suggest 1-2 cost-optimization ideas.

Only handle budget. Do not discuss venue or vendor choices.
"""

budget_agent = create_agent(model=specialist_llm, tools=TOOLS_BUDGET, system_prompt=BUDGET_ROLE)


def run_budget_agent(context: str) -> str:
    return _run_agent(budget_agent, context)


# =====================================================
# SCHEDULE AGENT
# =====================================================
# No tools -- passing tools=None makes create_agent skip the tool-calling
# loop entirely and build a single model-call node instead. Same underlying
# function as the other three, just with nothing to call.

SCHEDULE_ROLE = """
You are the Schedule Agent for an event planning system.

Given the event details you're told (including the venue and attendee count,
if known), write a simple day-of timeline -- from setup through to closing --
with approximate times. You have no tools; reason only from what's given.
"""

schedule_agent = create_agent(model=specialist_llm, tools=None, system_prompt=SCHEDULE_ROLE)


def run_schedule_agent(context: str) -> str:
    return _run_agent(schedule_agent, context)
