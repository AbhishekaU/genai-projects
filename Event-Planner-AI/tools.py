"""
Tools for the Smart Event Planning Assistant.

Each tool is a plain Python function decorated with @tool, which is what lets a
LangChain agent "see" it and decide when to call it. The docstring on each tool
is not just documentation for humans -- it's the instruction the LLM reads to
decide whether a tool fits the current task, so it's written for the model.
"""

import os

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"


def _serper_search(query: str) -> str:
    """
    Shared helper (not a tool itself) that calls Serper's Google Search API.
    Falls back to a clearly-labeled mock result if no SERPER_API_KEY is set,
    so the whole system still runs end to end without a key.
    """
    if not SERPER_API_KEY:
        return f"[MOCK RESULT - no SERPER_API_KEY set] Searched for: {query}"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query}

    try:
        response = requests.post(SERPER_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"Search failed: {e}"

    organic = data.get("organic", [])[:5]
    if not organic:
        return "No results found."

    lines = []
    for item in organic:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        lines.append(f"- {title}: {snippet} ({link})")

    return "\n".join(lines)


@tool
def search_venues(query: str) -> str:
    """
    Search for event venues. Pass a query describing the event type, city,
    number of attendees, and budget -- for example:
    "wedding venues in Bangalore for 100 guests under 5 lakh rupees".
    Returns a short list of matching venues with links.
    """
    return _serper_search(f"{query} venue")


@tool
def search_vendors(query: str) -> str:
    """
    Search for event vendors such as catering, decoration, photography, or
    entertainment. Pass a query describing the vendor type and city -- for
    example: "birthday party catering services in Bangalore".
    Returns a short list of matching vendors with links.
    """
    return _serper_search(query)


@tool
def calculate_budget(costs: str) -> str:
    """
    Calculate a total budget breakdown. Pass costs as comma-separated
    "label:amount" pairs -- for example:
    "venue:20000,catering:15000,decoration:5000".
    Returns a line-by-line breakdown plus the total. Always use this tool for
    any arithmetic instead of computing totals yourself.
    """
    try:
        items = []
        for part in costs.split(","):
            label, amount = part.split(":")
            items.append((label.strip(), float(amount.strip())))
    except Exception:
        return (
            "Could not parse costs. Please provide them as comma-separated "
            "label:amount pairs, e.g. venue:20000,catering:15000"
        )

    total = sum(amount for _, amount in items)
    lines = [f"- {label}: Rs. {amount:,.0f}" for label, amount in items]
    lines.append(f"Total: Rs. {total:,.0f}")
    return "\n".join(lines)


# One tool list per specialist agent, ready to import in agents.py.
# The Schedule Agent has none -- it works purely by reasoning over
# what's already in the shared state (venue, attendee count, etc.).
TOOLS_VENUE = [search_venues]
TOOLS_VENDOR = [search_vendors]
TOOLS_BUDGET = [calculate_budget]
