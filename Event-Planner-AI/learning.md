# Learning Log — Smart Event Planning Assistant (LangChain)

This file is the running record of *what we learned and why*, one section per file,
in build order.

---

## 0. Before we start — where this project sits

You already learned (in `Langchain/`):
- A single ReAct agent with tools (`Langchainnew.py`)
- Memory bolted onto a single agent (`Shoppingagent/`)
- Multi-agent *patterns* hand-built without a framework (`multi-agent-demo/`)

This project is the missing piece: **a multi-agent system built with LangChain alone**
— no LangGraph yet. Shared state becomes a plain Python object instead of manually
stitched prompt strings, and routing becomes structured output instead of
`if "travel" in text.lower()`. Everything else (agents, tools, the loop) is code you
can read top to bottom.

---

## 1. `state.py` — the shared state object

**What it does:** Defines `EventState`, a plain Python `@dataclass` that acts like a
shared clipboard passed between the Coordinator and the four specialists. Each
specialist owns exactly one field (`venue_info`, `budget_info`, `vendor_info`,
`schedule_info`) and only ever writes to its own.

**Key concepts:**
- `@dataclass` — auto-generates `__init__`/`__repr__` for a class that's just typed
  data, so you don't hand-write boilerplate.
- Why a dataclass here and not a `dict`: a typo like `state["venu_info"]` fails
  silently with a dict (creates a new wrong key); `state.venu_info` on a dataclass
  is an immediate `AttributeError`.
- Why a dataclass here and not `TypedDict`/Pydantic: `TypedDict` is what LangGraph
  uses for its state (skipped — no LangGraph yet); Pydantic is reserved for data that
  needs *validation*, especially LLM output (used later in `supervisor.py`). Our own
  state is data we fully control, so plain dataclass is the right-sized tool.
- `field(default_factory=list)` for `history` — avoids a classic Python bug where a
  mutable default (`= []`) would be shared across every instance instead of each
  `EventState` getting its own empty list.

**Helper methods:** `record()` (append to the audit trail), `missing_fields()` (which
specialists haven't run yet — this is what the Coordinator uses to route), and
`known_info()` (renders everything gathered so far as one text block to drop into a
prompt).

---

## 2. `tools.py` — domain tools for the specialists

**What it does:** Defines three `@tool`-decorated functions the agents can call:
`search_venues`, `search_vendors` (both call Serper's search API, with a labeled
mock fallback if `SERPER_API_KEY` is missing), and `calculate_budget` (a
deterministic calculator — no LLM arithmetic).

**Key concepts:**
- **Tool Framework** — a tool is a plain function the LLM can *choose* to call. The
  `@tool` decorator is what makes a function visible to an agent at all.
- **The docstring is the instruction manual the LLM reads**, not just a comment for
  humans — it decides whether a tool fits the current situation and what to pass in.
  A vague docstring is the #1 cause of an agent picking the wrong tool.
- **Let code do arithmetic, not the LLM** — `calculate_budget` exists because LLMs
  are unreliable at exact math; the tool guarantees correct totals every time.
- Tool lists per agent (`TOOLS_VENUE`, `TOOLS_VENDOR`, `TOOLS_BUDGET`) — pre-packaged
  toolkits so each specialist only gets the gadgets relevant to its job.

---

## 3. `agents.py` — the four specialist agents

**What it does:** Builds Venue, Vendor, Budget (tool-using) and Schedule
(reasoning-only) as LangChain agents, each wrapped in a uniform
`run_x_agent(context: str) -> str` function so `orchestrator.py` can call any of them
identically.

**Key concepts:**
- **Model Abstraction** — `ChatGoogleGenerativeAI` from `langchain-google-genai` is
  the Gemini equivalent of `ChatOllama`; swapping providers is a one-line change.
- **`create_agent`** — LangChain 1.x's current agent constructor. Give it a model, a
  list of tools, and a plain `system_prompt`; it manages the "call a tool → look at
  the result → decide again" loop internally, using the model's native tool-calling.
- **`tools=None` skips the loop entirely** — the Schedule agent has nothing to call,
  so passing no tools makes `create_agent` build a single model-call node instead of
  a tool-calling loop. Same function, same shape, just nothing to call.
- **Uniform interface across different implementations** — every specialist exposes
  `run_x_agent(context) -> str`, so `orchestrator.py` never needs to know or care
  which one runs a loop and which one doesn't. This is also what makes "add a 5th
  agent later" a safe, additive change.

### ⚠️ Version lesson: why this file was rewritten once

The first version of `agents.py` used `create_react_agent` + `AgentExecutor` (the
same combo `Langchain/Langchainnew.py` uses) with a hand-written ReAct prompt
template. Running it crashed with:

```
ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'
```

**Why:** the installed `langchain` was version **1.3.16** (LangChain 1.x). As of
LangChain 1.0, the older `initialize_agent` / `create_react_agent` / `AgentExecutor`
agent-building style was moved out of the core `langchain` package into a separate
legacy compatibility package (`langchain_classic`), which wasn't installed. `langchain`
1.x's own `langchain.agents` module now only exposes the new `create_agent` API.

**The fix:** rewrite the file on `create_agent` instead of installing the legacy
package — this matches the original goal of using **current** LangChain concepts, and
it's simpler code too (no manual ReAct prompt template with `{tools}`/`{tool_names}`/
`{agent_scratchpad}` placeholders to maintain).

**The lesson to keep:** LangChain moves fast, and package/version mismatches between
what a tutorial shows and what `pip install` actually gives you are common and
expected — not a sign something is fundamentally wrong. When an import that "should"
exist doesn't, check the installed version (`pip show langchain`) before assuming the
code is broken.

### ⚠️ Bug lesson: `.content` isn't always a plain string

After the fix above, the app ran, but the final report rendered as a raw Python list
instead of text:

```
[{'type': 'text', 'text': '# Final Birthday Event Plan...', 'extras': {'signature': 'Etc6Ct...'}}]
```

**Why:** `.content` on a LangChain `AIMessage` is *usually* a plain string, so
`return response.content` normally just works. But Gemini's "thinking" models attach
a **thought signature** — an opaque token the model uses to stay internally
consistent across a multi-step turn — as extra metadata on the reply. When that
happens, `.content` becomes a **list of content blocks** instead of a string, e.g.
`[{"type": "text", "text": "...", "extras": {"signature": "..."}}]`, and printing that
list raw is exactly the "clumsy" Python-repr output that showed up.

**The fix:** added `utils.py` with one small helper, `extract_text(content)`, that
handles both shapes — returns a plain string as-is, or pulls just the `"text"` fields
out of a list of blocks and joins them. Every place that previously did
`response.content` (in `agents.py`'s `_run_agent` and `supervisor.py`'s
`generate_final_report`) now goes through `extract_text(...)` instead.

**The lesson to keep:** don't assume a model response field always has the shape you
first saw it have — richer/newer models often attach extra structure (signatures,
reasoning traces, citations) alongside the plain answer, and code that reads `.content`
directly should normalize it rather than assume "it's always a string."

---

## 4. `supervisor.py` — the Coordinator / router

**What it does:** Decides which specialist runs next (or whether the plan is done)
by asking an LLM to fill in a `RoutingDecision` object, then, at the end, merges
everything into one final report.

**Key concepts:**
- **Structured output** (`with_structured_output` + a Pydantic `BaseModel`) — forces
  the LLM's answer into a guaranteed shape (`next: Literal["venue", "budget", ...]`)
  instead of free text you'd have to guess the meaning of. This directly replaces the
  fragile `"travel" in decision.lower()` string-matching from
  `multi-agent-demo/workflows/supervisor_flow.py`.
- **Pydantic vs dataclass, revisited** — `RoutingDecision` uses Pydantic (not a
  dataclass like `EventState`) because this data comes from an LLM and needs
  validation; `Literal[...]` makes invalid answers structurally impossible.
- **LCEL pipe (`|`)** — `ROUTER_PROMPT | coordinator_llm.with_structured_output(...)`
  chains a prompt straight into a model call, same pipe idea from `langchain.md`.
- **Two LLM "personalities"** — `coordinator_llm` uses `temperature=0` (routing should
  be consistent) vs `specialist_llm` in `agents.py` at `temperature=0.3` (a little more
  natural language is fine for recommendations) — same pattern as `Hospital-AI`'s
  `manager_llm` vs `worker_llm`.
- **Guardrail prompting** — the final-report prompt explicitly says "do not invent new
  information, only organize what's given," to stop the LLM padding the report.

---

## 5. `orchestrator.py` — the hand-written orchestration loop

**What it does:** Runs the actual multi-agent workflow: repeatedly ask the Coordinator
what's next, run that specialist, write its result into `EventState`, and stop once
the Coordinator says `"done"` (or a safety cap is hit).

**Key concepts:**
- **Workflow Orchestration without a framework** — this is the plain-Python
  equivalent of what LangGraph's `StateGraph` would do automatically; useful to
  understand by hand before ever reaching for that abstraction.
- Two small dictionaries (`SPECIALISTS`, `STATE_FIELD`) translate the Coordinator's
  plain-word decision into "which function to call" and "which state field to write
  into" — this is only possible because every agent shares the same
  `run_x_agent(context) -> str` interface from `agents.py`.
- **`MAX_STEPS` safety cap** — an AI router could in theory never say "done"; a hard
  step limit guarantees the loop always terminates.
- **`for...else`** — the `else` on the `for` loop only runs if the loop finished
  *without* hitting `break`, i.e. exactly the "ran out of steps" case. A genuinely
  uncommon but useful bit of Python.
- **`on_step` callback** — the orchestrator has zero knowledge of Streamlit; it just
  optionally announces progress to whoever's listening, which is what lets `app.py`
  show live updates without this file depending on it.

---

## 6. `app.py` — Streamlit web UI

**What it does:** A web page wrapping the orchestrator: a form collects event details,
a live progress box shows each agent finishing, and the final report renders once
everything's done.

**Key concepts:**
- **`st.form`** — batches inputs so Streamlit only reruns the script once (on submit),
  not on every keystroke.
- **`st.status` + the `on_step` callback** — draws a live "in progress" box that gets
  updated as each specialist finishes, instead of freezing the page for the whole run.
- **`st.session_state`** — Streamlit reruns the *entire script* on every interaction
  (even unrelated ones, like opening an expander). Without saving `final_state` into
  `session_state`, it would disappear the moment the user does anything else on the
  page, since the `if submitted:` block only executes right after the button click.
- **Clean separation** — `app.py` calls `run_event_planner()` and displays whatever
  comes back; it has no idea LangChain exists. The whole backend works identically
  whether driven from this page or from `orchestrator.py`'s own terminal test.

### Visual redesign (post-build polish)

Once the app worked, it was upgraded from the plain single-column layout above to:
- **Theme via `.streamlit/config.toml`**, not custom CSS — a warm rose/coral palette,
  defined once as color tokens, safe across Streamlit upgrades.
- **Sidebar for the input form** — frees the main area entirely for results.
- **`st.badge` status checkmarks** inside `st.status` for live per-agent progress,
  instead of plain text lines.
- **`st.metric` glance row**, a bordered `st.container` around the report, a
  **download button**, and Material Symbols icons throughout instead of the
  plain-text/emoji-only version.

This is Streamlit polish, not a LangChain concept — noted here for completeness, not
as a core lesson.
