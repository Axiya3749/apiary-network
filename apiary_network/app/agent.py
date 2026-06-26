"""The Backyard Apiary Network -- an ADK multi-agent workflow.

The graph
---------
                                   ┌──────────────┐
   user query (+ optional photo) ─►  save_query   │
                                   └──────┬───────┘
                          ┌───────────────┴───────────────┐
                     'proceed'                    'blocked_injection'
                          │                               │
                    triage_agent                   security_block
                          │
                    route_request
                          │
        ┌─────────────────┬───────────────────┬───────────────┐
  'log_inspection'    'ask_advice'      'check_history'    'unrelated'
        │                  │                   │                │
  inspection_agent    advisor_agent       history_agent    handle_unrelated
   (MCP tools,              │
   vision-capable)   treatment_safety_gate
                      (deterministic, no LLM)
                            │
              ┌─────────────┼──────────────┐
            'clean'      'caution'      'blocked'
              │              │              │
          give_advice   give_advice   alert_keeper
                        +safety caveat  (escalate to the keeper,
                                         no chemical-specific advice)

Why a multi-agent graph (vs. one prompt)
-----------------------------------------
A single prompt would have to juggle parsing field notes, calling live
weather/forage tools, AND making a safety call all in one pass -- and an LLM
can be talked out of its own safety reasoning if it is mixed into the same
turn as the recommendation. Splitting "propose" (advisor_agent, an LLM) from
"check" (treatment_safety_gate, deterministic Python) means the safety
decision is reproducible and auditable independent of what the model said.

Routing (triage -> route_request) is done WITHOUT an LLM call wherever
possible so the control flow stays predictable, and the apicultural-safety
gate is forced onto every path that could end in a treatment recommendation.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Optional

from google.adk.agents import Context, LlmAgent
from google.adk.apps import App
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.workflow import START, Edge, Workflow, node
from google.adk.workflow.utils._workflow_graph_utils import build_node
from mcp import StdioServerParameters
from pydantic import BaseModel

from apiary_network import config
from apiary_network.app import security

# --- The shared MCP connection ----------------------------------------------
# One stdio connection to apiary_data_server.py, reused by every agent that
# needs hive data, the forage calendar, or the live weather feed. The server
# process is launched and owned by the toolset; agents only ever see the
# tool-call protocol boundary, never the SQLite file directly.
apiary_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python3",
            args=[str(config.MCP_SERVER_PATH)],
            # PYTHONPATH is required so the subprocess can `from apiary_network
            # import config` regardless of the notebook's working directory --
            # `python3 <script>` only auto-adds the *script's own* directory to
            # sys.path, not the package root above it.
            env={"APIARY_DB_PATH": str(config.DB_PATH), "PYTHONPATH": str(config.ROOT.parent)},
        ),
        timeout=30,
    )
)


# --- Structured outputs for the LLM nodes that need them --------------------
class TriageResult(BaseModel):
    category: Literal["log_inspection", "ask_advice", "check_history", "unrelated"]
    reasoning: str


class AdvisorPlan(BaseModel):
    """What the advisor agent proposes, in a shape the deterministic safety
    gate can check without re-parsing free text."""

    hive_id: Optional[str] = None
    region: Optional[str] = None
    recommendation_summary: str
    proposes_treatment: bool
    proposed_treatment_type: Optional[Literal["oxalic_acid", "formic_acid_strips", "thymol"]] = None
    outdoor_temp_f: Optional[float] = None
    is_nectar_flow: Optional[bool] = None
    days_since_last_treatment: Optional[int] = None


# --- Step 1: deterministic input gate ---------------------------------------
@node
def save_query(ctx: Context, node_input: Any) -> None:
    """Redact location PII and screen for prompt injection BEFORE any LLM sees
    the text. Zero tokens are spent on malicious or location-sensitive input
    that gets caught here."""
    from google.adk.utils.content_utils import extract_text_from_content

    raw_text = node_input if isinstance(node_input, str) else extract_text_from_content(ctx.user_content)
    redacted_text, redactions = security.redact_location(raw_text)
    injected = security.detect_injection(raw_text)

    ctx.state["query"] = {"text": redacted_text, "redactions": redactions, "injection_detected": injected}
    # ADK's instruction templating only accepts valid Python identifiers as
    # state keys (verified against the installed instructions_utils.py) --
    # "query[text]" is NOT a valid identifier, so {query[text]} would have
    # silently passed through as literal text in every prompt below rather
    # than being replaced. query_text is the flat, identifier-safe mirror
    # that the LlmAgent instructions actually reference.
    ctx.state["query_text"] = redacted_text
    ctx.route = "blocked_injection" if injected else "proceed"


@node
def security_block(ctx: Context) -> str:
    """Terminal node for caught injection attempts. Deliberately generic --
    it does not explain *why* it was caught, which would hand back a recipe
    for evading the filter next time."""
    return (
        "I can't act on that request as written -- it looks like it's trying to "
        "override the safety checks built into this assistant. If you have a "
        "genuine question about hive care, feel free to rephrase it."
    )


# --- Step 2: cheap LLM triage, then deterministic routing -------------------
triage_agent = LlmAgent(
    name="triage",
    model=config.TRIAGE_MODEL,
    instruction=(
        "Classify the beekeeper's message into exactly one category:\n"
        "- 'log_inspection': they are reporting what they observed during a hive check.\n"
        "- 'ask_advice': they are asking what to do (treatment, feeding, timing, general care).\n"
        "- 'check_history': they are asking about a hive's past records or trends.\n"
        "- 'unrelated': anything not about beekeeping.\n\n"
        "Message: {query_text}"
    ),
    output_schema=TriageResult,
    output_key="triage",
)


@node
def route_request(ctx: Context) -> None:
    """Deterministic routing -- no LLM call needed once triage has classified
    the message."""
    triage = ctx.state.get("triage") or {}
    ctx.route = triage.get("category", "unrelated")


# --- Step 3a: log_inspection branch -----------------------------------------
inspection_agent = LlmAgent(
    name="inspection_agent",
    model=config.ADK_MODEL,
    instruction=(
        "The beekeeper is reporting an inspection. Extract these fields from "
        "their message and call the log_inspection tool to save them:\n"
        "hive_id, brood_pattern, queen_seen (bool), eggs_seen (bool), mite_count "
        "(numeric, default 0 if not mentioned), temperament, stores_lbs "
        "(estimate if not given), notes (anything else worth keeping).\n\n"
        "If a photo of a frame or the hive interior is attached, look for brood "
        "pattern (solid vs. spotty), visible eggs/larvae, queen cells (a sign of "
        "swarming or supersedure), and visible mites on bees, and fold those "
        "observations into the fields above before calling the tool.\n\n"
        "After logging, confirm what was saved in one or two plain sentences.\n\n"
        "Message: {query_text}"
    ),
    tools=[apiary_mcp_toolset],
)


# --- Step 3b: ask_advice branch ----------------------------------------------
advisor_agent = LlmAgent(
    name="advisor_agent",
    model=config.ADK_MODEL,
    instruction=(
        "The beekeeper is asking for care advice. You MUST call get_current_weather "
        "and get_forage_calendar before answering, even if you think you already know "
        "the answer -- never guess or assume the temperature or nectar-flow status. "
        "Identify the hive_id and region (use get_hive_history if region is unclear; "
        "if you cannot determine a real latitude/longitude for the region, use a "
        "reasonable representative coordinate for that region rather than skipping the "
        "weather call entirely). Also call get_treatment_thresholds and "
        "days_since_last_treatment.\n\n"
        "Produce a recommendation_summary in plain language, AND fill in the "
        "structured fields truthfully from the ACTUAL tool results -- if you are "
        "recommending a specific miticide treatment, set proposes_treatment=true and "
        "fill in proposed_treatment_type, outdoor_temp_f, is_nectar_flow and "
        "days_since_last_treatment from what the tools returned, not from assumption. "
        "If a tool call failed or returned no data, leave the corresponding field null "
        "rather than inventing a plausible-sounding value -- a deterministic safety "
        "check runs after you and is designed to treat null fields as 'unverified', "
        "which is the correct, honest outcome when a tool call didn't succeed.\n\n"
        "If you are NOT recommending a chemical treatment (e.g. just 'add a super' or "
        "'keep monitoring'), set proposes_treatment=false and leave the treatment "
        "fields null.\n\n"
        "Message: {query_text}"
    ),
    tools=[apiary_mcp_toolset],
    output_schema=AdvisorPlan,
    output_key="advisor_plan",
)


@node
def treatment_safety_gate(ctx: Context) -> None:
    """The apicultural-safety equivalent of a procurement bulk-order gate:
    pure thresholds, no LLM, fully auditable. Runs on every 'ask_advice' path,
    but only applies the chemical-specific checks when a treatment was
    actually proposed."""
    plan = ctx.state.get("advisor_plan") or {}
    ctx.state["advisor_summary"] = plan.get("recommendation_summary", "")
    if not plan.get("proposes_treatment"):
        ctx.state["safety"] = {"route": "clean", "reasons": ["No treatment was proposed; routine advice only."]}
        ctx.state["safety_route"] = "clean"
        ctx.state["safety_reasons_text"] = "No treatment was proposed; routine advice only."
        ctx.route = "clean"
        return

    verdict = security.check_treatment_safety(
        treatment_type=plan.get("proposed_treatment_type"),
        outdoor_temp_f=plan.get("outdoor_temp_f"),
        is_nectar_flow=plan.get("is_nectar_flow"),
        days_since_last_treatment=plan.get("days_since_last_treatment"),
    )
    ctx.state["safety"] = verdict.to_dict()
    # Flat, identifier-safe mirrors for instruction templating -- see the
    # comment in save_query for why {safety[route]}/{safety[reasons]} would
    # have silently failed to substitute.
    ctx.state["safety_route"] = verdict.route
    ctx.state["safety_reasons_text"] = " ".join(verdict.reasons)
    ctx.route = verdict.route


give_advice = LlmAgent(
    name="give_advice",
    model=config.ADK_MODEL,
    instruction=(
        "Relay the advisor's recommendation to the beekeeper in friendly, "
        "concrete language: {advisor_summary}\n\n"
        "Safety check result: {safety_route}. Reasons noted: {safety_reasons_text}\n\n"
        "If the safety check route is 'caution', clearly state the caveat "
        "before giving the recommendation -- do not bury it. If it is 'clean', "
        "you can present the recommendation plainly."
    ),
)


@node
def alert_keeper(ctx: Context) -> str:
    """Terminal escalation node. Deliberately spends zero further LLM tokens
    discussing the specific chemical/temperature combination that got
    blocked -- the keeper's own judgment (or a mentor's) is what's needed
    next, not a more persuasive AI argument."""
    reasons = (ctx.state.get("safety") or {}).get("reasons", [])
    reason_text = " ".join(reasons) if reasons else "A safety threshold was not met."
    return (
        "I'm not going to recommend proceeding with this treatment right now. "
        f"{reason_text} This is a case for your own judgment -- consider "
        "checking with your local beekeeping mentor or extension office before treating."
    )


# --- Step 3c: check_history branch ------------------------------------------
history_agent = LlmAgent(
    name="history_agent",
    model=config.ADK_MODEL,
    instruction=(
        "The beekeeper wants to know about a hive's history or trends. Use the "
        "get_hive_history tool to pull recent inspections/treatments and "
        "summarize the trend (mite counts rising/falling, brood pattern "
        "consistency, treatment history) in plain language.\n\n"
        "Message: {query_text}"
    ),
    tools=[apiary_mcp_toolset],
)


# --- Step 3d: unrelated branch -----------------------------------------------
@node
def handle_unrelated(ctx: Context) -> str:
    """Deterministic, zero-token redirect for off-topic queries."""
    return "I'm built to help with backyard beekeeping -- hive inspections, treatment timing, and care history. Could you rephrase your question around one of your hives?"


# --- Assemble the workflow ---------------------------------------------------
apiary_workflow = Workflow(
    name="apiary_network",
    edges=[
        (START, save_query),
        (save_query, {"proceed": triage_agent, "blocked_injection": security_block}),
        (triage_agent, route_request),
        (
            route_request,
            {
                "log_inspection": inspection_agent,
                "ask_advice": advisor_agent,
                "check_history": history_agent,
                "unrelated": handle_unrelated,
            },
        ),
        (advisor_agent, treatment_safety_gate),
        # 'clean' and 'caution' both continue to give_advice -- ADK's graph
        # builder treats two routing-map keys pointing at the same node as a
        # duplicate edge, so this one transition is expressed as an explicit
        # Edge with a multi-value route instead of two dict entries.
        Edge(from_node=treatment_safety_gate, to_node=build_node(give_advice), route=["clean", "caution"]),
        (treatment_safety_gate, {"blocked": alert_keeper}),
    ],
)

# `App.root_agent` accepts BaseAgent or Any -- a Workflow fits the "Any" slot.
# This is the same App object reused by the FastAPI deployment wrapper in
# fast_api_app.py, so the agent that runs in this notebook is the one that
# would be deployed -- no divergence between demo and production.
app = App(name="apiary_network", root_agent=apiary_workflow)
