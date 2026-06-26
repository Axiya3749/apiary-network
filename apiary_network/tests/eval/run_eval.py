"""Scenario-based eval harness for the apiary_network workflow.

Run with: python3 -m apiary_network.tests.eval.run_eval

This drives the REAL workflow (via InMemoryRunner) against every scenario in
datasets/scenarios.json and checks the deterministic claims in eval_config.yaml:
- routing_correctness: did ctx.state['triage']['category'] (or the injection
  route) match what the scenario expects?
- safety_gate_containment: for scenarios where a treatment was proposed, does
  re-running check_treatment_safety on the advisor's own reported fields
  agree with the route the gate actually took? (Catches a gate that silently
  stopped being called, not just a gate that computes the wrong answer.)
- location_privacy: for the redaction scenario, is the GPS pattern gone from
  the saved query text?

This requires a real GOOGLE_API_KEY and network access to Gemini, since the
triage/advisor/etc. nodes are real LlmAgents -- it is not a pure offline
check (that's what tests/test_security.py is for). It is designed to run on
Kaggle/Colab with a key configured, not in an offline sandbox.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from google.adk.runners import InMemoryRunner
from google.genai import types

from apiary_network import config
from apiary_network.app import security
from apiary_network.app.agent import app as adk_app

_SCENARIOS_PATH = Path(__file__).resolve().parent / "datasets" / "scenarios.json"


async def _run_one(runner: InMemoryRunner, scenario: dict) -> dict:
    session_id = str(uuid.uuid4())
    user_id = "eval_runner"
    await runner.session_service.create_session(app_name=adk_app.name, user_id=user_id, session_id=session_id)
    msg = types.Content(role="user", parts=[types.Part(text=scenario["query"])])

    final_text = ""
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=msg):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text = part.text
        elif isinstance(getattr(event, "output", None), str) and event.output:
            final_text = event.output

    session = await runner.session_service.get_session(app_name=adk_app.name, user_id=user_id, session_id=session_id)
    state = dict(session.state) if session else {}

    result = {"id": scenario["id"], "final_text": final_text, "state": state, "checks": {}}

    # --- routing_correctness ---
    if "expected_route" in scenario:
        actual = (state.get("query") or {}).get("injection_detected")
        result["checks"]["routing_correctness"] = (
            "blocked_injection" if actual else "proceed"
        ) == scenario["expected_route"]
    elif "expected_triage_category" in scenario:
        actual_category = (state.get("triage") or {}).get("category")
        result["checks"]["routing_correctness"] = actual_category == scenario["expected_triage_category"]

    # --- safety_gate_containment (re-derive independently) ---
    if "expected_safety_route_if_treatment_proposed" in scenario:
        plan = state.get("advisor_plan") or {}
        if plan.get("proposes_treatment"):
            re_derived = security.check_treatment_safety(
                treatment_type=plan.get("proposed_treatment_type"),
                outdoor_temp_f=plan.get("outdoor_temp_f"),
                is_nectar_flow=bool(plan.get("is_nectar_flow")),
                days_since_last_treatment=plan.get("days_since_last_treatment"),
            )
            actual_route = (state.get("safety") or {}).get("route")
            result["checks"]["safety_gate_containment"] = re_derived.route == actual_route
        else:
            result["checks"]["safety_gate_containment"] = None  # no treatment proposed; not applicable

    # --- location_privacy ---
    if scenario.get("expected_redaction"):
        query_text = (state.get("query") or {}).get("text", "")
        _, found = security.redact_location(query_text)
        result["checks"]["location_privacy"] = found == []  # nothing LEFT to redact means it was already redacted

    return result


async def run_all() -> list[dict]:
    scenarios = json.loads(_SCENARIOS_PATH.read_text())
    runner = InMemoryRunner(app=adk_app)
    results = []
    # Each scenario triggers at least 2 model calls (triage + a specialist agent).
    # The Gemini free tier caps gemini-2.5-flash at 5 requests/minute -- pacing
    # between scenarios (not just within one) keeps an 8-scenario run from
    # tripping a 429 RESOURCE_EXHAUSTED partway through. Override with the
    # EVAL_SCENARIO_DELAY_SECONDS env var (e.g. set to 0 once you're on a paid
    # tier with a higher quota).
    delay_seconds = float(os.environ.get("EVAL_SCENARIO_DELAY_SECONDS", "15"))
    for i, scenario in enumerate(scenarios):
        try:
            results.append(await _run_one(runner, scenario))
        except Exception as exc:  # noqa: BLE001
            results.append({"id": scenario["id"], "error": str(exc)})
        if i < len(scenarios) - 1 and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
    return results


def main() -> int:
    if not config.ROOT.exists():
        print("Repo not found; run from the notebook root.")
        return 1
    results = asyncio.run(run_all())
    n_fail = 0
    for r in results:
        if "error" in r:
            print(f"[ERROR] {r['id']}: {r['error']}")
            n_fail += 1
            continue
        checks = r["checks"]
        ok = all(v in (True, None) for v in checks.values())
        n_fail += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {r['id']}: {checks}")
    print(f"\n{len(results) - n_fail}/{len(results)} scenarios passed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
