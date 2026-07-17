# Backyard Apiary Network

A multi-agent system (Google ADK 2.x) that helps a backyard beekeeper log
hive inspections, get treatment/care advice grounded in real weather data,
and review hive history -- with a deterministic safety gate standing between
any AI-generated recommendation and an actual treatment decision.

**Track:** Agents for Good (agriculture)

## The problem

A backyard beekeeper inspects hives every 1-2 weeks and has to make several
judgment calls each time: is the mite count high enough to treat? Is it too
hot or cold to treat safely right now? Should I worry, or is this normal?
New keepers especially lack the pattern-recognition that takes years to
build, and getting the timing wrong (treating during nectar flow, missing a
swarm signal) can directly cost a colony.

## Why agents

The right answer depends on combining several live, changing inputs --
inspection findings, current weather, treatment history, seasonal forage
calendar -- and some of those decisions are safety-critical enough that they
should not be left to "the LLM seemed confident." That argues for a team of
specialists plus a deterministic gate, not one prompt trying to do
everything at once.

## Architecture

```
                                   ┌──────────────┐
   user query (+ optional photo) ─►  save_query   │  redact location PII,
                                   └──────┬───────┘  screen for injection
                          ┌───────────────┴───────────────┐
                     'proceed'                    'blocked_injection'
                          │                               │
                    triage_agent (LLM)             security_block
                          │
                    route_request (deterministic)
                          │
        ┌─────────────────┬───────────────────┬───────────────┐
  'log_inspection'    'ask_advice'      'check_history'    'unrelated'
        │                  │                   │                │
  inspection_agent    advisor_agent       history_agent    handle_unrelated
   (MCP tools,              │
   vision-capable)   treatment_safety_gate (deterministic, no LLM)
                            │
              ┌─────────────┼──────────────┐
            'clean'      'caution'      'blocked'
              │              │              │
          give_advice   give_advice   alert_keeper
                        +safety caveat  (escalate to the keeper)
```

### Key concepts demonstrated
| Concept | Where |
|---|---|
| Multi-agent system (ADK) | `app/agent.py` -- a real `Workflow` graph of `LlmAgent`s and deterministic `@node` functions |
| MCP Server | `mcp_server/apiary_data_server.py` -- hive DB, live weather (Open-Meteo), synthetic forage calendar |
| Security features | `app/security.py` -- location-PII redaction, prompt-injection detection, deterministic treatment-safety gate |
| Agent skills | `.agents/skills/` -- one procedural script, one template asset, one instructions-only threat-model skill |
| Deployability | `app/fast_api_app.py` -- the same `App` object the notebook demos, wrapped for HTTP |
| Antigravity | demonstrated in the submission video, not in code |

## Setup

```bash
pip install -e ".[dev]"
export GOOGLE_API_KEY="your-key-here"   # never commit this
python3 -m apiary_network.data.seed_db   # populate the synthetic hive database
pytest apiary_network/tests              # deterministic unit tests (no API key needed)
python3 -m apiary_network.tests.eval.run_eval   # full scenario eval (needs API key + network)
```

To run the deployment wrapper locally:
```bash
uvicorn apiary_network.app.fast_api_app:api --reload
```

## Honest limitations / next steps
- The forage/bloom calendar is synthetic (hand-authored by month/region), not
  pulled from a real phenology data source -- a real next step would be
  integrating something like the USA National Phenology Network's API.
- There is no authentication layer yet (see the Spoofing section of the
  `apiary-threat-model` skill) -- fine for a single-keeper hobbyist demo, a
  real gap for any multi-user deployment.
- Photo-based inspection analysis is wired into `inspection_agent`'s
  instructions (Gemini is natively multimodal) but the notebook demo is
  text-only since no real hive photos are available in this environment --
  worth demonstrating live with an actual photo in the submission video.

