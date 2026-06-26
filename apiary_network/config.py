"""Central configuration for the Backyard Apiary Intelligence Network.

Why this file exists
--------------------
The agents, the MCP server, the security layer and the evaluation harness all
need to agree on the *same* paths, model id and apicultural safety thresholds.
Centralising them here means there is one source of truth: change a number
once and the whole system follows. A real deployment would inject overrides
(model id, DB location, region defaults) via environment variables instead of
editing code.
"""
from __future__ import annotations

import os
from pathlib import Path

# Resolve the project root from this file's location so the code runs no
# matter what the current working directory is (Kaggle/Colab mount notebooks
# differently each session).
ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("APIARY_DB_PATH", ROOT / "data" / "apiary.db"))
MCP_SERVER_PATH = ROOT / "mcp_server" / "apiary_data_server.py"

# --- Model -----------------------------------------------------------------
# Override with the ADK_MODEL env var to trade cost for quality without
# touching code.
ADK_MODEL = os.environ.get("ADK_MODEL", "gemini-2.5-flash")

# triage_agent does a single, simple one-of-four classification with no tool
# calls -- exactly the high-volume, low-complexity task Flash-Lite is built
# for. The rest of the agents (in particular advisor_agent, whose tool-calling
# accuracy feeds directly into the safety gate) stay on the stronger ADK_MODEL.
# Independently overridable so either can be tuned without touching the other.
TRIAGE_MODEL = os.environ.get("TRIAGE_MODEL", "gemini-2.5-flash-lite")

# --- Apicultural safety policy ----------------------------------------------
# These numbers are deliberately simple defaults for a hobbyist/backyard
# context. A real deployment would let the keeper tune them per-product
# label and per-region in a settings UI rather than hardcoding them.

# Varroa mite economic threshold: above this (mites per 100 bees from an
# alcohol-wash or per-24h sticky-board count), treatment is recommended.
VARROA_TREATMENT_THRESHOLD = 3.0

# Most common backyard miticides (oxalic acid vaporization/dribble, formic
# acid strips, thymol) have a labelled safe application temperature window.
# Outside of it they can be ineffective or harm the colony/queen.
MIN_TREATMENT_TEMP_F = 50.0
MAX_TREATMENT_TEMP_F = 85.0

# Treating during a strong nectar flow risks contaminating honey that will
# be harvested for consumption -- flag for caution rather than block outright,
# since sometimes treatment is still the right call (e.g. severe infestation).
# Months are defaults for a temperate Northern Hemisphere climate; override
# per-region via the forage calendar tool.
DEFAULT_NECTAR_FLOW_MONTHS = {4, 5, 6, 7}

APPROVED_TREATMENT_TYPES = {"oxalic_acid", "formic_acid_strips", "thymol"}

# A treatment applied to the same hive within this many days of the last one
# is flagged -- repeated miticide use without monitoring risks resistance.
MIN_DAYS_BETWEEN_TREATMENTS = 14

# --- Privacy -----------------------------------------------------------------
# Hive theft is a real, documented risk for beekeepers. Exact coordinates or
# street addresses should never reach an LLM prompt or a log file.
LOCATION_REDACTION_LABEL = "[LOCATION REDACTED]"
