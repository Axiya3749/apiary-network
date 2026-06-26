"""Apiary-data MCP server (stdio transport).

Why an MCP server (and not just Python functions)
--------------------------------------------------
Exposing the hive database, forage calendar and weather feed through the
Model Context Protocol -- rather than bolting plain functions onto one agent --
buys three things:

  1. Decoupling  - the data layer is a separate process with its own
                   lifecycle; agents only speak the protocol, not SQL.
  2. Reuse       - any MCP-capable client (other agents, IDEs, Antigravity,
                   Claude, etc.) can consume these tools without our code.
  3. Governance  - tool access is brokered over a well-defined boundary that
                   can be permissioned and audited.

Bonus authenticity: `get_current_weather` calls the real, free Open-Meteo API
(no key required) instead of synthesizing it -- one genuinely live external
feed alongside the reproducible synthetic hive catalogue. If the notebook
environment has no internet access enabled, it falls back to a labelled
placeholder reading rather than failing the whole tool call.

This server is intentionally self-contained: it only needs the DB path
(passed via the APIARY_DB_PATH environment variable by whoever launches it)
and network access for the weather call.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from apiary_network import config

mcp = FastMCP("apiary-data-server")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --- Synthetic, deterministic forage calendar -------------------------------
# Real bloom timing varies by micro-climate; this is a simplified, reproducible
# stand-in keyed by region + month. A documented "next bonus" (see the
# Summary section) is swapping this for a real phenology API.
_FORAGE_CALENDAR: dict[str, dict[int, list[str]]] = {
    "Northeast": {
        3: ["red maple", "willow"], 4: ["dandelion", "fruit trees"],
        5: ["black locust", "clover starting"], 6: ["clover", "basswood"],
        7: ["basswood tail end", "knotweed starting"], 8: ["goldenrod starting"],
        9: ["goldenrod", "aster"], 10: ["aster tail end"],
    },
    "Mid-Atlantic": {
        3: ["maple", "henbit"], 4: ["fruit trees", "tulip poplar"],
        5: ["tulip poplar", "clover"], 6: ["clover", "privet"],
        7: ["knotweed"], 8: ["goldenrod starting"], 9: ["goldenrod", "aster"], 10: ["aster"],
    },
    "Upper Midwest": {
        4: ["willow", "maple"], 5: ["dandelion", "fruit trees"],
        6: ["clover", "basswood"], 7: ["basswood", "alfalfa"],
        8: ["goldenrod starting"], 9: ["goldenrod", "aster"],
    },
    "Pacific Northwest": {
        3: ["willow"], 4: ["bigleaf maple", "fruit trees"], 5: ["blackberry starting"],
        6: ["blackberry"], 7: ["blackberry tail end", "fireweed"], 8: ["fireweed tail end"],
    },
}


@mcp.tool()
def get_forage_calendar(region: str, month: int) -> str:
    """Look up the major nectar sources blooming in a region/month and whether
    it counts as nectar-flow season for the treatment-safety gate.

    Args:
        region: One of the configured regions, e.g. "Northeast".
        month: Calendar month as an integer 1-12.
    """
    blooms = _FORAGE_CALENDAR.get(region, {}).get(month, [])
    is_nectar_flow = month in config.DEFAULT_NECTAR_FLOW_MONTHS or bool(blooms)
    return json.dumps({"region": region, "month": month, "blooming": blooms, "is_nectar_flow": is_nectar_flow})


@mcp.tool()
def get_current_weather(latitude: float, longitude: float) -> str:
    """Fetch the current outdoor temperature (Fahrenheit) for a location using
    the free Open-Meteo API. No API key required. Falls back to a labelled
    placeholder if the network call fails (e.g. internet disabled in the
    notebook environment)."""
    try:
        import urllib.request

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            "&current=temperature_2m&temperature_unit=fahrenheit"
        )
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        temp_f = data["current"]["temperature_2m"]
        return json.dumps({"temperature_f": temp_f, "source": "open-meteo.com (live)"})
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the tool call
        return json.dumps(
            {
                "temperature_f": None,
                "source": "unavailable",
                "error": f"Could not reach weather API ({exc}). Enable internet access for this notebook, "
                "or have the keeper report current temperature directly.",
            }
        )


@mcp.tool()
def get_treatment_thresholds() -> str:
    """Return the configured apicultural safety thresholds (mite economic
    threshold, safe treatment temperature window, minimum days between
    treatments, approved treatment types)."""
    return json.dumps(
        {
            "varroa_treatment_threshold_per_100_bees": config.VARROA_TREATMENT_THRESHOLD,
            "min_treatment_temp_f": config.MIN_TREATMENT_TEMP_F,
            "max_treatment_temp_f": config.MAX_TREATMENT_TEMP_F,
            "min_days_between_treatments": config.MIN_DAYS_BETWEEN_TREATMENTS,
            "approved_treatment_types": sorted(config.APPROVED_TREATMENT_TYPES),
        }
    )


@mcp.tool()
def get_hive_history(hive_id: str, limit: int = 5) -> str:
    """Return the most recent inspections and treatments on record for a hive.

    Args:
        hive_id: The hive identifier, e.g. "A1".
        limit: Maximum number of most-recent inspections to return.
    """
    conn = _connect()
    inspections = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM inspections WHERE hive_id = ? ORDER BY inspection_date DESC LIMIT ?",
            (hive_id, limit),
        )
    ]
    treatments = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM treatments WHERE hive_id = ? ORDER BY treatment_date DESC LIMIT ?",
            (hive_id, limit),
        )
    ]
    conn.close()
    if not inspections and not treatments:
        return json.dumps({"error": f"No records found for hive_id '{hive_id}'."})
    return json.dumps({"hive_id": hive_id, "inspections": inspections, "treatments": treatments})


@mcp.tool()
def log_inspection(
    hive_id: str,
    brood_pattern: str,
    queen_seen: bool,
    eggs_seen: bool,
    mite_count: float,
    temperament: str,
    stores_lbs: float,
    notes: str = "",
) -> str:
    """Write a new structured inspection record for a hive.

    All fields are required except notes. mite_count is per-100-bees or
    per-24h sticky-board count, whichever the keeper used consistently.
    """
    if mite_count < 0:
        return json.dumps({"error": "mite_count must be non-negative."})
    conn = _connect()
    exists = conn.execute("SELECT 1 FROM hives WHERE hive_id = ?", (hive_id,)).fetchone()
    if not exists:
        conn.close()
        return json.dumps({"error": f"Unknown hive_id '{hive_id}'. Register the hive first."})
    conn.execute(
        """INSERT INTO inspections
           (hive_id, inspection_date, brood_pattern, queen_seen, eggs_seen,
            mite_count, temperament, stores_lbs, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            hive_id,
            date.today().isoformat(),
            brood_pattern,
            int(queen_seen),
            int(eggs_seen),
            mite_count,
            temperament,
            stores_lbs,
            notes,
        ),
    )
    conn.commit()
    conn.close()
    return json.dumps({"status": "logged", "hive_id": hive_id, "inspection_date": date.today().isoformat()})


@mcp.tool()
def log_treatment(hive_id: str, treatment_type: str, outdoor_temp_f: float) -> str:
    """Record that a treatment was applied to a hive today. This is a pure
    data-write tool -- the SAFETY DECISION about whether to proceed happens
    earlier in the deterministic security gate, not here."""
    conn = _connect()
    exists = conn.execute("SELECT 1 FROM hives WHERE hive_id = ?", (hive_id,)).fetchone()
    if not exists:
        conn.close()
        return json.dumps({"error": f"Unknown hive_id '{hive_id}'."})
    conn.execute(
        "INSERT INTO treatments (hive_id, treatment_date, treatment_type, outdoor_temp_f) VALUES (?, ?, ?, ?)",
        (hive_id, date.today().isoformat(), treatment_type, outdoor_temp_f),
    )
    conn.commit()
    conn.close()
    return json.dumps({"status": "logged", "hive_id": hive_id, "treatment_type": treatment_type})


@mcp.tool()
def days_since_last_treatment(hive_id: str) -> str:
    """Return the number of days since the most recent treatment on record for
    a hive, or null if none is on record."""
    conn = _connect()
    row = conn.execute(
        "SELECT treatment_date FROM treatments WHERE hive_id = ? ORDER BY treatment_date DESC LIMIT 1",
        (hive_id,),
    ).fetchone()
    conn.close()
    if not row:
        return json.dumps({"hive_id": hive_id, "days_since_last_treatment": None})
    last = datetime.fromisoformat(row["treatment_date"]).date()
    delta = (date.today() - last).days
    return json.dumps({"hive_id": hive_id, "days_since_last_treatment": delta})


if __name__ == "__main__":
    mcp.run(transport="stdio")
