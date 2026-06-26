"""Deterministic security primitives for the apiary workflow.

Why deterministic (not "ask the LLM")
--------------------------------------
Two safety-critical decisions live here, and neither should be left to a
model's judgement:

  1. Location privacy. Hive theft is a real, documented problem for
     beekeepers -- exact coordinates or addresses in a query or inspection
     note must never reach an LLM prompt or a log file, because that prompt
     and that log are themselves a leak vector.
  2. Treatment safety. Recommending a miticide at the wrong temperature, or
     during a strong nectar flow, can harm a colony or contaminate honey.
     That is an apicultural-safety fact, not a matter of model "opinion", so
     it is enforced with plain Python before a recommendation is finalised.

Every check here is regex/threshold-based and runs in a `@node` BEFORE any
LLM sees the text (location redaction) or finalises an answer (safety gate).
This mirrors the "place deterministic security checks before the LlmAgent"
pattern from the course materials, re-targeted from procurement/expense
approval to apicultural safety.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from apiary_network import config

# --- Prompt-injection signatures --------------------------------------------
INJECTION_KEYWORDS = [
    "ignore previous",
    "ignore prior",
    "ignore all previous instructions",
    "disregard the safety",
    "bypass the safety",
    "skip the safety check",
    "without checking",
    "auto-approve",
    "you are now",
    "act as",
    "override the rules",
    "no need to verify",
]

_INJECTION_PATTERN = re.compile("|".join(re.escape(k) for k in INJECTION_KEYWORDS), re.IGNORECASE)

# --- Location PII patterns ---------------------------------------------------
# Decimal-degree GPS pairs, e.g. "42.3601, -71.0589" or "42.3601 -71.0589".
_GPS_PATTERN = re.compile(r"-?\d{1,3}\.\d{3,8}\s*,?\s*-?\d{1,3}\.\d{3,8}")
# A loose street-address heuristic: number + word(s) + common street suffix.
_STREET_PATTERN = re.compile(
    r"\b\d{1,5}\s+([A-Za-z]+\s){1,4}(St|Street|Ave|Avenue|Rd|Road|Ln|Lane|Dr|Drive|Way|Ct|Court)\b",
    re.IGNORECASE,
)


def detect_injection(text: str) -> bool:
    """Return True if the text contains a likely prompt-injection signature."""
    return bool(_INJECTION_PATTERN.search(text or ""))


def redact_location(text: str) -> tuple[str, list[str]]:
    """Strip GPS coordinates and street addresses before anything is logged or
    sent to the model. Returns (redacted_text, list_of_redaction_types_found)."""
    found: list[str] = []
    redacted = text or ""
    if _GPS_PATTERN.search(redacted):
        found.append("gps_coordinates")
        redacted = _GPS_PATTERN.sub(config.LOCATION_REDACTION_LABEL, redacted)
    if _STREET_PATTERN.search(redacted):
        found.append("street_address")
        redacted = _STREET_PATTERN.sub(config.LOCATION_REDACTION_LABEL, redacted)
    return redacted, found


@dataclass
class SafetyVerdict:
    route: str  # "clean" | "caution" | "blocked"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"route": self.route, "reasons": self.reasons}


def check_treatment_safety(
    treatment_type: str | None,
    outdoor_temp_f: float | None,
    is_nectar_flow: bool | None,
    days_since_last_treatment: int | None,
) -> SafetyVerdict:
    """Deterministically classify a proposed treatment as clean / caution / blocked.

    This is the apicultural-safety equivalent of the procurement security gate's
    bulk-order policy check: pure thresholds, no LLM, fully auditable.

    Critical design point: missing data is NOT treated as "passes the check."
    An advisor agent that failed to call its weather/forage tools and left
    outdoor_temp_f or is_nectar_flow as None must produce a 'caution' verdict,
    not a silent 'clean' one -- "I don't know the temperature" must never
    collapse into "the temperature is fine." is_nectar_flow is explicitly
    three-state (True / False / None-unknown) for this reason; do not coerce
    it with bool() before calling this function.
    """
    reasons: list[str] = []
    blocked = False
    caution = False

    if treatment_type and treatment_type not in config.APPROVED_TREATMENT_TYPES:
        reasons.append(
            f"'{treatment_type}' is not in the approved treatment list "
            f"({sorted(config.APPROVED_TREATMENT_TYPES)})."
        )
        blocked = True

    if outdoor_temp_f is None:
        reasons.append(
            "Outdoor temperature could not be verified (the weather tool was not "
            "called or returned no data) -- confirm the current temperature is "
            f"between {config.MIN_TREATMENT_TEMP_F:.0f}-{config.MAX_TREATMENT_TEMP_F:.0f}F before treating."
        )
        caution = True
    elif outdoor_temp_f < config.MIN_TREATMENT_TEMP_F or outdoor_temp_f > config.MAX_TREATMENT_TEMP_F:
        reasons.append(
            f"Outdoor temperature {outdoor_temp_f:.0f}F is outside the labelled safe "
            f"application window ({config.MIN_TREATMENT_TEMP_F:.0f}-"
            f"{config.MAX_TREATMENT_TEMP_F:.0f}F)."
        )
        blocked = True

    if is_nectar_flow is None:
        reasons.append(
            "Nectar-flow status could not be verified (the forage calendar tool was "
            "not called or returned no data) -- confirm whether this is peak nectar "
            "flow season before treating, since that risks contaminating honey meant for harvest."
        )
        caution = True
    elif is_nectar_flow:
        reasons.append("This is peak nectar flow season -- treating now risks contaminating honey meant for harvest.")
        caution = True

    if days_since_last_treatment is not None and days_since_last_treatment < config.MIN_DAYS_BETWEEN_TREATMENTS:
        reasons.append(
            f"This hive was treated {days_since_last_treatment} day(s) ago, under the "
            f"{config.MIN_DAYS_BETWEEN_TREATMENTS}-day minimum -- repeated treatment without a fresh "
            f"mite count risks resistance and unnecessary chemical load."
        )
        caution = True

    if blocked:
        return SafetyVerdict(route="blocked", reasons=reasons)
    if caution:
        return SafetyVerdict(route="caution", reasons=reasons)
    return SafetyVerdict(route="clean", reasons=reasons or ["Within labelled temperature window, not nectar flow, no recent treatment on record."])
