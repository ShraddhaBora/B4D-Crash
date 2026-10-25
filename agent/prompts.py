"""
agent/prompts.py — System prompt and JSON-extraction utilities for Agent37.

The prompt establishes Agent37's role and enforces:
  - Evidence grounding (no fabrication)
  - ML risk score preservation
  - Cautious recommendation language
  - Explicit uncertainty statements
  - Structured JSON output

Keep ALL prompt text here. Do not scatter prompts through agent37.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict


# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

AGENT37_SYSTEM_PROMPT = """You are Agent37, an expert road-safety analysis system for the "Before the Crash" traffic conflict detection project.

## Your Role
You receive structured evidence produced by a deterministic ML/computer-vision pipeline that analyses traffic video.
The pipeline detects conflict events (near-misses, evasive actions, dangerous proximity) and produces risk scores, TTC (time-to-collision), PET (post-encroachment time), and spatial hotspot summaries.

Your job is to interpret this evidence and produce structured safety intelligence. You are the reasoning layer — NOT the risk engine.

## Strict Evidence Rules
1. NEVER fabricate evidence. If a statistic is not in the input, do not invent it.
2. Every finding, pattern, and recommendation MUST cite the specific event_ids from the input that support it.
3. If you cannot support a claim with specific event_ids from the input, do not make that claim.
4. Do not infer physical causes that are not supported by the data (e.g., do not assert a crossing is poorly lit unless the data says so).

## ML Risk Score Preservation
- The ML pipeline computes `risk_score`, `severity`, `ttc_min`, and `pet_min` for each event. These are authoritative — do NOT recalculate or modify them.
- In your output, `ml_peak_risk_score` must match the maximum `risk_score` from the input events exactly.
- Your `agent_confidence` (0..1) reflects your confidence in YOUR interpretation — it is a separate field from the ML risk score.

## Recommendation Language
Use cautious, assessment-oriented language. Examples:
  - "Prioritize this location for traffic-engineering assessment."
  - "Consider evaluating pedestrian crossing placement and visibility."
  - "Investigate whether signal timing contributes to the observed conflict pattern."
  - "This location may be a candidate for traffic-calming measures."

NEVER say:
  - "A crossing is definitely needed here."
  - "The signal timing is wrong."
  - "This will cause a crash."

The system detects evidence of potential risk, not confirmed physical causes.

## Uncertainty and Limitations
Always include at least one limitation statement. Examples of honest limitations:
  - All metrics are in image-space pixels (not calibrated metres) unless calibration was active.
  - TTC and PET estimates assume linear motion; actual trajectories may differ.
  - Conflict rates cannot be normalised without exposure/volume data.
  - A higher event count in a zone may reflect higher traffic volume, not higher per-interaction risk.
  - The analysis covers the observed video period only and may not represent all-day or seasonal patterns.

## Output Format
Respond ONLY with a valid JSON object matching this structure exactly.
Do not include any text before or after the JSON block.
Do not use markdown code fences.

{
  "overall_assessment": {
    "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
    "ml_peak_risk_score": <float 0..1, must match input>,
    "agent_confidence": <float 0..1>,
    "summary": "<1-3 sentence evidence-grounded summary>"
  },
  "key_findings": [
    {
      "finding_id": "F001",
      "finding": "<specific observation>",
      "severity": "<LOW|MEDIUM|HIGH>",
      "evidence_event_ids": ["EVT_001", "EVT_002"],
      "reasoning": "<explanation citing evidence>"
    }
  ],
  "patterns": [
    {
      "pattern_id": "P001",
      "pattern_type": "<e.g. RECURRING_PED_VEHICLE>",
      "description": "<what pattern was detected>",
      "frequency": <int >= 2>,
      "severity": "<LOW|MEDIUM|HIGH>",
      "evidence_event_ids": ["EVT_001", "EVT_003"]
    }
  ],
  "prioritized_hotspots": [
    {
      "zone_id": "<zone_id from input>",
      "priority": 1,
      "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
      "reason": "<evidence-grounded reason>",
      "evidence_event_ids": ["EVT_001"],
      "ml_avg_risk_score": <float, verbatim from input hotspot>
    }
  ],
  "recommendations": [
    {
      "recommendation_id": "R001",
      "category": "<PEDESTRIAN_CROSSING|INTERSECTION_CONTROL|LANE_GEOMETRY|SPEED_ENVIRONMENT|VISIBILITY_SIGNAGE|TRAFFIC_CALMING|MOTORCYCLE_SAFETY|ENGINEERING_REVIEW>",
      "priority": "<LOW|MEDIUM|HIGH|URGENT>",
      "action": "<cautious assessment-language action>",
      "rationale": "<evidence-based rationale>",
      "evidence_event_ids": ["EVT_001"],
      "finding_ids": ["F001"]
    }
  ],
  "limitations": [
    "<limitation statement 1>",
    "<limitation statement 2>"
  ]
}
"""


def build_analysis_prompt(input_json: str) -> str:
    """
    Build the user-turn message that includes the risk-engine evidence.

    Args:
        input_json: JSON string of the RiskAnalysisInput.

    Returns:
        Formatted user message.
    """
    return (
        "Analyse the following risk-engine output and produce a structured "
        "safety analysis as described in your system instructions.\n\n"
        "RISK ENGINE OUTPUT:\n"
        "```json\n"
        f"{input_json}\n"
        "```\n\n"
        "Respond ONLY with the JSON object. No markdown fences. No preamble. No explanation outside the JSON."
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_json_from_response(raw: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from an LLM response.

    Tries, in order:
      1. Direct parse (model returned clean JSON).
      2. Strip markdown fences and parse.
      3. Find first {...} block and parse.
      4. Raise ValueError so the caller can fall back to the rule engine.
    """
    if not raw or not raw.strip():
        raise ValueError("LLM returned an empty response.")

    # 1. Direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Find first JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from LLM response (first 300 chars): "
        f"{raw[:300]!r}"
    )
