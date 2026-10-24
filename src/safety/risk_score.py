"""
Phase 15 — Risk Score.

Combines TTC, PET, evasive action, and vulnerability into a transparent
component-based score.  All component weights are configurable.
"""

from __future__ import annotations

import math
from src.schema import (
    CandidateInteraction, EvasiveAction, PETResult, RiskScore, TTCResult,
)


def compute_risk_score(
    candidate: CandidateInteraction,
    ttc: TTCResult,
    pet: PETResult,
    evasive: list,
    weights: dict | None = None,
    ttc_threshold: float = 3.0,
    pet_threshold: float = 2.0,
) -> RiskScore:
    """
    Risk score in [0, 1] built from explicit components.

    Component scores:
        ttc_score         — normalised criticality of TTC (1 = collision imminent)
        pet_score         — normalised criticality of PET
        evasive_score     — presence and intensity of evasive manoeuvres
        vulnerability_score — 1.0 if VRU involved, 0.5 otherwise

    Overall = weighted sum, clipped to [0, 1].
    """
    w = weights or {"ttc": 0.35, "pet": 0.30, "evasive": 0.20, "vulnerability": 0.15}

    # TTC component: 1.0 when TTC→0, 0.0 when TTC≥threshold
    if ttc.valid and ttc.value < math.inf:
        ttc_score = max(0.0, 1.0 - ttc.value / max(ttc_threshold, 1e-3))
    else:
        ttc_score = 0.0

    # PET component
    if pet.valid and pet.value < math.inf:
        pet_score = max(0.0, 1.0 - pet.value / max(pet_threshold, 1e-3))
    else:
        pet_score = 0.0

    # Evasive component
    if not evasive:
        evasive_score = 0.0
    else:
        types = {a.action_type for a in evasive}
        evasive_score = min(1.0, 0.3 * ("STOP" in types) +
                                      0.25 * ("BRAKING" in types) +
                                      0.25 * ("SWERVE" in types) +
                                      0.05 * len(evasive))

    # Vulnerability component
    vru = {"person", "bicycle", "motorcycle"}
    involves_vru = candidate.class_a in vru or candidate.class_b in vru
    vulnerability_score = 1.0 if involves_vru else 0.5

    overall = (
        w.get("ttc", 0.35)           * ttc_score
        + w.get("pet", 0.30)         * pet_score
        + w.get("evasive", 0.20)     * evasive_score
        + w.get("vulnerability", 0.15) * vulnerability_score
    )
    overall = round(min(1.0, max(0.0, overall)), 4)

    return RiskScore(
        track_id_a=candidate.track_id_a,
        track_id_b=candidate.track_id_b,
        ttc_score=round(ttc_score, 4),
        pet_score=round(pet_score, 4),
        evasive_score=round(evasive_score, 4),
        vulnerability_score=round(vulnerability_score, 4),
        overall_score=overall,
    )
