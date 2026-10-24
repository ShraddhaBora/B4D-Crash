"""
Phase 14 — Conflict Classifier (rule-based, interpretable).

Takes a candidate interaction + its TTC/PET/evasive-action evidence
and produces a ConflictResult with machine-readable reasons.

No neural network. Every decision path is explicit and inspectable.
"""

from __future__ import annotations

import math
from typing import List

from src.schema import (
    CandidateInteraction, ConflictClass, ConflictResult,
    EvasiveAction, PETResult, TTCResult,
)


def classify_conflict(
    candidate: CandidateInteraction,
    ttc: TTCResult,
    pet: PETResult,
    evasive: List[EvasiveAction],
    ttc_threshold: float = 3.0,
    pet_threshold: float = 2.0,
) -> ConflictResult:
    """
    Rule-based conflict classification.

    Classification logic:
        HIGH_RISK  — TTC critical AND (evasive action present OR PET near-zero)
        POTENTIAL  — TTC critical OR PET critical OR evasive action present
        NORMAL     — nothing critical
    """
    reasons: List[str] = []
    metrics: dict = {}

    # ── TTC evaluation ───────────────────────────────────────────────────────
    ttc_critical = ttc.valid and ttc.value < ttc_threshold
    if ttc_critical:
        reasons.append("ttc_critical")
    metrics["ttc_value"] = ttc.value if ttc.valid else None
    metrics["ttc_threshold"] = ttc_threshold

    # ── PET evaluation ───────────────────────────────────────────────────────
    pet_critical = pet.valid and pet.value < pet_threshold
    if pet_critical:
        reasons.append("pet_critical")
    if pet.valid and pet.value == 0.0:
        reasons.append("simultaneous_encroachment")
    metrics["pet_value"] = pet.value if pet.valid else None
    metrics["pet_threshold"] = pet_threshold

    # ── Evasive action ───────────────────────────────────────────────────────
    has_braking = any(a.action_type == "BRAKING" for a in evasive)
    has_swerve  = any(a.action_type == "SWERVE"  for a in evasive)
    has_stop    = any(a.action_type == "STOP"     for a in evasive)

    if has_braking:
        reasons.append("evasive_braking")
    if has_swerve:
        reasons.append("evasive_swerve")
    if has_stop:
        reasons.append("evasive_stop")

    evasive_present = has_braking or has_swerve or has_stop
    metrics["evasive_action_count"] = len(evasive)

    # ── Vulnerability bonus ──────────────────────────────────────────────────
    vru = {"person", "bicycle", "motorcycle"}
    involves_vru = candidate.class_a in vru or candidate.class_b in vru
    if involves_vru:
        reasons.append("vru_involved")

    # ── Decision ─────────────────────────────────────────────────────────────
    if ttc_critical and (evasive_present or (pet.valid and pet.value < pet_threshold * 0.5)):
        classification = ConflictClass.HIGH_RISK_CONFLICT
        severity = _severity(ttc, pet, evasive, involves_vru, high=True)
    elif ttc_critical or pet_critical or evasive_present:
        classification = ConflictClass.POTENTIAL_CONFLICT
        severity = _severity(ttc, pet, evasive, involves_vru, high=False)
    else:
        classification = ConflictClass.NORMAL
        severity = 0.0
        reasons.append("no_critical_indicators")

    confidence = _confidence(ttc, pet, evasive)

    return ConflictResult(
        classification=classification,
        severity=round(severity, 3),
        confidence=round(confidence, 3),
        reasons=reasons,
        metrics=metrics,
    )


def _severity(ttc: TTCResult, pet: PETResult, evasive: List[EvasiveAction],
              involves_vru: bool, high: bool) -> float:
    base = 0.7 if high else 0.4
    if ttc.valid and ttc.value < math.inf:
        base += 0.1 * max(0.0, 1.0 - ttc.value / 10.0)
    if pet.valid and pet.value < math.inf:
        base += 0.1 * max(0.0, 1.0 - pet.value / 5.0)
    if evasive:
        base += 0.05
    if involves_vru:
        base = min(1.0, base * 1.2)
    return min(1.0, base)


def _confidence(ttc: TTCResult, pet: PETResult, evasive: List[EvasiveAction]) -> float:
    score = 0.0
    weight = 0.0
    if ttc.valid:
        score  += ttc.confidence * 0.5
        weight += 0.5
    if pet.valid:
        score  += pet.confidence * 0.3
        weight += 0.3
    if evasive:
        score  += 0.2
        weight += 0.2
    return score / max(weight, 0.01)
