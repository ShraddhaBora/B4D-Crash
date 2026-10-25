"""
agent/recommendation_rules.py — Lightweight deterministic recommendation rules.

This module serves two purposes:
  1. PRIMARY: Used as a fallback when the LLM call fails or returns
     unvalidatable JSON. Guarantees Agent37 always produces output.
  2. SECONDARY: Provides the rule logic that the LLM prompt is expected
     to replicate, enabling comparison and regression testing.

Rules are evidence-driven: a recommendation is only generated when the
input contains sufficient supporting events. No evidence → no recommendation.

Language is deliberately cautious:
  "Prioritize for assessment", "Consider evaluating", "Investigate whether"
  — not "intervention is required".
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .schemas import (
    AgentAnalysisOutput,
    ConflictEventInput,
    ConflictPattern,
    HotspotInput,
    KeyFinding,
    OverallAssessment,
    PrioritizedHotspot,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RiskAnalysisInput,
    RiskLevel,
    SeverityLevel,
)

import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Conflict type groupings (mirrors src/schema.py interaction_type conventions)
_PED_VEHICLE_TYPES = {"VEH_PED", "VEH_PED_CROSSING", "PED_VEH"}
_VEH_VEH_TYPES     = {"VEH_VEH", "VEH_VEH_REAR_END", "VEH_VEH_CROSSING"}
_MOTO_TYPES        = {"VEH_MOTO", "MOTO_VEH", "MOTO_MOTO"}

# Severity threshold (mirrors src.hotspots.aggregation threshold)
_HIGH_SEVERITY_THRESHOLD = 0.7

# Pattern detection minimum frequency
_PATTERN_MIN_FREQ = 2

# Evasive-action reason codes (from src.safety.conflict_classifier)
_BRAKING_CODES = {"hard_braking", "BRAKING", "evasive_braking", "evasive_speed_reduction"}
_SWERVE_CODES  = {"swerve", "SWERVE", "evasive_heading_change"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _severity_label(score: float) -> SeverityLevel:
    if score >= _HIGH_SEVERITY_THRESHOLD:
        return SeverityLevel.HIGH
    if score >= 0.4:
        return SeverityLevel.MEDIUM
    return SeverityLevel.LOW


def _risk_level_from_score(score: float) -> RiskLevel:
    if score >= 0.85:
        return RiskLevel.CRITICAL
    if score >= 0.65:
        return RiskLevel.HIGH
    if score >= 0.40:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _rec_priority_from_risk(risk_level: RiskLevel) -> RecommendationPriority:
    return {
        RiskLevel.CRITICAL: RecommendationPriority.URGENT,
        RiskLevel.HIGH:     RecommendationPriority.HIGH,
        RiskLevel.MEDIUM:   RecommendationPriority.MEDIUM,
        RiskLevel.LOW:      RecommendationPriority.LOW,
    }[risk_level]


def _events_by_type(events: List[ConflictEventInput], types: set) -> List[ConflictEventInput]:
    return [e for e in events if e.conflict_type.upper() in types]


def _high_severity_events(events: List[ConflictEventInput]) -> List[ConflictEventInput]:
    return [e for e in events if e.severity >= _HIGH_SEVERITY_THRESHOLD]


def _events_with_braking(events: List[ConflictEventInput]) -> List[ConflictEventInput]:
    return [e for e in events if any(r in _BRAKING_CODES for r in e.reasons)]


# ──────────────────────────────────────────────────────────────────────────────
# Findings
# ──────────────────────────────────────────────────────────────────────────────

def derive_key_findings(inp: RiskAnalysisInput) -> List[KeyFinding]:
    """Derive evidence-backed findings from the input. No fabrication."""
    findings: List[KeyFinding] = []
    events = inp.conflict_events
    fid = 1

    # Finding: pedestrian-vehicle conflicts dominate
    ped_veh = _events_by_type(events, _PED_VEHICLE_TYPES)
    if len(ped_veh) >= _PATTERN_MIN_FREQ:
        pct = round(100 * len(ped_veh) / len(events)) if events else 0
        high_ped = _high_severity_events(ped_veh)
        findings.append(KeyFinding(
            finding_id=f"F{fid:03d}",
            finding=(
                f"{len(ped_veh)} of {len(events)} detected conflict events involve "
                f"pedestrian-vehicle interactions ({pct}%)."
            ),
            severity=SeverityLevel.HIGH if high_ped else SeverityLevel.MEDIUM,
            evidence_event_ids=[e.event_id for e in ped_veh],
            reasoning=(
                f"{len(ped_veh)} events classified as pedestrian-vehicle conflicts. "
                f"{len(high_ped)} are high-severity (ML severity ≥ {_HIGH_SEVERITY_THRESHOLD}). "
                "Pedestrians are vulnerable road users; even moderate-severity conflicts "
                "carry elevated injury risk."
            ),
        ))
        fid += 1

    # Finding: vehicle-vehicle conflicts
    veh_veh = _events_by_type(events, _VEH_VEH_TYPES)
    if len(veh_veh) >= _PATTERN_MIN_FREQ:
        high_vv = _high_severity_events(veh_veh)
        findings.append(KeyFinding(
            finding_id=f"F{fid:03d}",
            finding=(
                f"{len(veh_veh)} vehicle-vehicle conflict events detected, "
                f"{len(high_vv)} classified as high-severity."
            ),
            severity=SeverityLevel.HIGH if high_vv else SeverityLevel.MEDIUM,
            evidence_event_ids=[e.event_id for e in veh_veh],
            reasoning=(
                f"Vehicle-vehicle interactions make up {len(veh_veh)} events. "
                f"High-severity count: {len(high_vv)}. "
                "Repeated vehicle-vehicle conflicts may indicate inadequate lane separation, "
                "sight-distance issues, or conflicting turning movements."
            ),
        ))
        fid += 1

    # Finding: motorcycle conflicts
    moto = _events_by_type(events, _MOTO_TYPES)
    if len(moto) >= _PATTERN_MIN_FREQ:
        findings.append(KeyFinding(
            finding_id=f"F{fid:03d}",
            finding=f"{len(moto)} conflict events involve motorcycles.",
            severity=SeverityLevel.MEDIUM,
            evidence_event_ids=[e.event_id for e in moto],
            reasoning=(
                "Motorcycles are vulnerable road users with reduced visibility. "
                f"{len(moto)} conflicts detected. Merging and lane-sharing patterns "
                "may contribute."
            ),
        ))
        fid += 1

    # Finding: low TTC events
    low_ttc = [
        e for e in events
        if e.ttc_min is not None and e.ttc_min < inp.ttc_threshold_s
    ]
    if low_ttc:
        min_ttc_val = min(e.ttc_min for e in low_ttc)
        findings.append(KeyFinding(
            finding_id=f"F{fid:03d}",
            finding=(
                f"{len(low_ttc)} events have TTC below the configured threshold "
                f"({inp.ttc_threshold_s}s). Minimum observed TTC: {min_ttc_val:.2f}s."
            ),
            severity=SeverityLevel.HIGH if min_ttc_val < 1.0 else SeverityLevel.MEDIUM,
            evidence_event_ids=[e.event_id for e in low_ttc],
            reasoning=(
                f"TTC < {inp.ttc_threshold_s}s indicates limited time available for road users "
                "to react. These events represent the highest-criticality proximity encounters "
                "in the observed period. Note: TTC is computed in image-space; "
                "calibrated physical values may differ."
            ),
        ))
        fid += 1

    # Finding: hard-braking concentration
    braking = _events_with_braking(events)
    if len(braking) >= _PATTERN_MIN_FREQ:
        findings.append(KeyFinding(
            finding_id=f"F{fid:03d}",
            finding=f"{len(braking)} events include detected hard-braking or evasive deceleration.",
            severity=SeverityLevel.MEDIUM,
            evidence_event_ids=[e.event_id for e in braking],
            reasoning=(
                "Repeated hard-braking events may indicate unexpected stopping conditions "
                "such as poor sight distance, sudden queue formation, or ambiguous signal phasing."
            ),
        ))
        fid += 1

    # Finding: high overall conflict frequency
    if len(events) >= 10:
        high_sev = _high_severity_events(events)
        findings.append(KeyFinding(
            finding_id=f"F{fid:03d}",
            finding=(
                f"High overall conflict frequency: {len(events)} events detected in the "
                f"observed period, {len(high_sev)} high-severity."
            ),
            severity=SeverityLevel.HIGH if len(high_sev) >= 5 else SeverityLevel.MEDIUM,
            evidence_event_ids=[e.event_id for e in high_sev or events[:5]],
            reasoning=(
                f"A total of {len(events)} conflict events were flagged by the ML pipeline. "
                f"Without exposure (volume) normalisation, high event counts may partially "
                "reflect high traffic volumes. Engineering investigation should assess both "
                "per-interaction risk and total event frequency."
            ),
        ))

    return findings


# ──────────────────────────────────────────────────────────────────────────────
# Patterns
# ──────────────────────────────────────────────────────────────────────────────

def detect_patterns(inp: RiskAnalysisInput) -> List[ConflictPattern]:
    """Detect recurring conflict patterns. Requires ≥2 events to form a pattern."""
    patterns: List[ConflictPattern] = []
    events = inp.conflict_events
    pid = 1

    # Pattern: recurring pedestrian-vehicle conflicts
    ped_veh = _events_by_type(events, _PED_VEHICLE_TYPES)
    if len(ped_veh) >= _PATTERN_MIN_FREQ:
        # Check if they cluster in the same zone
        zone_counts: Dict[str, List[str]] = {}
        for e in ped_veh:
            z = e.zone_id or "UNKNOWN"
            zone_counts.setdefault(z, []).append(e.event_id)
        dominant_zone, dominant_ids = max(zone_counts.items(), key=lambda x: len(x[1]))

        if len(dominant_ids) >= _PATTERN_MIN_FREQ:
            patterns.append(ConflictPattern(
                pattern_id=f"P{pid:03d}",
                pattern_type="RECURRING_PED_VEHICLE_ZONE",
                description=(
                    f"Recurring pedestrian-vehicle conflicts concentrated in zone "
                    f"'{dominant_zone}' ({len(dominant_ids)} of {len(ped_veh)} "
                    "pedestrian-vehicle events)."
                ),
                frequency=len(dominant_ids),
                severity=SeverityLevel.HIGH,
                evidence_event_ids=dominant_ids,
            ))
            pid += 1

    # Pattern: repeated evasive braking
    braking = _events_with_braking(events)
    if len(braking) >= _PATTERN_MIN_FREQ:
        patterns.append(ConflictPattern(
            pattern_id=f"P{pid:03d}",
            pattern_type="REPEATED_EVASIVE_BRAKING",
            description=(
                f"Evasive braking or hard deceleration detected in {len(braking)} events, "
                "suggesting repeated unexpected stopping conditions."
            ),
            frequency=len(braking),
            severity=SeverityLevel.MEDIUM,
            evidence_event_ids=[e.event_id for e in braking],
        ))
        pid += 1

    # Pattern: high-severity cluster in one zone
    for hotspot in inp.hotspots:
        if hotspot.high_risk_count >= _PATTERN_MIN_FREQ and hotspot.event_ids:
            zone_events = [e for e in events if e.event_id in set(hotspot.event_ids)]
            high_ids = [e.event_id for e in zone_events if e.severity >= _HIGH_SEVERITY_THRESHOLD]
            if len(high_ids) >= _PATTERN_MIN_FREQ:
                patterns.append(ConflictPattern(
                    pattern_id=f"P{pid:03d}",
                    pattern_type="HIGH_SEVERITY_CLUSTER",
                    description=(
                        f"Zone '{hotspot.zone_id}' contains {hotspot.high_risk_count} "
                        f"high-severity events (ML avg risk: {hotspot.avg_risk_score:.2f})."
                    ),
                    frequency=hotspot.high_risk_count,
                    severity=SeverityLevel.HIGH,
                    evidence_event_ids=high_ids,
                ))
                pid += 1

    return patterns


# ──────────────────────────────────────────────────────────────────────────────
# Hotspot prioritization
# ──────────────────────────────────────────────────────────────────────────────

def prioritize_hotspots(inp: RiskAnalysisInput) -> List[PrioritizedHotspot]:
    """Rank hotspots. ML avg_risk_score is preserved verbatim."""
    result: List[PrioritizedHotspot] = []

    # Sort by (high_risk_count, avg_risk_score) — mirrors src/hotspots/ranking.py
    ranked = sorted(
        inp.hotspots,
        key=lambda h: (h.high_risk_count, h.avg_risk_score),
        reverse=True,
    )

    for rank, h in enumerate(ranked, 1):
        risk_level = _risk_level_from_score(h.avg_risk_score)
        dominant_type = (
            max(h.conflict_type_dist, key=h.conflict_type_dist.get)
            if h.conflict_type_dist else "unknown"
        )
        evidence_ids = h.event_ids if h.event_ids else []
        if not evidence_ids:
            # Fall back to event_ids from matching conflict events
            evidence_ids = [
                e.event_id for e in inp.conflict_events
                if e.zone_id == h.zone_id
            ]
        if not evidence_ids:
            continue  # No traceable evidence — skip this zone

        result.append(PrioritizedHotspot(
            zone_id=h.zone_id,
            priority=rank,
            risk_level=risk_level,
            reason=(
                f"Zone '{h.zone_id}' has {h.event_count} conflict events "
                f"({h.high_risk_count} high-severity) with ML average risk score "
                f"{h.avg_risk_score:.2f}. Dominant conflict type: {dominant_type}."
                + (
                    f" Minimum observed TTC: {h.min_ttc:.2f}s."
                    if h.min_ttc is not None else ""
                )
            ),
            evidence_event_ids=evidence_ids[:10],  # cap list length
            ml_avg_risk_score=h.avg_risk_score,
        ))

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Recommendations
# ──────────────────────────────────────────────────────────────────────────────

def generate_recommendations(
    inp: RiskAnalysisInput,
    findings: List[KeyFinding],
) -> List[Recommendation]:
    """Generate evidence-backed recommendations. No recommendation without evidence."""
    recs: List[Recommendation] = []
    events = inp.conflict_events
    rid = 1

    def _finding_ids_for(event_ids: List[str]) -> List[str]:
        """Return finding IDs whose evidence overlaps with event_ids."""
        evt_set = set(event_ids)
        return [f.finding_id for f in findings if set(f.evidence_event_ids) & evt_set]

    # Recommendation: pedestrian crossing assessment
    ped_veh = _events_by_type(events, _PED_VEHICLE_TYPES)
    if len(ped_veh) >= _PATTERN_MIN_FREQ:
        risk = _risk_level_from_score(
            max(e.risk_score for e in ped_veh)
        )
        ids = [e.event_id for e in ped_veh]
        recs.append(Recommendation(
            recommendation_id=f"R{rid:03d}",
            category=RecommendationCategory.PEDESTRIAN_CROSSING,
            priority=_rec_priority_from_risk(risk),
            action=(
                "Prioritize this location for traffic-engineering assessment of pedestrian "
                "crossing placement, visibility, signal phasing, and crossing guard requirements."
            ),
            rationale=(
                f"{len(ped_veh)} pedestrian-vehicle conflict events were detected by the ML pipeline. "
                "Pedestrians are vulnerable road users. The observed pattern warrants assessment "
                "of whether crossing infrastructure meets current safety standards."
            ),
            evidence_event_ids=ids,
            finding_ids=_finding_ids_for(ids),
        ))
        rid += 1

        # Signal timing sub-recommendation if low TTC
        low_ttc_ped = [
            e for e in ped_veh
            if e.ttc_min is not None and e.ttc_min < inp.ttc_threshold_s
        ]
        if low_ttc_ped:
            recs.append(Recommendation(
                recommendation_id=f"R{rid:03d}",
                category=RecommendationCategory.PEDESTRIAN_CROSSING,
                priority=RecommendationPriority.HIGH,
                action=(
                    "Investigate whether pedestrian signal phase duration and lead-pedestrian "
                    "interval are adequate given the observed crossing demand."
                ),
                rationale=(
                    f"{len(low_ttc_ped)} pedestrian-vehicle events have TTC below "
                    f"{inp.ttc_threshold_s}s. Short TTC during pedestrian crossing may indicate "
                    "insufficient pedestrian phase or vehicle non-compliance."
                ),
                evidence_event_ids=[e.event_id for e in low_ttc_ped],
                finding_ids=_finding_ids_for([e.event_id for e in low_ttc_ped]),
            ))
            rid += 1

        # Traffic calming sub-recommendation
        high_ped = _high_severity_events(ped_veh)
        if high_ped:
            recs.append(Recommendation(
                recommendation_id=f"R{rid:03d}",
                category=RecommendationCategory.TRAFFIC_CALMING,
                priority=RecommendationPriority.MEDIUM,
                action=(
                    "Consider evaluating traffic-calming measures in the pedestrian conflict zone "
                    "(e.g., raised crossings, kerb extensions, reduced-speed zone)."
                ),
                rationale=(
                    f"{len(high_ped)} high-severity pedestrian-vehicle events suggest vehicle "
                    "speeds or trajectories that may not be compatible with pedestrian safety. "
                    "Physical speed management could reduce conflict severity."
                ),
                evidence_event_ids=[e.event_id for e in high_ped],
                finding_ids=_finding_ids_for([e.event_id for e in high_ped]),
            ))
            rid += 1

    # Recommendation: vehicle-vehicle lane/intersection assessment
    veh_veh = _events_by_type(events, _VEH_VEH_TYPES)
    if len(veh_veh) >= _PATTERN_MIN_FREQ:
        risk = _risk_level_from_score(max(e.risk_score for e in veh_veh))
        ids = [e.event_id for e in veh_veh]
        recs.append(Recommendation(
            recommendation_id=f"R{rid:03d}",
            category=RecommendationCategory.INTERSECTION_CONTROL,
            priority=_rec_priority_from_risk(risk),
            action=(
                "Consider assessing intersection priority, turning-conflict geometry, "
                "and sight distance for vehicle-vehicle conflict zones."
            ),
            rationale=(
                f"{len(veh_veh)} vehicle-vehicle conflict events detected. "
                "Repeated conflicts may indicate inadequate lane separation, conflicting "
                "turning movements, or sight-distance deficiencies."
            ),
            evidence_event_ids=ids,
            finding_ids=_finding_ids_for(ids),
        ))
        rid += 1

    # Recommendation: motorcycle safety
    moto = _events_by_type(events, _MOTO_TYPES)
    if len(moto) >= _PATTERN_MIN_FREQ:
        ids = [e.event_id for e in moto]
        recs.append(Recommendation(
            recommendation_id=f"R{rid:03d}",
            category=RecommendationCategory.MOTORCYCLE_SAFETY,
            priority=RecommendationPriority.MEDIUM,
            action=(
                "Investigate motorcycle movement patterns, lane-sharing behaviour, "
                "and merging conflicts in the identified zone."
            ),
            rationale=(
                f"{len(moto)} motorcycle conflict events detected. Motorcycles have "
                "reduced conspicuity and are over-represented in serious injury crashes."
            ),
            evidence_event_ids=ids,
            finding_ids=_finding_ids_for(ids),
        ))
        rid += 1

    # Recommendation: hard-braking / sight distance / signage
    braking = _events_with_braking(events)
    if len(braking) >= _PATTERN_MIN_FREQ:
        ids = [e.event_id for e in braking]
        recs.append(Recommendation(
            recommendation_id=f"R{rid:03d}",
            category=RecommendationCategory.VISIBILITY_SIGNAGE,
            priority=RecommendationPriority.MEDIUM,
            action=(
                "Investigate whether unexpected stopping conditions (sight-distance "
                "deficiencies, signage ambiguity, or queue spill-back) contribute to "
                "the observed evasive braking pattern."
            ),
            rationale=(
                f"{len(braking)} events include hard-braking or evasive deceleration. "
                "Repeated braking events at the same location may indicate a recurring "
                "hazard that is not adequately signed or visible."
            ),
            evidence_event_ids=ids,
            finding_ids=_finding_ids_for(ids),
        ))
        rid += 1

    # Recommendation: engineering review for high-frequency zones
    for h in inp.hotspots:
        if h.event_count >= 5 and h.event_ids:
            ids = h.event_ids[:10]
            recs.append(Recommendation(
                recommendation_id=f"R{rid:03d}",
                category=RecommendationCategory.ENGINEERING_REVIEW,
                priority=(
                    RecommendationPriority.HIGH
                    if h.high_risk_count >= 3 else RecommendationPriority.MEDIUM
                ),
                action=(
                    f"Prioritize zone '{h.zone_id}' for comprehensive traffic-engineering "
                    "site investigation."
                ),
                rationale=(
                    f"Zone '{h.zone_id}' recorded {h.event_count} conflict events "
                    f"({h.high_risk_count} high-severity) with ML average risk "
                    f"{h.avg_risk_score:.2f}. High conflict frequency at a single location "
                    "warrants field investigation to identify the contributing factors."
                ),
                evidence_event_ids=ids,
                finding_ids=_finding_ids_for(ids),
            ))
            rid += 1
            break  # one engineering review rec is sufficient

    return recs


# ──────────────────────────────────────────────────────────────────────────────
# Limitations
# ──────────────────────────────────────────────────────────────────────────────

def derive_limitations(inp: RiskAnalysisInput) -> List[str]:
    """Derive honest uncertainty statements from the input context."""
    lims: List[str] = []

    if not inp.camera or not inp.camera.calibration_active:
        lims.append(
            "All spatial metrics (TTC, PET, speed, distance) are computed in image-space "
            "pixels. Camera calibration was not active; physical-unit values are unavailable. "
            "Conclusions about real-world distances and speeds cannot be drawn without calibration."
        )

    lims.append(
        "TTC and PET estimates assume approximately linear motion over the prediction horizon. "
        "Non-linear trajectories (turning, crossing) may result in over- or under-estimation."
    )

    has_exposure = any(h.exposure_based for h in inp.hotspots)
    if not has_exposure:
        lims.append(
            "Conflict rates are not normalised by traffic exposure (vehicle/pedestrian counts). "
            "Higher event counts in a zone may reflect higher traffic volumes rather than "
            "higher per-interaction risk. Exposure-normalised analysis is recommended before "
            "drawing comparative conclusions between zones."
        )

    if inp.traffic_stats and inp.traffic_stats.observation_duration_s:
        dur = inp.traffic_stats.observation_duration_s
        if dur < 3600:
            lims.append(
                f"The analysis covers {dur/60:.0f} minutes of video. Short observation periods "
                "may not capture peak-hour patterns, seasonal variation, or rare conflict types."
            )
    else:
        lims.append(
            "Observation duration is not known. The analysis cannot be placed in the context "
            "of daily or seasonal traffic patterns."
        )

    lims.append(
        "This system detects evidence of potential traffic conflicts from video. It does NOT "
        "confirm that a real near-miss or injury event occurred. Detection confidence depends "
        "on object detection quality, tracking stability, and scene occlusion."
    )

    return lims


# ──────────────────────────────────────────────────────────────────────────────
# Overall assessment
# ──────────────────────────────────────────────────────────────────────────────

def compute_overall_assessment(
    inp: RiskAnalysisInput,
    findings: List[KeyFinding],
) -> OverallAssessment:
    """
    Compute the overall risk assessment from the ML metrics.

    ml_peak_risk_score is copied verbatim from the ML pipeline.
    agent_confidence reflects reasoning quality (penalised for missing calibration).
    """
    ml_peak = inp.peak_risk_score
    risk_level = _risk_level_from_score(ml_peak)

    # Agent confidence: start at 0.75, penalise for uncertainty factors
    confidence = 0.75
    if not inp.camera or not inp.camera.calibration_active:
        confidence -= 0.15  # no calibration reduces interpretation confidence
    if not any(h.exposure_based for h in inp.hotspots):
        confidence -= 0.05  # no exposure data
    if inp.traffic_stats and inp.traffic_stats.observation_duration_s:
        if inp.traffic_stats.observation_duration_s < 600:
            confidence -= 0.10  # very short observation
    confidence = round(max(0.10, min(0.95, confidence)), 2)

    high_sev = inp.high_severity_count
    total = inp.total_event_count

    # Build evidence-grounded summary
    dominant_type = (
        max(inp.conflict_type_counts, key=inp.conflict_type_counts.get)
        if inp.conflict_type_counts else "unknown"
    )

    summary = (
        f"The ML pipeline detected {total} conflict event(s) in the observed video, "
        f"of which {high_sev} are classified as high-severity "
        f"(ML peak risk score: {ml_peak:.2f}). "
    )
    if total > 0:
        summary += (
            f"The dominant conflict type is '{dominant_type}'. "
        )
    if high_sev >= 3:
        summary += (
            "The concentration of high-severity events warrants prioritization for "
            "traffic-engineering assessment."
        )
    else:
        summary += (
            "The observed conflict pattern warrants monitoring and, if sustained, "
            "further traffic-engineering review."
        )

    return OverallAssessment(
        risk_level=risk_level,
        ml_peak_risk_score=ml_peak,
        agent_confidence=confidence,
        summary=summary.strip(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_deterministic_analysis(
    inp: RiskAnalysisInput,
    model_label: str = "deterministic-fallback",
) -> AgentAnalysisOutput:
    """
    Run the full deterministic analysis.

    Called when:
      - The LLM call fails or returns invalid JSON.
      - The caller explicitly requests deterministic mode.
      - Tests need reproducible output.

    Returns a fully validated AgentAnalysisOutput.
    """
    findings     = derive_key_findings(inp)
    patterns     = detect_patterns(inp)
    hotspots     = prioritize_hotspots(inp)
    recs         = generate_recommendations(inp, findings)
    limitations  = derive_limitations(inp)
    assessment   = compute_overall_assessment(inp, findings)

    return AgentAnalysisOutput(
        analysis_id=inp.analysis_id,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_used=model_label,
        overall_assessment=assessment,
        key_findings=findings,
        patterns=patterns,
        prioritized_hotspots=hotspots,
        recommendations=recs,
        limitations=limitations,
        input_event_count=len(inp.conflict_events),
        input_hotspot_count=len(inp.hotspots),
    )
