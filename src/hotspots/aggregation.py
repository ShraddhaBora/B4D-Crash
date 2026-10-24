"""
Phase 18 — Hotspot Aggregation.

Groups ConflictEvents by zone and computes per-zone safety statistics.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from src.logger import get_logger
from src.schema import ConflictEvent, HotspotSummary

log = get_logger(__name__)


def aggregate_hotspots(
    events: List[ConflictEvent],
    observation_duration_s: float,
) -> List[HotspotSummary]:
    """
    Aggregate conflict events by zone_id.

    Args:
        events:                All ConflictEvent records.
        observation_duration_s: Total video duration (seconds) — used for rate calculation.

    Returns sorted list of HotspotSummary (highest risk first).
    """
    by_zone: Dict[str, List[ConflictEvent]] = defaultdict(list)
    for ev in events:
        zid = ev.zone_id or "UNKNOWN"
        by_zone[zid].append(ev)

    summaries: List[HotspotSummary] = []
    for zone_id, zone_events in by_zone.items():
        high_risk = [e for e in zone_events if e.severity >= 0.7]
        risk_scores = [e.risk_score for e in zone_events]
        ttc_vals = [e.ttc_min for e in zone_events if e.ttc_min < math.inf]
        pet_vals = [e.pet_min for e in zone_events if e.pet_min < math.inf]

        conflict_types: Dict[str, int] = defaultdict(int)
        road_users: Dict[str, int] = defaultdict(int)
        for ev in zone_events:
            conflict_types[ev.conflict_type] += 1
            for cls in ev.classes:
                road_users[cls] += 1

        summaries.append(HotspotSummary(
            zone_id=zone_id,
            event_count=len(zone_events),
            high_risk_count=len(high_risk),
            avg_risk_score=round(sum(risk_scores) / len(risk_scores), 4),
            min_ttc=min(ttc_vals) if ttc_vals else math.inf,
            min_pet=min(pet_vals) if pet_vals else math.inf,
            conflict_type_dist=dict(conflict_types),
            road_user_dist=dict(road_users),
            observation_duration_s=observation_duration_s,
            exposure_based=False,   # raw-count ranking until exposure data available
        ))

    summaries.sort(key=lambda s: (s.high_risk_count, s.avg_risk_score), reverse=True)
    log.info("Hotspot aggregation: %d zones with events.", len(summaries))
    return summaries
