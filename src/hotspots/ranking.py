"""
Phase 19 — Exposure-Normalised Ranking.

Ranks hotspot zones.  When reliable exposure data is available (pedestrian
crossing counts, vehicle interactions), produces a conflict rate.
Otherwise, ranks by raw event count with a clear warning.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.logger import get_logger
from src.schema import HotspotSummary

log = get_logger(__name__)


def rank_hotspots(
    summaries: List[HotspotSummary],
    exposure: Optional[Dict[str, float]] = None,
) -> List[HotspotSummary]:
    """
    Sort hotspot summaries by risk priority.

    If `exposure` dict (zone_id → interaction count) is provided,
    compute conflict_rate = event_count / exposure[zone_id].
    Otherwise, rank by (high_risk_count, avg_risk_score) — raw observation count.
    """
    if exposure:
        def key(s: HotspotSummary):
            exp = exposure.get(s.zone_id, 1.0)
            rate = s.event_count / max(exp, 1.0)
            return (rate, s.avg_risk_score)
        log.info("Ranking hotspots by exposure-normalised conflict rate.")
    else:
        def key(s: HotspotSummary):
            return (s.high_risk_count, s.avg_risk_score)
        log.warning(
            "No exposure data provided — ranking hotspots by raw event count. "
            "Interpret with caution; high-traffic zones will rank higher."
        )

    ranked = sorted(summaries, key=key, reverse=True)
    for i, s in enumerate(ranked, 1):
        log.info(
            "  #%d  zone=%-10s  events=%d  high_risk=%d  avg_risk=%.3f",
            i, s.zone_id, s.event_count, s.high_risk_count, s.avg_risk_score,
        )
    return ranked
