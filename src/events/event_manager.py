"""
Phase 16 — Event Manager.

FIX #3 — The original EventManager created one event per track pair regardless
of how many temporal windows were sampled. This is not true temporal alert
merging; it is just per-pair metric accumulation.

Refactored so that:
  - Each call to record() represents ONE frame-level or window-level alert with
    a specific timestamp.
  - The manager accumulates frame-level alerts per (track_a, track_b) pair.
  - finalize() merges consecutive alerts that are within merge_window seconds
    of each other into a single ConflictEvent, then drops events shorter than
    minimum_duration.
  - The same ConflictEvent schema is preserved unchanged.

This means the pipeline can now sample safety metrics at multiple timestamps
per candidate (Fix #2) and feed each sample as a separate record() call,
producing events that accurately reflect the temporal extent of high-risk windows.
"""

from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional, Tuple

from src.logger import get_logger
from src.schema import (
    CandidateInteraction, ConflictClass, ConflictEvent, ConflictResult,
    EvasiveAction, PETResult, RiskScore, TTCResult,
)

log = get_logger(__name__)


class _FrameAlert:
    """A single frame-level / timestamp-level conflict alert."""

    __slots__ = (
        "timestamp", "ttc", "pet", "evasive", "result", "risk", "zone_id",
    )

    def __init__(
        self,
        timestamp: float,
        ttc: TTCResult,
        pet: PETResult,
        evasive: List[EvasiveAction],
        result: ConflictResult,
        risk: RiskScore,
        zone_id: Optional[str],
    ):
        self.timestamp = timestamp
        self.ttc = ttc
        self.pet = pet
        self.evasive = evasive
        self.result = result
        self.risk = risk
        self.zone_id = zone_id


class _PairRecord:
    """Accumulates frame-level alerts for one (track_a, track_b) pair."""

    def __init__(self, candidate: CandidateInteraction):
        self.candidate = candidate
        self.alerts: List[_FrameAlert] = []

    def add(
        self,
        timestamp: float,
        ttc: TTCResult,
        pet: PETResult,
        evasive: List[EvasiveAction],
        result: ConflictResult,
        risk: RiskScore,
        zone_id: Optional[str],
    ) -> None:
        self.alerts.append(_FrameAlert(timestamp, ttc, pet, evasive, result, risk, zone_id))

    def to_events(self, merge_window: float, minimum_duration: float) -> List[ConflictEvent]:
        """
        Merge consecutive alerts into ConflictEvents by temporal proximity.

        Alerts are sorted by timestamp.  Whenever the gap between consecutive
        alerts exceeds merge_window, a new event is started.
        """
        if not self.alerts:
            return []

        # Sort chronologically
        sorted_alerts = sorted(self.alerts, key=lambda a: a.timestamp)

        # Group into temporal clusters
        clusters: List[List[_FrameAlert]] = []
        current: List[_FrameAlert] = [sorted_alerts[0]]
        for alert in sorted_alerts[1:]:
            if alert.timestamp - current[-1].timestamp <= merge_window:
                current.append(alert)
            else:
                clusters.append(current)
                current = [alert]
        clusters.append(current)

        events: List[ConflictEvent] = []
        for cluster in clusters:
            ev = self._cluster_to_event(cluster)
            if ev is None:
                continue
            duration = ev.end_time - ev.start_time
            if duration < minimum_duration:
                log.debug(
                    "Dropped short event (%.2fs): tracks %d + %d",
                    duration, self.candidate.track_id_a, self.candidate.track_id_b,
                )
                continue
            # Drop NORMAL events with no reasons
            if ev.conflict_type == ConflictClass.NORMAL and not ev.reasons:
                continue
            events.append(ev)

        return events

    def _cluster_to_event(self, alerts: List[_FrameAlert]) -> Optional[ConflictEvent]:
        if not alerts:
            return None

        ttc_values:       List[float] = []
        pet_values:       List[float] = []
        speed_changes:    List[float] = []
        heading_changes:  List[float] = []
        classifications:  List[str]   = []
        severities:       List[float] = []
        confidences:      List[float] = []
        risk_scores:      List[float] = []
        reasons_union:    set          = set()
        zone_id:          Optional[str] = None
        peak_conflict_t:  Optional[float] = None
        best_ttc: float = math.inf

        for alert in alerts:
            if alert.ttc.valid:
                ttc_values.append(alert.ttc.value)
                if alert.ttc.value < best_ttc:
                    best_ttc = alert.ttc.value
                    peak_conflict_t = alert.timestamp
            if alert.pet.valid:
                pet_values.append(alert.pet.value)
            for a in alert.evasive:
                speed_changes.append(abs(a.speed_change_ratio))
                heading_changes.append(abs(a.heading_change_deg))
            classifications.append(alert.result.classification)
            severities.append(alert.result.severity)
            confidences.append(alert.result.confidence)
            risk_scores.append(alert.risk.overall_score)
            reasons_union.update(alert.result.reasons)
            if alert.zone_id:
                zone_id = alert.zone_id

        # Peak classification (most severe wins)
        for cls in (ConflictClass.HIGH_RISK_CONFLICT,
                    ConflictClass.POTENTIAL_CONFLICT,
                    ConflictClass.NORMAL):
            if cls in classifications:
                peak_cls = cls
                break
        else:
            peak_cls = ConflictClass.NORMAL

        start_t = float(alerts[0].timestamp)
        end_t   = float(alerts[-1].timestamp)

        # Fall back to candidate window boundaries if single-point alert
        if start_t == end_t:
            start_t = min(start_t, self.candidate.start_time)
            end_t   = max(end_t,   self.candidate.end_time)

        fps_approx = 30.0  # used only for frame indices
        return ConflictEvent(
            event_id=str(uuid.uuid4())[:8],
            start_time=start_t,
            end_time=end_t,
            start_frame=int(start_t * fps_approx),
            end_frame=int(end_t * fps_approx),
            track_ids=[self.candidate.track_id_a, self.candidate.track_id_b],
            classes=[self.candidate.class_a, self.candidate.class_b],
            conflict_type=self.candidate.interaction_type,
            zone_id=zone_id,
            ttc_min=min(ttc_values) if ttc_values else math.inf,
            pet_min=min(pet_values) if pet_values else math.inf,
            max_speed_change=max(speed_changes)   if speed_changes   else 0.0,
            max_heading_change=max(heading_changes) if heading_changes else 0.0,
            severity=max(severities)   if severities   else 0.0,
            confidence=max(confidences) if confidences else 0.0,
            risk_score=max(risk_scores) if risk_scores else 0.0,
            reasons=list(reasons_union),
            peak_conflict_time=peak_conflict_t,
        )


class EventManager:
    """
    Accumulates per-candidate frame-level conflict alerts and merges them into
    ConflictEvents grouped by temporal proximity.

    Call .record() for each (candidate, timestamp) sample after safety analysis.
    Call .finalize() to get the deduplicated event list.
    """

    def __init__(self, merge_window: float = 2.0, minimum_duration: float = 0.2):
        self.merge_window = merge_window
        self.minimum_duration = minimum_duration
        self._records: Dict[Tuple[int, int], _PairRecord] = {}

    def record(
        self,
        candidate: CandidateInteraction,
        ttc: TTCResult,
        pet: PETResult,
        evasive: List[EvasiveAction],
        result: ConflictResult,
        risk: RiskScore,
        zone_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record one frame-level alert for the given candidate pair.

        Args:
            timestamp: Alert timestamp (seconds). Defaults to candidate midpoint
                       for backwards compatibility with single-call usage.
        """
        key = (min(candidate.track_id_a, candidate.track_id_b),
               max(candidate.track_id_a, candidate.track_id_b))
        if key not in self._records:
            self._records[key] = _PairRecord(candidate)

        t = timestamp if timestamp is not None else (
            (candidate.start_time + candidate.end_time) / 2.0
        )
        self._records[key].add(t, ttc, pet, evasive, result, risk, zone_id)

    def finalize(self) -> List[ConflictEvent]:
        events: List[ConflictEvent] = []
        for rec in self._records.values():
            events.extend(rec.to_events(self.merge_window, self.minimum_duration))

        # Sort by risk score descending
        events.sort(key=lambda e: e.risk_score, reverse=True)
        log.info(
            "Events finalised: %d  (high-risk: %d)",
            len(events),
            sum(1 for e in events if e.severity >= 0.7),
        )
        return events
