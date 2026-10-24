"""
Phase 16 — Event Manager.

Merges repeated frame-level conflict alerts into deduplicated ConflictEvent records.
The same interaction can produce dozens of alerts per second; we collapse them
into one event with min/max/peak metrics.
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


class _AlertRecord:
    """Accumulates per-frame alert data for one candidate pair."""

    def __init__(self, candidate: CandidateInteraction):
        self.candidate = candidate
        self.ttc_values: List[float] = []
        self.pet_values: List[float] = []
        self.speed_changes: List[float] = []
        self.heading_changes: List[float] = []
        self.classifications: List[str] = []
        self.severities: List[float] = []
        self.confidences: List[float] = []
        self.risk_scores: List[float] = []
        self.reasons_union: set = set()
        self.zone_id: Optional[str] = None
        self.start_frame: int = candidate.start_frame
        self.end_frame:   int = candidate.end_frame
        self.start_time:  float = candidate.start_time
        self.end_time:    float = candidate.end_time

    def add(self, ttc: TTCResult, pet: PETResult,
            evasive: List[EvasiveAction], result: ConflictResult,
            risk: RiskScore, zone_id: Optional[str]) -> None:
        if ttc.valid:
            self.ttc_values.append(ttc.value)
        if pet.valid:
            self.pet_values.append(pet.value)
        for a in evasive:
            self.speed_changes.append(abs(a.speed_change_ratio))
            self.heading_changes.append(abs(a.heading_change_deg))
        self.classifications.append(result.classification)
        self.severities.append(result.severity)
        self.confidences.append(result.confidence)
        self.risk_scores.append(risk.overall_score)
        self.reasons_union.update(result.reasons)
        if zone_id:
            self.zone_id = zone_id

    def to_event(self) -> ConflictEvent:
        # Peak classification
        for cls in (ConflictClass.HIGH_RISK_CONFLICT,
                    ConflictClass.POTENTIAL_CONFLICT,
                    ConflictClass.NORMAL):
            if cls in self.classifications:
                peak_cls = cls
                break
        else:
            peak_cls = ConflictClass.NORMAL

        return ConflictEvent(
            event_id=str(uuid.uuid4())[:8],
            start_time=self.start_time,
            end_time=self.end_time,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            track_ids=[self.candidate.track_id_a, self.candidate.track_id_b],
            classes=[self.candidate.class_a, self.candidate.class_b],
            conflict_type=self.candidate.interaction_type,
            zone_id=self.zone_id,
            ttc_min=min(self.ttc_values) if self.ttc_values else math.inf,
            pet_min=min(self.pet_values) if self.pet_values else math.inf,
            max_speed_change=max(self.speed_changes) if self.speed_changes else 0.0,
            max_heading_change=max(self.heading_changes) if self.heading_changes else 0.0,
            severity=max(self.severities) if self.severities else 0.0,
            confidence=max(self.confidences) if self.confidences else 0.0,
            risk_score=max(self.risk_scores) if self.risk_scores else 0.0,
            reasons=list(self.reasons_union),
        )


class EventManager:
    """
    Accumulates per-candidate conflict data and merges into ConflictEvents.

    Call .record() for each candidate after safety analysis.
    Call .finalize() to get the deduplicated event list.
    """

    def __init__(self, merge_window: float = 2.0, minimum_duration: float = 0.2):
        self.merge_window = merge_window
        self.minimum_duration = minimum_duration
        self._records: Dict[Tuple[int, int], _AlertRecord] = {}

    def record(
        self,
        candidate: CandidateInteraction,
        ttc: TTCResult,
        pet: PETResult,
        evasive: List[EvasiveAction],
        result: ConflictResult,
        risk: RiskScore,
        zone_id: Optional[str] = None,
    ) -> None:
        key = (min(candidate.track_id_a, candidate.track_id_b),
               max(candidate.track_id_a, candidate.track_id_b))
        if key not in self._records:
            self._records[key] = _AlertRecord(candidate)
        self._records[key].add(ttc, pet, evasive, result, risk, zone_id)

    def finalize(self) -> List[ConflictEvent]:
        events: List[ConflictEvent] = []
        for rec in self._records.values():
            if rec.end_time - rec.start_time < self.minimum_duration:
                log.debug("Dropped short event (%.2fs): tracks %d + %d",
                          rec.end_time - rec.start_time,
                          rec.candidate.track_id_a, rec.candidate.track_id_b)
                continue
            ev = rec.to_event()
            if ev.conflict_type == ConflictClass.NORMAL and not ev.reasons:
                continue
            events.append(ev)

        # Sort by risk score descending
        events.sort(key=lambda e: e.risk_score, reverse=True)
        log.info("Events finalised: %d  (high-risk: %d)",
                 len(events),
                 sum(1 for e in events if e.severity >= 0.7))
        return events
