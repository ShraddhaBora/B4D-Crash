"""
agent/schemas.py — Pydantic v2 data contracts for Agent37.

INPUT  : RiskAnalysisInput   — built from ML risk-engine outputs.
OUTPUT : AgentAnalysisOutput — Agent37's evidence-grounded analysis.

Design rules:
  1. ML-produced fields (risk_score, ttc_min, pet_min, severity) are
     carried through unchanged.  Agent37 NEVER overwrites them.
  2. Agent37 adds its own `agent_confidence` (0..1) distinct from
     `ml_risk_score` / `severity`.
  3. Every finding, pattern, and recommendation carries
     `evidence_event_ids` so the frontend can link back to raw events.
  4. Optional fields use Optional[T] = None — no silent guessing.
  5. Enum literals for risk level / severity keep validation strict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# Shared enums
# ──────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class SeverityLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class RecommendationPriority(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    URGENT = "URGENT"


class RecommendationCategory(str, Enum):
    PEDESTRIAN_CROSSING  = "PEDESTRIAN_CROSSING"
    INTERSECTION_CONTROL = "INTERSECTION_CONTROL"
    LANE_GEOMETRY        = "LANE_GEOMETRY"
    SPEED_ENVIRONMENT    = "SPEED_ENVIRONMENT"
    VISIBILITY_SIGNAGE   = "VISIBILITY_SIGNAGE"
    TRAFFIC_CALMING      = "TRAFFIC_CALMING"
    MOTORCYCLE_SAFETY    = "MOTORCYCLE_SAFETY"
    ENGINEERING_REVIEW   = "ENGINEERING_REVIEW"


# ──────────────────────────────────────────────────────────────────────────────
# INPUT schema — mirrors ConflictEvent / HotspotSummary from src/schema.py
# ──────────────────────────────────────────────────────────────────────────────

class ConflictEventInput(BaseModel):
    """
    Single conflict event from the ML risk engine.
    Field names deliberately mirror src.schema.ConflictEvent.
    """
    event_id:           str
    start_time:         float                   = Field(ge=0.0)
    end_time:           float                   = Field(ge=0.0)
    start_frame:        Optional[int]           = None
    end_frame:          Optional[int]           = None
    track_ids:          List[int]               = Field(min_length=2)
    classes:            List[str]               = Field(min_length=2)
    conflict_type:      str                     # e.g. "VEH_PED", "VEH_VEH"
    zone_id:            Optional[str]           = None
    # ── ML-produced safety metrics (authoritative; never modified by Agent37) ─
    ttc_min:            Optional[float]         = None   # seconds; None = not applicable
    pet_min:            Optional[float]         = None   # seconds; None = not applicable
    max_speed_change:   Optional[float]         = None   # fraction of speed
    max_heading_change: Optional[float]         = None   # degrees
    severity:           float                   = Field(ge=0.0, le=1.0)
    confidence:         float                   = Field(ge=0.0, le=1.0)
    risk_score:         float                   = Field(ge=0.0, le=1.0)  # ML authoritative
    reasons:            List[str]               = Field(default_factory=list)
    peak_conflict_time: Optional[float]         = None
    # ── Evidence ──────────────────────────────────────────────────────────────
    evidence_clip_path: Optional[str]           = None

    @model_validator(mode="after")
    def end_after_start(self) -> "ConflictEventInput":
        if self.end_time < self.start_time:
            raise ValueError(
                f"Event {self.event_id}: end_time ({self.end_time}) < "
                f"start_time ({self.start_time})"
            )
        return self


class HotspotInput(BaseModel):
    """
    Per-zone aggregated summary from src.hotspots.aggregation.
    Field names deliberately mirror src.schema.HotspotSummary.
    """
    zone_id:                str
    event_count:            int                 = Field(ge=0)
    high_risk_count:        int                 = Field(ge=0)
    avg_risk_score:         float               = Field(ge=0.0, le=1.0)
    min_ttc:                Optional[float]     = None
    min_pet:                Optional[float]     = None
    conflict_type_dist:     Dict[str, int]      = Field(default_factory=dict)
    road_user_dist:         Dict[str, int]      = Field(default_factory=dict)
    observation_duration_s: float               = Field(ge=0.0)
    exposure_based:         bool                = False
    centroid_x:             Optional[float]     = None
    centroid_y:             Optional[float]     = None
    event_ids:              List[str]           = Field(default_factory=list)


class TrafficStats(BaseModel):
    """Session-level traffic statistics."""
    total_frames:           Optional[int]       = None
    total_detections:       Optional[int]       = None
    unique_track_count:     Optional[int]       = None
    observation_duration_s: Optional[float]     = None
    fps:                    Optional[float]     = None


class CameraInfo(BaseModel):
    """Camera / location metadata."""
    camera_id:          Optional[str]   = None
    location_name:      Optional[str]   = None
    location_lat:       Optional[float] = None
    location_lon:       Optional[float] = None
    mount_height_m:     Optional[float] = None
    frame_width_px:     Optional[int]   = None
    frame_height_px:    Optional[int]   = None
    calibration_active: bool            = False


class RiskAnalysisInput(BaseModel):
    """
    Complete input to Agent37.

    This is the boundary between the ML pipeline and the agent layer.
    The adapter (risk_adapter.py) builds this from:
      - conflict_events.csv  (real pipeline) or dummy_risk_output.json
      - hotspot_summary.csv  (real pipeline) or dummy_risk_output.json

    Replacing dummy data with real CSV outputs requires only adapter changes;
    Agent37 itself always sees this schema.
    """
    analysis_id:    str
    session_id:     Optional[str]               = None
    analysis_ts:    Optional[str]               = None   # ISO-8601 string
    video_path:     Optional[str]               = None

    camera:         Optional[CameraInfo]        = None
    traffic_stats:  Optional[TrafficStats]      = None

    # Safety configuration thresholds (from config.yaml)
    ttc_threshold_s:  float = Field(default=3.0, ge=0.0)
    pet_threshold_s:  float = Field(default=2.0, ge=0.0)

    conflict_events: List[ConflictEventInput]   = Field(default_factory=list)
    hotspots:        List[HotspotInput]         = Field(default_factory=list)

    # Aggregate convenience fields computed by the adapter
    total_event_count:      int                 = Field(default=0, ge=0)
    high_severity_count:    int                 = Field(default=0, ge=0)
    peak_risk_score:        float               = Field(default=0.0, ge=0.0, le=1.0)
    avg_risk_score:         float               = Field(default=0.0, ge=0.0, le=1.0)
    conflict_type_counts:   Dict[str, int]      = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_aggregate_fields(self) -> "RiskAnalysisInput":
        """Auto-populate aggregate fields if not explicitly set."""
        events = self.conflict_events
        if events and self.total_event_count == 0:
            self.total_event_count = len(events)
        if events and self.high_severity_count == 0:
            self.high_severity_count = sum(1 for e in events if e.severity >= 0.7)
        if events and self.peak_risk_score == 0.0:
            self.peak_risk_score = max((e.risk_score for e in events), default=0.0)
        if events and self.avg_risk_score == 0.0:
            scores = [e.risk_score for e in events]
            self.avg_risk_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        if events and not self.conflict_type_counts:
            counts: Dict[str, int] = {}
            for e in events:
                counts[e.conflict_type] = counts.get(e.conflict_type, 0) + 1
            self.conflict_type_counts = counts
        return self


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT schema — structured Agent37 analysis
# ──────────────────────────────────────────────────────────────────────────────

class OverallAssessment(BaseModel):
    """
    Top-level risk assessment.

    ml_peak_risk_score  — verbatim from the ML pipeline (never modified).
    agent_confidence    — Agent37's confidence in its own interpretation (0..1).
    risk_level          — derived categorical label (Agent37 assigns this).
    summary             — evidence-grounded narrative (1-3 sentences).
    """
    risk_level:         RiskLevel
    ml_peak_risk_score: float       = Field(ge=0.0, le=1.0)
    agent_confidence:   float       = Field(ge=0.0, le=1.0)
    summary:            str


class KeyFinding(BaseModel):
    """An individual evidence-backed observation."""
    finding_id:         str
    finding:            str
    severity:           SeverityLevel
    evidence_event_ids: List[str]   = Field(min_length=1)
    reasoning:          str


class ConflictPattern(BaseModel):
    """A recurring conflict pattern detected across multiple events."""
    pattern_id:         str
    pattern_type:       str
    description:        str
    frequency:          int         = Field(ge=2)
    severity:           SeverityLevel
    evidence_event_ids: List[str]   = Field(min_length=2)


class PrioritizedHotspot(BaseModel):
    """A zone prioritized for investigation with supporting rationale."""
    zone_id:            str
    priority:           int         = Field(ge=1)
    risk_level:         RiskLevel
    reason:             str
    evidence_event_ids: List[str]   = Field(min_length=1)
    ml_avg_risk_score:  float       = Field(ge=0.0, le=1.0)


class Recommendation(BaseModel):
    """
    An evidence-backed road-safety intervention recommendation.

    Language must be cautious — 'prioritize for assessment', 'consider
    evaluating', 'investigate whether' — not 'is definitely required'.
    """
    recommendation_id:  str
    category:           RecommendationCategory
    priority:           RecommendationPriority
    action:             str
    rationale:          str
    evidence_event_ids: List[str]   = Field(min_length=1)
    finding_ids:        List[str]   = Field(default_factory=list)


class AgentAnalysisOutput(BaseModel):
    """
    Complete Agent37 analysis output.

    All ML-produced numerical safety metrics are preserved verbatim;
    Agent37 adds interpretation, patterns, hotspot prioritization,
    recommendations, and explicit uncertainty statements.
    """
    analysis_id:        str
    agent_version:      str         = "37.0"
    generated_at:       str
    model_used:         str

    overall_assessment:   OverallAssessment
    key_findings:         List[KeyFinding]          = Field(default_factory=list)
    patterns:             List[ConflictPattern]     = Field(default_factory=list)
    prioritized_hotspots: List[PrioritizedHotspot]  = Field(default_factory=list)
    recommendations:      List[Recommendation]      = Field(default_factory=list)

    limitations:          List[str]                 = Field(default_factory=list)

    input_event_count:    int       = Field(ge=0)
    input_hotspot_count:  int       = Field(ge=0)
    llm_raw_response:     Optional[str] = None

    @model_validator(mode="after")
    def at_least_one_limitation(self) -> "AgentAnalysisOutput":
        if not self.limitations:
            raise ValueError(
                "AgentAnalysisOutput must contain at least one explicit limitation statement."
            )
        return self
