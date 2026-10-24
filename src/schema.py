"""
before_the_crash — shared typed data structures.

All pipeline modules import from here. Keeps inter-module contracts explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Perception layer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    frame_id: int
    timestamp: float          # seconds from video start
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xywh(self) -> Tuple[float, float, float, float]:
        return self.center_x, self.center_y, self.width, self.height


@dataclass
class TrackObservation:
    """A single tracked bounding-box observation at one time step."""
    frame_id: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def foot_x(self) -> float:
        """Estimated road-contact x — mid-bottom of bounding box."""
        return (self.x1 + self.x2) / 2.0

    @property
    def foot_y(self) -> float:
        """Estimated road-contact y — bottom edge of bounding box."""
        return self.y2


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory layer
# ──────────────────────────────────────────────────────────────────────────────

class TrackQuality:
    GOOD = "GOOD"
    SHORT = "SHORT"
    FRAGMENTED = "FRAGMENTED"
    IMPLAUSIBLE = "IMPLAUSIBLE"


@dataclass
class Trajectory:
    track_id: int
    class_name: str
    quality: str                                # TrackQuality constant
    timestamps: np.ndarray                      # shape (N,)  seconds
    raw_positions: np.ndarray                   # shape (N, 2)  [x, y] pixels
    positions: np.ndarray                       # shape (N, 2)  smoothed
    velocities: np.ndarray                      # shape (N, 2)  px/s
    speeds: np.ndarray                          # shape (N,)    px/s  scalar
    headings: np.ndarray                        # shape (N,)    degrees  0=right CW
    accelerations: np.ndarray                   # shape (N,)    px/s²


# ──────────────────────────────────────────────────────────────────────────────
# Interaction / Safety layer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateInteraction:
    track_id_a: int
    track_id_b: int
    class_a: str
    class_b: str
    interaction_type: str         # e.g. "VEH_PED" | "VEH_VEH"
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int


@dataclass
class TTCResult:
    value: float                  # seconds; np.inf if undefined
    valid: bool
    model: str                    # "linear_closing" | "closest_approach"
    confidence: float             # 0..1


@dataclass
class PETResult:
    value: float                  # seconds; np.inf if undefined
    valid: bool
    conflict_point: Optional[Tuple[float, float]]
    confidence: float


@dataclass
class EvasiveAction:
    track_id: int
    timestamp: float
    action_type: str              # "BRAKING" | "SWERVE" | "STOP"
    speed_change_ratio: float     # Δspeed / speed_before  (negative = deceleration)
    heading_change_deg: float


# ──────────────────────────────────────────────────────────────────────────────
# Conflict / Event layer
# ──────────────────────────────────────────────────────────────────────────────

class ConflictClass:
    NORMAL = "NORMAL"
    POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT"
    HIGH_RISK_CONFLICT = "HIGH_RISK_CONFLICT"


@dataclass
class ConflictResult:
    classification: str            # ConflictClass constant
    severity: float                # 0..1
    confidence: float              # 0..1
    reasons: List[str]             # machine-readable codes
    metrics: dict                  # raw metric values for transparency


@dataclass
class RiskScore:
    track_id_a: int
    track_id_b: int
    ttc_score: float
    pet_score: float
    evasive_score: float
    vulnerability_score: float
    overall_score: float


@dataclass
class ConflictEvent:
    event_id: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    track_ids: List[int]
    classes: List[str]
    conflict_type: str
    zone_id: Optional[str]
    ttc_min: float
    pet_min: float
    max_speed_change: float
    max_heading_change: float
    severity: float
    confidence: float
    risk_score: float
    reasons: List[str]


# ──────────────────────────────────────────────────────────────────────────────
# Hotspot layer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HotspotSummary:
    zone_id: str
    event_count: int
    high_risk_count: int
    avg_risk_score: float
    min_ttc: float
    min_pet: float
    conflict_type_dist: dict       # {"VEH_PED": 3, "VEH_VEH": 1}
    road_user_dist: dict
    observation_duration_s: float
    exposure_based: bool           # False = raw count ranking
