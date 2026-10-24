"""
Phase 10 — Candidate Pair Generation.

For each frame window, identifies pairs of tracks that could plausibly interact,
using cheap spatial + kinematic filters before running expensive safety metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.logger import get_logger
from src.schema import CandidateInteraction, Trajectory, TrackQuality

log = get_logger(__name__)

# Compatible interaction class pairs
_COMPATIBLE: List[Tuple[str, str]] = [
    ("person",     "car"),
    ("person",     "motorcycle"),
    ("person",     "bus"),
    ("person",     "truck"),
    ("car",        "car"),
    ("car",        "motorcycle"),
    ("car",        "bus"),
    ("car",        "truck"),
    ("motorcycle", "car"),
    ("motorcycle", "motorcycle"),
    ("motorcycle", "bus"),
    ("motorcycle", "truck"),
    ("bus",        "truck"),
    ("truck",      "bus"),
]
_COMPATIBLE_SET = {(a, b) for a, b in _COMPATIBLE} | {(b, a) for a, b in _COMPATIBLE}


def interaction_type(class_a: str, class_b: str) -> str:
    """
    Classify semantic interaction type based on object classes.
    Keeps pedestrian/person interactions strictly separate from motorcycle-vehicle interactions.

    Returns:
        "VEH_PED": If a pedestrian/person (or bicycle) is involved.
        "VEH_MOTO": If a motorcycle is involved (with vehicle or motorcycle).
        "VEH_VEH": Vehicle-to-vehicle interactions (car, bus, truck).
    """
    ped_classes = {"person", "pedestrian", "bicycle"}
    moto_classes = {"motorcycle"}

    is_ped = (class_a in ped_classes) or (class_b in ped_classes)
    is_moto = (class_a in moto_classes) or (class_b in moto_classes)

    if is_ped:
        return "VEH_PED"
    if is_moto:
        return "VEH_MOTO"
    return "VEH_VEH"


def generate_candidates(
    trajectories: Dict[int, Trajectory],
    max_distance: float = 150.0,
    fps: float = 30.0,
) -> List[CandidateInteraction]:
    """
    Find all (track_a, track_b) pairs where:
        1. Both tracks are GOOD quality.
        2. Classes are compatible for interaction.
        3. They are within max_distance pixels of each other at some shared timestamp.
        4. Relative motion indicates approach (closing distance).

    Returns one CandidateInteraction per pair (spanning their shared time window).
    """
    good_tracks = {tid: t for tid, t in trajectories.items() if t.quality == TrackQuality.GOOD}
    tids = list(good_tracks.keys())
    candidates: List[CandidateInteraction] = []

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            ta = good_tracks[tids[i]]
            tb = good_tracks[tids[j]]

            # Class compatibility
            pair = (ta.class_name, tb.class_name)
            if pair not in _COMPATIBLE_SET:
                continue

            # Find shared time window
            t_start = max(ta.timestamps[0], tb.timestamps[0])
            t_end   = min(ta.timestamps[-1], tb.timestamps[-1])
            if t_end - t_start < 0.5:   # less than 0.5 s overlap
                continue

            # Sample positions at shared timestamps
            shared_ts = _shared_timestamps(ta.timestamps, tb.timestamps)
            if len(shared_ts) < 2:
                continue

            pos_a = _interp_positions(ta.timestamps, ta.positions, shared_ts)
            pos_b = _interp_positions(tb.timestamps, tb.positions, shared_ts)

            dists = np.linalg.norm(pos_a - pos_b, axis=1)
            min_dist = float(np.min(dists))

            if min_dist > max_distance:
                continue

            # Check approach: sustained distance decrease (not just one noisy step)
            if not _has_sustained_approach(dists):
                continue

            candidates.append(CandidateInteraction(
                track_id_a=ta.track_id,
                track_id_b=tb.track_id,
                class_a=ta.class_name,
                class_b=tb.class_name,
                interaction_type=interaction_type(ta.class_name, tb.class_name),
                start_time=float(t_start),
                end_time=float(t_end),
                start_frame=int(t_start * fps),
                end_frame=int(t_end * fps),
            ))

    log.info("Candidate interactions: %d (from %d GOOD tracks)", len(candidates), len(good_tracks))
    return candidates


def _shared_timestamps(ts_a: np.ndarray, ts_b: np.ndarray, n_samples: int = 30) -> np.ndarray:
    t_start = max(ts_a[0], ts_b[0])
    t_end   = min(ts_a[-1], ts_b[-1])
    if t_end <= t_start:
        return np.array([])
    return np.linspace(t_start, t_end, min(n_samples, int((t_end - t_start) * 10) + 2))


def _interp_positions(timestamps: np.ndarray, positions: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    x = np.interp(query_ts, timestamps, positions[:, 0])
    y = np.interp(query_ts, timestamps, positions[:, 1])
    return np.stack([x, y], axis=1)



def _has_sustained_approach(
    dists: np.ndarray,
    min_consecutive: int = 3,
    min_fraction: float = 0.25,
) -> bool:
    """
    FIX #6 — Require a stable, sustained approach rather than a single closing step.

    Conditions (BOTH must be satisfied):
      1. There exists a run of at least `min_consecutive` consecutive samples
         where distance is monotonically decreasing.
      2. At least `min_fraction` of all consecutive-sample pairs show distance
         decreasing by more than 2 px (so isolated noise does not trigger).

    This prevents a brief oscillation or detector jitter from creating a candidate.
    """
    if len(dists) < 2:
        return False

    deltas = np.diff(dists)         # negative = closing

    # Condition 2: fraction of pairs showing meaningful closing
    closing = deltas < -2.0
    if closing.mean() < min_fraction:
        return False

    # Condition 1: longest consecutive run of closing steps
    max_run = 0
    current_run = 0
    for c in closing:
        if c:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run >= min_consecutive

