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
    vru = {"person", "bicycle", "motorcycle"}
    a_vru = class_a in vru
    b_vru = class_b in vru
    if a_vru or b_vru:
        return "VEH_PED"
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

            # Check approach: does distance decrease at some point?
            if not _has_approach(dists):
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


def _has_approach(dists: np.ndarray) -> bool:
    """True if distance is decreasing for at least one consecutive pair."""
    return bool(np.any(np.diff(dists) < -2.0))   # >2 px closing
