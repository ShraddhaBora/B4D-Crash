"""
Phase 4 + 5 — Track Quality Checks + Trajectory Reconstruction.

Builds Trajectory objects from raw TrackObservation lists.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.logger import get_logger
from src.schema import TrackObservation, Trajectory, TrackQuality
from src.trajectory.smoothing import smooth_positions
from src.trajectory.motion import compute_motion

log = get_logger(__name__)


def build_trajectories(
    observations: List[TrackObservation],
    min_track_length: int = 10,
    max_gap: int = 5,
    max_speed_pixels: float = 300.0,
    smoothing_window: int = 5,
) -> Dict[int, Trajectory]:
    """
    Convert a flat list of TrackObservations into per-track Trajectory objects.

    Returns:
        dict mapping track_id -> Trajectory (all quality levels included;
        check trajectory.quality before use in safety analysis).
    """
    # Group by track_id
    by_track: Dict[int, List[TrackObservation]] = defaultdict(list)
    for obs in observations:
        by_track[obs.track_id].append(obs)

    trajectories: Dict[int, Trajectory] = {}

    for tid, obs_list in by_track.items():
        # Sort chronologically
        obs_list.sort(key=lambda o: o.frame_id)

        # Determine dominant class (handles rare class flips)
        classes = [o.class_name for o in obs_list]
        class_name = max(set(classes), key=classes.count)

        timestamps = np.array([o.timestamp for o in obs_list])
        raw_positions = np.array([[o.foot_x, o.foot_y] for o in obs_list])

        # ── Quality checks ────────────────────────────────────────────────────
        quality = _assess_quality(obs_list, timestamps, raw_positions,
                                  min_track_length, max_gap, max_speed_pixels)

        if quality != TrackQuality.GOOD:
            log.debug("Track %d quality=%s  len=%d", tid, quality, len(obs_list))

        # ── Smooth + motion ───────────────────────────────────────────────────
        positions = smooth_positions(raw_positions, window=smoothing_window)
        # Defensive clamp: smoothing must not change array length, but clamp
        # to be safe so compute_motion never receives mismatched arrays.
        if len(positions) != len(timestamps):
            n_min = min(len(positions), len(timestamps))
            log.warning(
                "Track %d: positions len=%d != timestamps len=%d after smoothing — clamping to %d",
                tid, len(positions), len(timestamps), n_min,
            )
            positions  = positions[:n_min]
            timestamps = timestamps[:n_min]
        velocities, speeds, headings, accelerations = compute_motion(positions, timestamps)

        trajectories[tid] = Trajectory(
            track_id=tid,
            class_name=class_name,
            quality=quality,
            timestamps=timestamps,
            raw_positions=raw_positions,
            positions=positions,
            velocities=velocities,
            speeds=speeds,
            headings=headings,
            accelerations=accelerations,
        )

    good_count = sum(1 for t in trajectories.values() if t.quality == TrackQuality.GOOD)
    log.info("Trajectories built: %d total  |  %d GOOD", len(trajectories), good_count)
    return trajectories


def _assess_quality(
    obs_list: List[TrackObservation],
    timestamps: np.ndarray,
    positions: np.ndarray,
    min_track_length: int,
    max_gap: int,
    max_speed_pixels: float,
) -> str:
    n = len(obs_list)

    if n < min_track_length:
        return TrackQuality.SHORT

    # Gap detection: consecutive frame_id differences
    frame_ids = np.array([o.frame_id for o in obs_list])
    gaps = np.diff(frame_ids)
    if np.any(gaps > max_gap):
        return TrackQuality.FRAGMENTED

    # Implausible velocity: pixels per second
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        dp = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            speeds = np.where(dt > 0, dp / dt, 0.0)
        if np.any(speeds > max_speed_pixels):
            return TrackQuality.IMPLAUSIBLE

    return TrackQuality.GOOD
