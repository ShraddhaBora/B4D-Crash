"""
Phase 11 — Time-To-Collision (TTC).

Two models are implemented:
  1. linear_closing  — for roughly same-path motion (gap / closing speed).
  2. closest_approach — general relative-motion model; time to minimum separation.

FIX #1 — A positive time-to-closest-approach is NOT automatically valid TTC.
The predicted minimum separation must also be below a conflict radius
(objects must actually get close enough to constitute a plausible collision).
Without this check, objects that merely share a closing direction anywhere on
screen produce spuriously valid TTC values.

FIX #2 — compute_ttc_multi() evaluates TTC at multiple timestamps across the
candidate interaction window and returns the minimum valid TTC together with
the timestamp of the peak conflict, replacing the single-midpoint evaluation.

Units: image-space (px, px/s).  NOT meters unless calibration is active.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from src.schema import CandidateInteraction, Trajectory, TTCResult


# Default conflict radius in pixels.  Objects must be predicted to reach at
# least this close to produce a valid TTC from the closest-approach model.
_DEFAULT_CONFLICT_RADIUS_PX: float = 50.0


def compute_ttc(
    ta: Trajectory,
    tb: Trajectory,
    candidate: CandidateInteraction,
    t_query: float,
    conflict_radius: float = _DEFAULT_CONFLICT_RADIUS_PX,
) -> TTCResult:
    """
    Compute TTC for a candidate interaction at time t_query.

    Tries closest-approach first; falls back to linear_closing.
    Returns TTCResult with value=inf if TTC is undefined/not applicable.

    Args:
        conflict_radius: Minimum predicted separation (px) that must be
            reached for closest-approach TTC to be considered valid.
    """
    pos_a = _interp_vec(ta.timestamps, ta.positions, t_query)
    pos_b = _interp_vec(tb.timestamps, tb.positions, t_query)
    vel_a = _interp_vec(ta.timestamps, ta.velocities, t_query)
    vel_b = _interp_vec(tb.timestamps, tb.velocities, t_query)

    if pos_a is None or pos_b is None or vel_a is None or vel_b is None:
        return TTCResult(value=math.inf, valid=False, model="none", confidence=0.0)

    rel_pos = pos_b - pos_a
    rel_vel = vel_b - vel_a

    result = _ttc_closest_approach(rel_pos, rel_vel, conflict_radius)
    if result.valid or result.model == "closest_approach_miss":
        return result

    return _ttc_linear_closing(pos_a, pos_b, vel_a, vel_b)


def compute_ttc_multi(
    ta: Trajectory,
    tb: Trajectory,
    candidate: CandidateInteraction,
    n_samples: int = 10,
    conflict_radius: float = _DEFAULT_CONFLICT_RADIUS_PX,
) -> Tuple[TTCResult, float]:
    """
    Evaluate TTC at n_samples timestamps across the candidate interaction window.

    Returns:
        (min_ttc_result, peak_conflict_timestamp)
        min_ttc_result — the TTCResult with the smallest valid TTC value found;
                         if no valid TTC is found, returns the invalid result.
        peak_conflict_timestamp — timestamp at which min TTC occurred (or midpoint).
    """
    t_start = candidate.start_time
    t_end = candidate.end_time
    if t_end <= t_start:
        mid = (t_start + t_end) / 2.0
        return compute_ttc(ta, tb, candidate, mid, conflict_radius), mid

    sample_ts = np.linspace(t_start, t_end, max(n_samples, 2))
    best_result: TTCResult = TTCResult(value=math.inf, valid=False, model="none", confidence=0.0)
    best_t: float = float((t_start + t_end) / 2.0)

    for t in sample_ts:
        r = compute_ttc(ta, tb, candidate, float(t), conflict_radius)
        if r.valid and r.value < best_result.value:
            best_result = r
            best_t = float(t)

    return best_result, best_t


def _ttc_closest_approach(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    conflict_radius: float,
) -> TTCResult:
    """
    t* = -(rel_pos · rel_vel) / |rel_vel|²

    This is the time at which the two objects are closest in a linear-motion model.

    FIX: TTC is only valid when:
      1. t* > 0  (still approaching, not already separating)
      2. predicted minimum separation < conflict_radius  (they actually get close)
    """
    rv_sq = float(np.dot(rel_vel, rel_vel))
    if rv_sq < 1e-6:
        return TTCResult(value=math.inf, valid=False, model="closest_approach", confidence=0.0)

    t_star = -float(np.dot(rel_pos, rel_vel)) / rv_sq

    if t_star <= 0:
        # Objects already at or past closest approach
        return TTCResult(value=math.inf, valid=False, model="closest_approach", confidence=0.0)

    # Predicted position of minimum separation
    min_sep_pos = rel_pos + t_star * rel_vel
    min_sep_dist = float(np.linalg.norm(min_sep_pos))

    if min_sep_dist > conflict_radius:
        # Objects will pass each other at a safe distance — not a collision candidate
        return TTCResult(value=math.inf, valid=False, model="closest_approach_miss", confidence=0.0)

    # Confidence: higher when relative speed is large and current distance is small
    speed = math.sqrt(rv_sq)
    dist = float(np.linalg.norm(rel_pos))
    conf = min(1.0, speed / max(dist, 1.0))

    return TTCResult(value=t_star, valid=True, model="closest_approach", confidence=round(conf, 3))


def _ttc_linear_closing(pos_a, pos_b, vel_a, vel_b) -> TTCResult:
    """Gap / closing-speed model along the axis connecting the two objects."""
    gap = float(np.linalg.norm(pos_b - pos_a))
    direction = (pos_b - pos_a) / max(gap, 1e-6)
    closing_speed = float(np.dot(vel_a - vel_b, direction))

    if closing_speed <= 0:
        return TTCResult(value=math.inf, valid=False, model="linear_closing", confidence=0.0)

    ttc = gap / closing_speed
    conf = min(1.0, closing_speed / max(gap, 1.0) * 10)
    return TTCResult(value=ttc, valid=True, model="linear_closing", confidence=round(conf, 3))


def _interp_vec(timestamps: np.ndarray, vectors: np.ndarray, t: float):
    """Linearly interpolate a (N,2) array at time t.  Returns None if out of range."""
    if t < timestamps[0] - 0.1 or t > timestamps[-1] + 0.1:
        return None
    t_clamp = float(np.clip(t, timestamps[0], timestamps[-1]))
    x = float(np.interp(t_clamp, timestamps, vectors[:, 0]))
    y = float(np.interp(t_clamp, timestamps, vectors[:, 1]))
    return np.array([x, y])
