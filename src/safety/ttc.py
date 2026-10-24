"""
Phase 11 — Time-To-Collision (TTC).

Two models are implemented:
  1. linear_closing  — for roughly same-path motion (gap / closing speed).
  2. closest_approach — general relative-motion model; time to minimum separation.

Units: image-space (px, px/s).  NOT meters unless calibration is active.
"""

from __future__ import annotations

import math
import numpy as np

from src.schema import CandidateInteraction, Trajectory, TTCResult


def compute_ttc(
    ta: Trajectory,
    tb: Trajectory,
    candidate: CandidateInteraction,
    t_query: float,
) -> TTCResult:
    """
    Compute TTC for a candidate interaction at time t_query.

    Tries closest-approach first; falls back to linear_closing.
    Returns TTCResult with value=inf if TTC is undefined/not applicable.
    """
    # Interpolate positions and velocities at t_query
    pos_a = _interp_vec(ta.timestamps, ta.positions, t_query)
    pos_b = _interp_vec(tb.timestamps, tb.positions, t_query)
    vel_a = _interp_vec(ta.timestamps, ta.velocities, t_query)
    vel_b = _interp_vec(tb.timestamps, tb.velocities, t_query)

    if pos_a is None or pos_b is None or vel_a is None or vel_b is None:
        return TTCResult(value=math.inf, valid=False, model="none", confidence=0.0)

    rel_pos = pos_b - pos_a   # relative position (B relative to A)
    rel_vel = vel_b - vel_a   # relative velocity

    result = _ttc_closest_approach(rel_pos, rel_vel)
    if result.valid:
        return result

    # Fallback: linear closing model along the line joining centres
    return _ttc_linear_closing(pos_a, pos_b, vel_a, vel_b)


def _ttc_closest_approach(rel_pos: np.ndarray, rel_vel: np.ndarray) -> TTCResult:
    """
    t* = - (rel_pos · rel_vel) / |rel_vel|²

    This is the time at which the two objects are closest in a linear-motion model.
    TTC is defined only when t* > 0 (they are still approaching).
    """
    rv_sq = float(np.dot(rel_vel, rel_vel))
    if rv_sq < 1e-6:
        return TTCResult(value=math.inf, valid=False, model="closest_approach", confidence=0.0)

    t_star = -float(np.dot(rel_pos, rel_vel)) / rv_sq

    if t_star <= 0:
        return TTCResult(value=math.inf, valid=False, model="closest_approach", confidence=0.0)

    # Confidence: higher when relative speed is large and distance is small
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
