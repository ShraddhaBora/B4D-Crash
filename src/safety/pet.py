"""
Phase 12 — Post-Encroachment Time (PET).

PET = |time_B_enters_conflict_region - time_A_exits_conflict_region|

FIX #5 — PET is a crossing-oriented metric. It is NOT universally applicable.
Previous code produced a PETResult(valid=True) for any pair of tracks that
happened to be within conflict_radius of each other, even when both tracks
were travelling in parallel (never actually crossing).

A crossing geometry is detected by checking whether the approach direction of
A and the approach direction of B have a large angular divergence (i.e. they
are coming from meaningfully different directions toward the conflict point).

For non-crossing geometries (nearly parallel or same-direction motion at the
conflict point) the result is explicitly marked valid=False with a note that
PET is not applicable, rather than silently returning a finite value.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.schema import CandidateInteraction, PETResult, Trajectory

# Minimum angular divergence (degrees) between A and B approach vectors
# required for the interaction to be treated as a crossing.
_MIN_CROSSING_ANGLE_DEG: float = 30.0


def compute_pet(
    ta: Trajectory,
    tb: Trajectory,
    candidate: CandidateInteraction,
    conflict_radius: float = 30.0,
    min_crossing_angle_deg: float = _MIN_CROSSING_ANGLE_DEG,
) -> PETResult:
    """
    Compute PET for a crossing-type candidate interaction.

    Strategy:
        1. Find the conflict point — location of minimum separation.
        2. Check that interaction geometry is actually crossing-like.
        3. Define a circle of radius `conflict_radius` around the conflict point.
        4. Find when each track enters/exits that circle.
        5. PET = time between last exit of A and first entry of B (or vice versa).

    Returns PETResult with valid=False if:
        - The overlap window is too short.
        - The geometry is not crossing-like (parallel / same-direction).
        - Neither or only one track enters the conflict region.
    """
    shared_ts = np.linspace(
        max(ta.timestamps[0], tb.timestamps[0]),
        min(ta.timestamps[-1], tb.timestamps[-1]),
        200,
    )
    if len(shared_ts) < 2:
        return PETResult(value=math.inf, valid=False, conflict_point=None,
                         confidence=0.0, applicable=False,
                         inapplicable_reason="insufficient_overlap")

    pos_a_t = _interp_traj(ta, shared_ts)
    pos_b_t = _interp_traj(tb, shared_ts)

    dists = np.linalg.norm(pos_a_t - pos_b_t, axis=1)
    min_idx = int(np.argmin(dists))
    conflict_pt = tuple(((pos_a_t[min_idx] + pos_b_t[min_idx]) / 2.0).tolist())

    # --- Crossing geometry check ---
    if not _is_crossing_geometry(pos_a_t, pos_b_t, min_idx, min_crossing_angle_deg):
        return PETResult(
            value=math.inf, valid=False, conflict_point=conflict_pt,
            confidence=0.0, applicable=False,
            inapplicable_reason="not_crossing_geometry",
        )

    cp = np.array(conflict_pt)

    # Times each track is inside the conflict circle
    times_a_in = shared_ts[np.linalg.norm(pos_a_t - cp, axis=1) <= conflict_radius]
    times_b_in = shared_ts[np.linalg.norm(pos_b_t - cp, axis=1) <= conflict_radius]

    if len(times_a_in) == 0 or len(times_b_in) == 0:
        return PETResult(value=math.inf, valid=False, conflict_point=conflict_pt,
                         confidence=0.0, applicable=True,
                         inapplicable_reason="no_conflict_region_entry")

    exit_a = float(times_a_in[-1])
    enter_b = float(times_b_in[0])
    exit_b = float(times_b_in[-1])
    enter_a = float(times_a_in[0])

    # PET: time gap between one road user leaving and the other entering
    pet1 = enter_b - exit_a   # A exits then B enters
    pet2 = enter_a - exit_b   # B exits then A enters

    if pet1 > 0:
        pet = pet1
    elif pet2 > 0:
        pet = pet2
    else:
        # Overlap: both in conflict region simultaneously — PET = 0 (worst case)
        pet = 0.0

    # Confidence: higher when both tracks clearly pass through the region
    conf = min(1.0, (len(times_a_in) + len(times_b_in)) / (0.1 * len(shared_ts) + 1))

    return PETResult(
        value=float(pet),
        valid=True,
        conflict_point=conflict_pt,
        confidence=round(conf, 3),
        applicable=True,
        inapplicable_reason=None,
    )


def _is_crossing_geometry(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    conflict_idx: int,
    min_angle_deg: float,
) -> bool:
    """
    True if the approach directions of A and B differ by at least min_angle_deg.

    We look at a window of positions before the conflict point to estimate
    the approach direction of each track.  If both are approaching from
    nearly the same direction (e.g. following / parallel), PET is not
    applicable.
    """
    half = max(1, conflict_idx // 2)
    idx_before = max(0, conflict_idx - half)

    def approach_vec(positions: np.ndarray) -> Optional[np.ndarray]:
        p_before = positions[idx_before]
        p_at = positions[conflict_idx]
        v = p_at - p_before
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return None
        return v / norm

    va = approach_vec(pos_a)
    vb = approach_vec(pos_b)

    if va is None or vb is None:
        return False

    cos_angle = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_angle))
    return angle_deg >= min_angle_deg


def _interp_traj(t: Trajectory, query_ts: np.ndarray) -> np.ndarray:
    x = np.interp(query_ts, t.timestamps, t.positions[:, 0])
    y = np.interp(query_ts, t.timestamps, t.positions[:, 1])
    return np.stack([x, y], axis=1)
