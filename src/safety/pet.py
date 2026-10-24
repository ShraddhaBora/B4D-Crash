"""
Phase 12 — Post-Encroachment Time (PET).

PET = |time_B_enters_conflict_region - time_A_exits_conflict_region|

Used primarily for crossing-type interactions.
The conflict region is the spatial overlap (bounding-box intersection) of the
two road users' trajectories, or their closest-approach bounding box.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from src.schema import CandidateInteraction, PETResult, Trajectory


def compute_pet(
    ta: Trajectory,
    tb: Trajectory,
    candidate: CandidateInteraction,
    conflict_radius: float = 30.0,
) -> PETResult:
    """
    Compute PET for a crossing-type candidate interaction.

    Strategy:
        1. Find the conflict point — location of minimum separation.
        2. Define a circle of radius `conflict_radius` around it.
        3. Find when each track enters/exits that circle.
        4. PET = time between last exit of A and first entry of B (or vice versa).

    Returns PETResult with value=inf if there is no crossing or tracks overlap.
    """
    # Find conflict point: time and location of minimum distance
    shared_ts = np.linspace(
        max(ta.timestamps[0], tb.timestamps[0]),
        min(ta.timestamps[-1], tb.timestamps[-1]),
        200,
    )
    if len(shared_ts) < 2:
        return PETResult(value=math.inf, valid=False, conflict_point=None, confidence=0.0)

    pos_a_t = _interp_traj(ta, shared_ts)
    pos_b_t = _interp_traj(tb, shared_ts)

    dists = np.linalg.norm(pos_a_t - pos_b_t, axis=1)
    min_idx = int(np.argmin(dists))
    conflict_pt = tuple(((pos_a_t[min_idx] + pos_b_t[min_idx]) / 2.0).tolist())

    cp = np.array(conflict_pt)

    # Times each track is inside the conflict circle
    times_a_in = shared_ts[np.linalg.norm(pos_a_t - cp, axis=1) <= conflict_radius]
    times_b_in = shared_ts[np.linalg.norm(pos_b_t - cp, axis=1) <= conflict_radius]

    if len(times_a_in) == 0 or len(times_b_in) == 0:
        # Neither or only one enters the conflict region — PET not applicable
        return PETResult(value=math.inf, valid=False, conflict_point=conflict_pt, confidence=0.0)

    exit_a = float(times_a_in[-1])   # last time A is inside
    enter_b = float(times_b_in[0])   # first time B is inside
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
    )


def _interp_traj(t: Trajectory, query_ts: np.ndarray) -> np.ndarray:
    x = np.interp(query_ts, t.timestamps, t.positions[:, 0])
    y = np.interp(query_ts, t.timestamps, t.positions[:, 1])
    return np.stack([x, y], axis=1)
