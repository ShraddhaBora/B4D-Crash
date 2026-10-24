"""
Phase 13 — Evasive Action Detection.

Looks for abrupt speed reductions, sudden heading changes, or stops
that occur in temporal proximity to a candidate interaction.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from src.schema import CandidateInteraction, EvasiveAction, Trajectory


def detect_evasive_actions(
    ta: Trajectory,
    tb: Trajectory,
    candidate: CandidateInteraction,
    speed_change_threshold: float = 0.35,
    heading_change_threshold: float = 25.0,
    window_s: float = 2.0,
) -> List[EvasiveAction]:
    """
    Detect evasive actions in both tracks within the candidate interaction window
    (extended by window_s on each side for context).

    Returns list of EvasiveAction objects.
    """
    t_lo = candidate.start_time - window_s
    t_hi = candidate.end_time   + window_s
    actions: List[EvasiveAction] = []

    for traj in (ta, tb):
        mask = (traj.timestamps >= t_lo) & (traj.timestamps <= t_hi)
        if mask.sum() < 3:
            continue

        ts = traj.timestamps[mask]
        sp = traj.speeds[mask]
        hd = traj.headings[mask]

        # ── Speed reduction ──────────────────────────────────────────────────
        for i in range(1, len(sp) - 1):
            speed_before = float(sp[i - 1])
            speed_now    = float(sp[i])
            if speed_before < 1.0:  # stationary — skip
                continue
            ratio = (speed_before - speed_now) / speed_before
            if ratio >= speed_change_threshold:
                actions.append(EvasiveAction(
                    track_id=traj.track_id,
                    timestamp=float(ts[i]),
                    action_type="BRAKING",
                    speed_change_ratio=-ratio,
                    heading_change_deg=0.0,
                ))

        # ── Heading change ───────────────────────────────────────────────────
        for i in range(1, len(hd)):
            delta = _heading_delta(float(hd[i - 1]), float(hd[i]))
            if abs(delta) >= heading_change_threshold:
                actions.append(EvasiveAction(
                    track_id=traj.track_id,
                    timestamp=float(ts[i]),
                    action_type="SWERVE",
                    speed_change_ratio=0.0,
                    heading_change_deg=delta,
                ))

        # ── Near-stop ───────────────────────────────────────────────────────
        for i in range(1, len(sp)):
            if float(sp[i - 1]) > 5.0 and float(sp[i]) < 2.0:
                actions.append(EvasiveAction(
                    track_id=traj.track_id,
                    timestamp=float(ts[i]),
                    action_type="STOP",
                    speed_change_ratio=-1.0,
                    heading_change_deg=0.0,
                ))

    # Deduplicate within 0.3 s window
    return _dedup_actions(actions, window=0.3)


def _heading_delta(h1: float, h2: float) -> float:
    d = h2 - h1
    return float((d + 180) % 360 - 180)


def _dedup_actions(actions: List[EvasiveAction], window: float = 0.3) -> List[EvasiveAction]:
    """Remove duplicate actions of the same type for the same track within `window` seconds."""
    seen: set = set()
    result = []
    for a in sorted(actions, key=lambda x: x.timestamp):
        key = (a.track_id, a.action_type, round(a.timestamp / window))
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result
