"""
Phase 13 — Evasive Action Detection.

Looks for abrupt speed reductions, sudden heading changes, or stops
that occur in temporal proximity to a candidate interaction.

FIX #4 — Do NOT treat any braking or heading change anywhere in the candidate
window as evidence of a near-miss.

Evasive actions are only retained when they occur within a configurable
temporal window around the *peak conflict timestamp* (the point of minimum TTC,
or the midpoint if TTC is unavailable).  The old implementation searched the
entire interaction window extended by window_s on each side, which meant normal
deceleration or turning unrelated to the conflict could be incorrectly flagged.

The key change:
  - accept an optional `peak_conflict_time` parameter.
  - filter detected actions to only those within evasive_window_s of it.
  - the wider search range (candidate window + window_s) is kept for detection,
    but the filter ensures only contextually-relevant actions are returned.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from src.schema import CandidateInteraction, EvasiveAction, Trajectory


def detect_evasive_actions(
    ta: Trajectory,
    tb: Trajectory,
    candidate: CandidateInteraction,
    speed_change_threshold: float = 0.35,
    heading_change_threshold: float = 25.0,
    window_s: float = 2.0,
    peak_conflict_time: Optional[float] = None,
    evasive_window_s: float = 2.0,
) -> List[EvasiveAction]:
    """
    Detect evasive actions in both tracks.

    Args:
        speed_change_threshold: Fractional speed drop required to call BRAKING.
        heading_change_threshold: Degrees per step required to call SWERVE.
        window_s: How far outside the candidate window to look for actions.
        peak_conflict_time: Timestamp of the minimum TTC / peak conflict.
            If None, uses the candidate midpoint.
        evasive_window_s: Only return actions within this many seconds of
            peak_conflict_time.  Prevents flagging routine deceleration as evasion.

    Returns list of EvasiveAction objects filtered to peak-conflict proximity.
    """
    # Search range — wider than filter range so we detect actions slightly outside
    t_lo = candidate.start_time - window_s
    t_hi = candidate.end_time + window_s
    actions: List[EvasiveAction] = []

    for traj in (ta, tb):
        mask = (traj.timestamps >= t_lo) & (traj.timestamps <= t_hi)
        if mask.sum() < 3:
            continue

        ts = traj.timestamps[mask]
        sp = traj.speeds[mask]
        hd = traj.headings[mask]

        # Speed reduction (BRAKING)
        for i in range(1, len(sp) - 1):
            speed_before = float(sp[i - 1])
            speed_now    = float(sp[i])
            if speed_before < 1.0:   # stationary — skip
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

        # Heading change (SWERVE)
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

        # Near-stop (STOP)
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
    actions = _dedup_actions(actions, window=0.3)

    # FIX #4 — Filter to peak-conflict temporal proximity
    ref_t = peak_conflict_time if peak_conflict_time is not None else (
        (candidate.start_time + candidate.end_time) / 2.0
    )
    actions = [a for a in actions if abs(a.timestamp - ref_t) <= evasive_window_s]

    return actions


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
