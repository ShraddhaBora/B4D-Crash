"""
Phase 7 — Motion Estimation.

Computes velocity, speed, heading, and acceleration from smoothed positions.
All values are in image-space (pixels/second, degrees) — NOT physical units
unless camera calibration has been applied upstream.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_motion(
    positions: np.ndarray,   # (N, 2)  smoothed [x, y]
    timestamps: np.ndarray,  # (N,)    seconds
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        velocities    (N, 2)  image-space px/s  [vx, vy]
        speeds        (N,)    scalar px/s
        headings      (N,)    degrees  (0=right, CW positive)
        accelerations (N,)    scalar px/s²  (magnitude of Δvelocity/Δt)
    """
    # Guard: must have matching lengths; clamp to shorter array defensively.
    n = min(len(positions), len(timestamps))
    positions  = positions[:n]
    timestamps = timestamps[:n]

    velocities    = np.zeros((n, 2))
    speeds        = np.zeros(n)
    headings      = np.zeros(n)
    accelerations = np.zeros(n)

    if n < 2:
        return velocities, speeds, headings, accelerations

    # ── Velocity via finite differences (vectorised, no per-index loop) ────────
    # Build arrays of Δposition and Δtime using forward diff at start,
    # backward diff at end, central diff everywhere else.
    dp = np.zeros((n, 2))
    dt = np.zeros(n)

    if n == 2:
        # Only two points — forward diff for [0], backward diff for [1]
        dp[0] = positions[1] - positions[0]
        dt[0] = timestamps[1] - timestamps[0]
        dp[1] = positions[1] - positions[0]
        dt[1] = timestamps[1] - timestamps[0]
    else:
        # Forward difference at index 0
        dp[0] = positions[1] - positions[0]
        dt[0] = timestamps[1] - timestamps[0]
        # Backward difference at last index
        dp[-1] = positions[-1] - positions[-2]
        dt[-1] = timestamps[-1] - timestamps[-2]
        # Central difference for interior indices 1..n-2
        dp[1:-1] = positions[2:] - positions[:-2]
        dt[1:-1] = timestamps[2:] - timestamps[:-2]

    # Avoid division by zero
    safe_dt = np.where(dt > 0, dt, np.inf)
    velocities = dp / safe_dt[:, np.newaxis]

    speeds  = np.linalg.norm(velocities, axis=1)
    headings = np.degrees(np.arctan2(velocities[:, 1], velocities[:, 0]))

    # ── Acceleration via finite differences on velocity ────────────────────────
    dv = np.zeros((n, 2))
    at = np.zeros(n)

    if n == 2:
        dv[0]  = velocities[1] - velocities[0]
        at[0]  = timestamps[1] - timestamps[0]
        dv[-1] = velocities[1] - velocities[0]
        at[-1] = timestamps[1] - timestamps[0]
    else:
        dv[0]  = velocities[1] - velocities[0]
        at[0]  = timestamps[1] - timestamps[0]
        dv[-1] = velocities[-1] - velocities[-2]
        at[-1] = timestamps[-1] - timestamps[-2]
        dv[1:-1] = velocities[2:] - velocities[:-2]
        at[1:-1] = timestamps[2:] - timestamps[:-2]

    safe_at = np.where(at > 0, at, np.inf)
    accelerations = np.linalg.norm(dv, axis=1) / safe_at

    return velocities, speeds, headings, accelerations


def heading_delta_deg(h1: float, h2: float) -> float:
    """Signed angular difference h2 - h1, in (-180, 180]."""
    d = h2 - h1
    return float((d + 180) % 360 - 180)
