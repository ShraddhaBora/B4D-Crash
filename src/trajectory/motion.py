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
    n = len(positions)
    velocities = np.zeros((n, 2))
    speeds = np.zeros(n)
    headings = np.zeros(n)
    accelerations = np.zeros(n)

    if n < 2:
        return velocities, speeds, headings, accelerations

    # Central differences for interior points; forward/backward at edges
    for i in range(n):
        if i == 0:
            dt = timestamps[1] - timestamps[0]
            dp = positions[1] - positions[0]
        elif i == n - 1:
            dt = timestamps[-1] - timestamps[-2]
            dp = positions[-1] - positions[-2]
        else:
            dt = timestamps[i + 1] - timestamps[i - 1]
            dp = positions[i + 1] - positions[i - 1]

        if dt > 0:
            v = dp / dt
        else:
            v = np.zeros(2)

        velocities[i] = v
        speeds[i] = float(np.linalg.norm(v))
        # atan2(vy, vx) -> convert to degrees; 0=right, CW positive (image coords)
        headings[i] = float(np.degrees(np.arctan2(v[1], v[0])))

    # Acceleration: magnitude of velocity change per time
    for i in range(n):
        if i == 0:
            dt = timestamps[1] - timestamps[0]
            dv = velocities[1] - velocities[0]
        elif i == n - 1:
            dt = timestamps[-1] - timestamps[-2]
            dv = velocities[-1] - velocities[-2]
        else:
            dt = timestamps[i + 1] - timestamps[i - 1]
            dv = velocities[i + 1] - velocities[i - 1]

        accelerations[i] = float(np.linalg.norm(dv) / dt) if dt > 0 else 0.0

    return velocities, speeds, headings, accelerations


def heading_delta_deg(h1: float, h2: float) -> float:
    """Signed angular difference h2 - h1, in (-180, 180]."""
    d = h2 - h1
    return float((d + 180) % 360 - 180)
