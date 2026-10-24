"""
Phase 6 — Trajectory Smoothing.

Reduces bounding-box position jitter so velocity/acceleration estimates are usable.
Uses Savitzky-Golay if the track is long enough; falls back to moving average.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def smooth_positions(positions: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth a (N, 2) position array.

    Strategy:
        - N >= window and window >= 3: Savitzky-Golay (polyorder=2)
        - otherwise: uniform moving average
        - N == 1: return as-is

    Window is forced odd (required by savgol_filter).
    """
    n = len(positions)
    if n <= 1:
        return positions.copy()

    # Enforce odd window, minimum 3
    w = max(3, window if window % 2 == 1 else window + 1)

    if n >= w:
        smoothed = np.stack([
            savgol_filter(positions[:, 0], window_length=w, polyorder=2),
            savgol_filter(positions[:, 1], window_length=w, polyorder=2),
        ], axis=1)
    else:
        # Moving average with available length
        half = max(1, n // 4)
        kernel = np.ones(half * 2 + 1) / (half * 2 + 1)
        smoothed = np.stack([
            np.convolve(positions[:, 0], kernel, mode="same"),
            np.convolve(positions[:, 1], kernel, mode="same"),
        ], axis=1)

    return smoothed
