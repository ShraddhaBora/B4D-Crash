"""
Phase 8 — Camera Calibration (optional homography).

Works in disabled mode by default — pipeline runs in image coordinates.
Enable by providing image_points and world_points in config.yaml.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.logger import get_logger

log = get_logger(__name__)


class Calibration:
    """
    Wraps an optional homography H that maps image (px) → road-plane (m or arbitrary).

    When disabled, transform() is a no-op (returns image coordinates unchanged).
    """

    def __init__(self, enabled: bool = False,
                 image_points: Optional[List] = None,
                 world_points: Optional[List] = None):
        self.enabled = enabled
        self.H: Optional[np.ndarray] = None

        if enabled:
            if not image_points or not world_points:
                log.warning("Calibration enabled but no points provided — disabling.")
                self.enabled = False
            else:
                img_pts = np.array(image_points, dtype=np.float32)
                wld_pts = np.array(world_points, dtype=np.float32)
                self.H, status = self._compute_homography(img_pts, wld_pts)
                if self.H is None:
                    log.warning("Homography computation failed — calibration disabled.")
                    self.enabled = False
                else:
                    log.info("Calibration homography computed (%d point pairs).", len(img_pts))
        else:
            log.info("Calibration disabled — pipeline operates in image coordinates.")

    @staticmethod
    def _compute_homography(img_pts: np.ndarray, wld_pts: np.ndarray):
        try:
            import cv2
            H, status = cv2.findHomography(img_pts, wld_pts, cv2.RANSAC)
            return H, status
        except Exception as e:
            return None, None

    def transform(self, x: float, y: float) -> Tuple[float, float]:
        """Map image (x,y) to world coordinates.  Returns (x,y) unchanged if disabled."""
        if not self.enabled or self.H is None:
            return x, y
        pt = np.array([[[x, y]]], dtype=np.float32)
        import cv2
        dst = cv2.perspectiveTransform(pt, self.H)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    def transform_array(self, positions: np.ndarray) -> np.ndarray:
        """Map (N,2) array.  Returns unchanged if disabled."""
        if not self.enabled or self.H is None:
            return positions
        import cv2
        pts = positions[:, np.newaxis, :].astype(np.float32)
        dst = cv2.perspectiveTransform(pts, self.H)
        return dst[:, 0, :]


def from_config(cfg: dict) -> Calibration:
    cal_cfg = cfg.get("calibration", {})
    return Calibration(
        enabled=cal_cfg.get("enabled", False),
        image_points=cal_cfg.get("image_points", []),
        world_points=cal_cfg.get("world_points", []),
    )
