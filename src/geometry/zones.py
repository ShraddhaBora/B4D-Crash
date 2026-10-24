"""
Phase 9 — Conflict Zones.

Defines named spatial zones as simple polygons.
Zones can be loaded from config or auto-generated as a uniform grid.
Events are assigned to whichever zone contains/is closest to the conflict point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class Zone:
    id: str
    polygon: np.ndarray       # (K, 2) vertices in image coordinates
    description: str = ""
    conflict_types: List[str] = field(default_factory=list)

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon test (ray casting)."""
        poly = self.polygon
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def centroid(self) -> Tuple[float, float]:
        cx = float(np.mean(self.polygon[:, 0]))
        cy = float(np.mean(self.polygon[:, 1]))
        return cx, cy

    def distance_to(self, x: float, y: float) -> float:
        cx, cy = self.centroid()
        return float(np.hypot(x - cx, y - cy))


class ZoneManager:
    """Manages a collection of zones and provides zone-lookup for events."""

    def __init__(self, zones: List[Zone]):
        self.zones = zones
        log.info("ZoneManager: %d zones loaded.", len(zones))

    def find_zone(self, x: float, y: float) -> Optional[Zone]:
        """Return first zone that contains (x, y), else closest zone."""
        for z in self.zones:
            if z.contains(x, y):
                return z
        if self.zones:
            return min(self.zones, key=lambda z: z.distance_to(x, y))
        return None

    def zone_id(self, x: float, y: float) -> str:
        z = self.find_zone(x, y)
        return z.id if z else "UNKNOWN"


def make_grid_zones(frame_width: int, frame_height: int, grid_size: int = 5) -> ZoneManager:
    """
    Auto-generate a uniform grid of rectangular zones covering the frame.
    Zone IDs are 'R{row}C{col}' (0-indexed).
    """
    cw = frame_width / grid_size
    ch = frame_height / grid_size
    zones: List[Zone] = []
    for r in range(grid_size):
        for c in range(grid_size):
            x0, y0 = c * cw, r * ch
            x1, y1 = x0 + cw, y0 + ch
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            zones.append(Zone(
                id=f"R{r}C{c}",
                polygon=poly,
                description=f"Grid cell row={r} col={c}",
                conflict_types=["VEH_PED", "VEH_VEH"],
            ))
    log.info("Auto-generated %d grid zones (%dx%d) for frame %dx%d.",
             len(zones), grid_size, grid_size, frame_width, frame_height)
    return ZoneManager(zones)
