"""
Phase 3 — Multi-Object Tracking via ByteTrack (through Ultralytics).

Accepts standardised Detection objects and produces TrackObservation objects
with persistent track IDs.  The tracker state is maintained between frames.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.logger import get_logger
from src.schema import Detection, TrackObservation

log = get_logger(__name__)


class ByteTracker:
    """
    Thin wrapper around Ultralytics ByteTrack that converts our Detection
    schema ↔ tracker input/output.

    We feed detections frame-by-frame, and get back TrackObservation objects
    with stable integer track_ids.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.60,
        track_buffer: int = 30,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self._tracker = None
        self._init_tracker()

    def _init_tracker(self) -> None:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            # Ultralytics requires an argparse-style object
            args = _TrackerArgs(
                track_high_thresh=self.track_high_thresh,
                track_low_thresh=self.track_low_thresh,
                new_track_thresh=self.new_track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=0.8,
            )
            self._tracker = BYTETracker(args, frame_rate=30)
            self._use_ultralytics_bytetrack = True
            log.info("ByteTracker initialised (Ultralytics BYTETracker)")
        except (ImportError, Exception) as e:
            log.warning("Ultralytics BYTETracker unavailable (%s); falling back to simple IoU tracker.", e)
            self._tracker = _SimpleIoUTracker(iou_threshold=0.3, max_age=self.track_buffer)
            self._use_ultralytics_bytetrack = False

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> List[TrackObservation]:
        """
        Feed detections for one frame, get back track observations.
        frame is passed for re-identification feature extraction (ByteTrack uses appearance).
        """
        if not detections:
            if self._use_ultralytics_bytetrack:
                self._tracker.update(np.empty((0, 6)), frame)
            else:
                self._tracker.update([])
            return []

        if self._use_ultralytics_bytetrack:
            return self._update_bytetrack(detections, frame)
        else:
            return self._update_simple(detections)

    def _update_bytetrack(self, detections: List[Detection], frame: np.ndarray) -> List[TrackObservation]:
        # Format: [x1, y1, x2, y2, conf, cls_id]
        dets_np = np.array([
            [d.x1, d.y1, d.x2, d.y2, d.confidence, d.class_id]
            for d in detections
        ], dtype=np.float32)

        tracks = self._tracker.update(dets_np, frame)
        observations: List[TrackObservation] = []

        for t in tracks:
            # Ultralytics track format: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
            x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            tid = int(t[4])
            conf = float(t[5]) if len(t) > 5 else 0.5
            cls_id = int(t[6]) if len(t) > 6 else 0

            # Find best matching detection for metadata
            best_det = _match_detection(detections, x1, y1, x2, y2)

            observations.append(TrackObservation(
                frame_id=detections[0].frame_id,
                timestamp=detections[0].timestamp,
                track_id=tid,
                class_id=cls_id,
                class_name=best_det.class_name if best_det else str(cls_id),
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
        return observations

    def _update_simple(self, detections: List[Detection]) -> List[TrackObservation]:
        tracks = self._tracker.update(detections)
        observations: List[TrackObservation] = []
        for tid, det in tracks:
            observations.append(TrackObservation(
                frame_id=det.frame_id,
                timestamp=det.timestamp,
                track_id=tid,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
            ))
        return observations


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _match_detection(detections: List[Detection], x1, y1, x2, y2) -> Detection | None:
    """Return detection with highest IoU to the given box."""
    best, best_iou = None, 0.0
    for d in detections:
        iou = _iou(d.x1, d.y1, d.x2, d.y2, x1, y1, x2, y2)
        if iou > best_iou:
            best_iou, best = iou, d
    return best


def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class _TrackerArgs:
    """Minimal namespace to satisfy Ultralytics BYTETracker constructor."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _SimpleIoUTracker:
    """
    Dead-simple IoU-based tracker as fallback when BYTETracker is unavailable.
    Not production quality but sufficient for validation.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, dict] = {}  # tid -> {det, age}

    def update(self, detections: List[Detection]) -> List[tuple]:
        """Returns list of (track_id, Detection)."""
        # Age out stale tracks
        to_delete = [tid for tid, t in self._tracks.items() if t["age"] > self.max_age]
        for tid in to_delete:
            del self._tracks[tid]

        # Increment ages
        for t in self._tracks.values():
            t["age"] += 1

        assigned: dict[int, Detection] = {}  # tid -> det
        used_dets: set[int] = set()

        # Match detections to existing tracks by IoU
        for tid, t in self._tracks.items():
            best_iou, best_idx = 0.0, -1
            last = t["det"]
            for i, d in enumerate(detections):
                if i in used_dets:
                    continue
                iou = _iou(last.x1, last.y1, last.x2, last.y2, d.x1, d.y1, d.x2, d.y2)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                assigned[tid] = detections[best_idx]
                used_dets.add(best_idx)
                self._tracks[tid] = {"det": detections[best_idx], "age": 0}

        # New tracks for unmatched detections
        for i, d in enumerate(detections):
            if i not in used_dets:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"det": d, "age": 0}
                assigned[tid] = d

        return list(assigned.items())
