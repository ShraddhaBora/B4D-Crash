"""
Phase 3 — Multi-Object Tracking.

Architecture change (Ultralytics ≥ 8.2 API incompatibility fix):
──────────────────────────────────────────────────────────────────
The original code called BYTETracker.update(numpy_array, frame) which
immediately fails in Ultralytics ≥ 8.2 because _split_detections()
accesses results.conf on what it expects to be an Ultralytics Results
object, not a NumPy array.

Root cause:  The internal BYTETracker API is not stable and changed
             between Ultralytics versions.  Direct use of
             ultralytics.trackers.byte_tracker.BYTETracker is
             version-sensitive and must not be relied upon.

Chosen fix:  Detection + tracking are unified in YOLODetector.track()
             which calls model.track(persist=True, tracker="bytetrack.yaml")
             — the officially supported high-level Ultralytics API.
             This module now receives TrackObservation objects that were
             already produced by YOLODetector.track() and simply passes
             them through, maintaining the same external interface for
             the rest of the pipeline.

Fallback:    _SimpleIoUTracker is kept as a clearly-labelled fallback.
             When it is active, logs will never claim ByteTrack is in use.

Validation stages (logged explicitly):
  [TRACKER INIT]   — class constructed
  [TRACKER READY]  — first frame processed without exception
  [TRACKER STABLE] — track IDs persisted across ≥ 2 consecutive frames
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.logger import get_logger
from src.schema import Detection, TrackObservation

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public tracker class
# ──────────────────────────────────────────────────────────────────────────────

class ByteTracker:
    """
    Pipeline-facing tracker wrapper.

    In the new architecture, YOLODetector.track() performs both detection
    AND ByteTrack tracking via model.track(persist=True, tracker='bytetrack.yaml').
    This class receives the resulting TrackObservation list from the detector
    and passes it through, acting as a thin validation/logging layer.

    If the detector is running in detect-only mode (no YOLO.track support),
    this class falls back to _SimpleIoUTracker and logs that fact clearly.

    Public interface is unchanged from the pipeline's perspective:
        observations = tracker.update(detections, frame)   [legacy mode only]
        observations = tracker.passthrough(observations)   [new unified mode]
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

        # Unified mode: YOLODetector.track() handles tracking; this becomes a no-op.
        # Legacy fallback mode: _SimpleIoUTracker is used when called via update().
        self._fallback = _SimpleIoUTracker(
            iou_threshold=0.3,
            max_age=track_buffer,
        )

        # Tracking state for validation logging
        self._frames_processed = 0
        self._prev_track_ids: set[int] = set()
        self._stable_confirmed = False

        log.info(
            "[TRACKER INIT] ByteTracker constructed "
            "(unified mode: tracking handled by YOLODetector.track). "
            "Fallback: _SimpleIoUTracker available if needed."
        )

    def passthrough(self, observations: List[TrackObservation]) -> List[TrackObservation]:
        """
        In unified mode (YOLODetector.track), tracking is already done.
        This method validates and logs state, then returns observations as-is.
        """
        self._log_frame_state(observations)
        return observations

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> List[TrackObservation]:
        """
        Legacy fallback interface: convert Detections via _SimpleIoUTracker.

        This is called only when the pipeline uses detect-only mode (not
        YOLODetector.track).  Logs explicitly that ByteTrack is NOT active.
        """
        if self._frames_processed == 0:
            log.warning(
                "[TRACKER FALLBACK] tracker.update() called with Detection list. "
                "This means YOLODetector.track() is not being used. "
                "ACTIVE TRACKER: SimpleIoUTracker (NOT ByteTrack). "
                "Track quality will be degraded."
            )

        observations = self._update_simple(detections)
        self._log_frame_state(observations)
        return observations

    def _update_simple(self, detections: List[Detection]) -> List[TrackObservation]:
        tracks = self._fallback.update(detections)
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

    def _log_frame_state(self, observations: List[TrackObservation]) -> None:
        self._frames_processed += 1
        current_ids = {o.track_id for o in observations}

        if self._frames_processed == 1:
            log.info(
                "[TRACKER READY] First frame processed. "
                "Track IDs this frame: %s  (count=%d)",
                sorted(current_ids), len(current_ids),
            )

        elif self._frames_processed == 2 and not self._stable_confirmed:
            persisted = current_ids & self._prev_track_ids
            if persisted:
                self._stable_confirmed = True
                log.info(
                    "[TRACKER STABLE] Track IDs persisted across frames 1→2: %s",
                    sorted(persisted),
                )
            else:
                log.warning(
                    "[TRACKER STABLE?] No track IDs persisted from frame 1 to frame 2. "
                    "Possible tracker re-initialisation or all tracks are new. "
                    "IDs frame1=%s  IDs frame2=%s",
                    sorted(self._prev_track_ids), sorted(current_ids),
                )

        self._prev_track_ids = current_ids


# ──────────────────────────────────────────────────────────────────────────────
# Fallback tracker
# ──────────────────────────────────────────────────────────────────────────────

class _SimpleIoUTracker:
    """
    Dead-simple IoU-based tracker.

    Used ONLY when YOLODetector.track() is unavailable.
    Never claims to be ByteTrack.  Not suitable for production-quality
    identity persistence; use only for debugging or CI dry-runs.
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

        for t in self._tracks.values():
            t["age"] += 1

        assigned: dict[int, Detection] = {}
        used_dets: set[int] = set()

        for tid, t in self._tracks.items():
            best_iou, best_idx = 0.0, -1
            last = t["det"]
            for i, d in enumerate(detections):
                if i in used_dets:
                    continue
                iou = _iou(last.x1, last.y1, last.x2, last.y2,
                           d.x1, d.y1, d.x2, d.y2)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                assigned[tid] = detections[best_idx]
                used_dets.add(best_idx)
                self._tracks[tid] = {"det": detections[best_idx], "age": 0}

        for i, d in enumerate(detections):
            if i not in used_dets:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"det": d, "age": 0}
                assigned[tid] = d

        return list(assigned.items())


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _match_detection(detections: List[Detection], x1, y1, x2, y2) -> Detection | None:
    """Return detection with highest IoU to the given box."""
    best, best_iou = None, 0.0
    for d in detections:
        iou = _iou(d.x1, d.y1, d.x2, d.y2, x1, y1, x2, y2)
        if iou > best_iou:
            best_iou, best = iou, d
    return best
