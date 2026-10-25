"""Phase 2 — YOLO Object Detection + ByteTrack Tracking (unified path).

Architecture
────────────
Detection and tracking are unified in a single model.track() call, which
uses Ultralytics' officially supported high-level API.  This avoids the
internal BYTETracker.update() API (version-sensitive, broke in ≥ 8.2).

Two methods are provided:

  detect(frame, frame_id, ts)
      Pure detection — returns Detection objects.
      Used by dry_run.py and unit tests.

  track(frame, frame_id, ts)
      Detection + ByteTrack in one call via model.track(persist=True).
      Returns (List[Detection], List[TrackObservation]).
      Used by the real pipeline (pipeline.py).

Validation guarantees
─────────────────────
  [DETECTOR INIT] — model loaded, device confirmed
  [DETECTOR CUDA] — explicit log of device used (warning if CPU when CUDA expected)
  [BYTETRACK ACTIVE] — first successful track() call with track IDs returned
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.logger import get_logger
from src.schema import Detection, TrackObservation

log = get_logger(__name__)

# COCO class names used by YOLOv8 pre-trained weights
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant",
    # Ultralytics provides the full list at model.names
}


class YOLODetector:
    """
    Loads a YOLOv8 pretrained model once and runs inference per frame.

    Args:
        model_path:  Path to .pt file or Ultralytics model name ("yolov8n.pt").
        confidence:  Minimum detection confidence.
        iou:         IoU threshold for NMS.
        class_ids:   List of COCO class IDs to keep.  None = keep all.
        device:      "cpu" | "cuda" | "cuda:0" | "mps"  (auto-detected if None).
        tracker_cfg: Tracker YAML name passed to model.track() (default: bytetrack.yaml).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.30,
        iou: float = 0.50,
        class_ids: List[int] | None = None,
        device: str | None = None,
        tracker_cfg: str = "bytetrack.yaml",
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install 'ultralytics>=8.2' lap"
            ) from exc

        self.confidence = confidence
        self.iou = iou
        self.class_ids = class_ids
        self.tracker_cfg = tracker_cfg
        self.device = device or self._auto_device()
        self._track_call_count = 0

        log.info("[DETECTOR INIT] Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)
        log.info(
            "[DETECTOR INIT] Ready | device=%s | conf=%.2f | iou=%.2f | classes=%s | tracker=%s",
            self.device, confidence, iou, class_ids, tracker_cfg,
        )

    # ── Device selection ──────────────────────────────────────────────────────

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                n = torch.cuda.get_device_name(0)
                log.info("[DETECTOR CUDA] CUDA available: %s", n)
                return "cuda"
            else:
                log.warning(
                    "[DETECTOR CUDA] torch.cuda.is_available() = False. "
                    "Falling back to CPU.  Ensure GPU driver and CUDA are installed "
                    "on the compute node before running the full pipeline."
                )
        except ImportError:
            log.warning("[DETECTOR CUDA] PyTorch not found; defaulting to CPU.")
        return "cpu"

    # ── Pure detection (no tracking) ─────────────────────────────────────────

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        """
        Run inference on one frame and return standardised Detection list.
        Does NOT update tracker state.  Use track() for the real pipeline.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )
        return self._results_to_detections(results, frame_id, timestamp)

    # ── Unified detection + tracking ─────────────────────────────────────────

    def track(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> Tuple[List[Detection], List[TrackObservation]]:
        """
        Run YOLO detection + ByteTrack in a single model.track() call.

        This uses the officially supported Ultralytics high-level API:
            model.track(source=frame, persist=True, tracker="bytetrack.yaml")

        tracker state is maintained internally by Ultralytics across calls
        because persist=True keeps the tracker alive between frames.

        Returns:
            detections    — Detection objects (same as detect(); for CSV logging)
            observations  — TrackObservation objects with stable track_ids from ByteTrack

        Raises:
            RuntimeError  — if model.track() raises on the first frame
        """
        results = self.model.track(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.class_ids,
            device=self.device,
            persist=True,           # keep tracker state across frames — REQUIRED
            tracker=self.tracker_cfg,
            verbose=False,
        )

        detections = self._results_to_detections(results, frame_id, timestamp)
        observations = self._results_to_observations(results, frame_id, timestamp)

        self._track_call_count += 1
        if self._track_call_count == 1:
            n_ids = len({o.track_id for o in observations})
            log.info(
                "[BYTETRACK ACTIVE] First track() call succeeded. "
                "Detections=%d  Observations=%d  Unique track IDs=%d",
                len(detections), len(observations), n_ids,
            )

        return detections, observations

    # ── Result converters ─────────────────────────────────────────────────────

    def _results_to_detections(
        self,
        results,
        frame_id: int,
        timestamp: float,
    ) -> List[Detection]:
        """Convert Ultralytics Results to Detection objects."""
        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
        return detections

    def _results_to_observations(
        self,
        results,
        frame_id: int,
        timestamp: float,
    ) -> List[TrackObservation]:
        """
        Convert Ultralytics Results (from model.track) to TrackObservation objects.

        model.track() populates boxes.id with ByteTrack-assigned integer track IDs.
        If boxes.id is None (no tracked objects this frame), returns [].

        Mapping:
            boxes.id[i]   -> track_id   (ByteTrack integer ID)
            boxes.cls[i]  -> class_id
            boxes.conf[i] -> confidence
            boxes.xyxy[i] -> x1,y1,x2,y2
        """
        observations: List[TrackObservation] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # boxes.id is None when tracker has no active tracks this frame
            track_ids = boxes.id
            if track_ids is None:
                # No tracked boxes — valid empty frame; do not crash
                log.debug(
                    "frame_id=%d: model.track() returned boxes but boxes.id is None "
                    "(tracker has no confirmed tracks yet).",
                    frame_id,
                )
                continue

            ids_np = track_ids.int().cpu().numpy()
            cls_np = boxes.cls.int().cpu().numpy()
            conf_np = boxes.conf.float().cpu().numpy()
            xyxy_np = boxes.xyxy.float().cpu().numpy()

            for i in range(len(ids_np)):
                cls_id = int(cls_np[i])
                cls_name = self.model.names.get(cls_id, str(cls_id))
                observations.append(TrackObservation(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    track_id=int(ids_np[i]),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=float(conf_np[i]),
                    x1=float(xyxy_np[i, 0]),
                    y1=float(xyxy_np[i, 1]),
                    x2=float(xyxy_np[i, 2]),
                    y2=float(xyxy_np[i, 3]),
                ))

        return observations
