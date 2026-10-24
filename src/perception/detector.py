"""Phase 2 — YOLO Object Detection.

Wraps Ultralytics YOLO and converts results to our standardised Detection schema.
The model is loaded once and reused across frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from src.logger import get_logger
from src.schema import Detection

log = get_logger(__name__)

# COCO class names used by YOLOv8 pre-trained weights
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant",
    # ... trimmed for brevity; Ultralytics provides the full list at model.names
}


class YOLODetector:
    """
    Loads a YOLOv8 pretrained model once and runs inference per frame.

    Args:
        model_path: Path to .pt file or Ultralytics model name (e.g. "yolov8n.pt").
        confidence:  Minimum detection confidence.
        iou:         IoU threshold for NMS.
        class_ids:   List of COCO class IDs to keep.  None = keep all.
        device:      "cpu" | "cuda" | "cuda:0" | "mps"  (auto-detected if None).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.30,
        iou: float = 0.50,
        class_ids: List[int] | None = None,
        device: str | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics") from exc

        log.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.class_ids = class_ids
        self.device = device or self._auto_device()
        log.info("Detector ready on device: %s  | conf=%.2f | iou=%.2f | classes=%s",
                 self.device, confidence, iou, class_ids)

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        """Run inference on one frame and return standardised Detection list."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
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
