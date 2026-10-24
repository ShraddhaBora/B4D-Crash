"""
Phase 1 — Video Ingestion.

VideoReader opens a video file and yields (frame, frame_index, timestamp, fps)
with optional frame skipping and resizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np

from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


def get_video_metadata(path: str) -> VideoMeta:
    """Return metadata for a video file without decoding frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur = fc / fps if fps > 0 else 0.0
    meta = VideoMeta(path=path, fps=fps, frame_count=fc, width=w, height=h, duration_s=dur)
    log.info(
        "Video metadata | %s | %.2f fps | %d frames | %dx%d | %.1f s",
        Path(path).name, fps, fc, w, h, dur,
    )
    return meta


class VideoReader:
    """
    Iterate over video frames with optional skipping and resizing.

    Usage:
        reader = VideoReader("video.mp4", frame_skip=1, resize_width=1280)
        for frame, frame_idx, ts, fps in reader:
            ...
    """

    def __init__(self, path: str, frame_skip: int = 1, resize_width: int = 0):
        if not Path(path).exists():
            raise FileNotFoundError(f"Video not found: {path}")
        self.path = path
        self.frame_skip = max(1, frame_skip)
        self.resize_width = resize_width
        self.meta = get_video_metadata(path)

    def __iter__(self) -> Generator[Tuple[np.ndarray, int, float, float], None, None]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to re-open video: {self.path}")

        fps = self.meta.fps
        frame_idx = 0
        emitted = 0
        prev_ts = -1.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_skip == 0:
                    ts = frame_idx / fps if fps > 0 else float(frame_idx)

                    # Monotonicity guard
                    if ts <= prev_ts and emitted > 0:
                        log.warning("Non-monotonic timestamp at frame %d (%.4f <= %.4f)", frame_idx, ts, prev_ts)
                    prev_ts = ts

                    if self.resize_width and self.resize_width != frame.shape[1]:
                        scale = self.resize_width / frame.shape[1]
                        new_h = int(frame.shape[0] * scale)
                        frame = cv2.resize(frame, (self.resize_width, new_h), interpolation=cv2.INTER_LINEAR)

                    yield frame, frame_idx, ts, fps
                    emitted += 1

                frame_idx += 1
        finally:
            cap.release()

        log.info("VideoReader finished | frames emitted: %d", emitted)
