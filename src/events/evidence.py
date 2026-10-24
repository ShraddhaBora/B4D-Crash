"""
Phase 17 — Evidence Generation.

For each conflict event, extracts a short annotated clip from the source video
and generates a fully annotated full-video output.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.logger import get_logger
from src.schema import ConflictEvent, Trajectory, TrackObservation

log = get_logger(__name__)

# Colour palette — track_id % len(PALETTE)
PALETTE = [
    (255, 60, 60),   (60, 180, 255),  (60, 255, 100),  (255, 200, 0),
    (200, 60, 255),  (0, 220, 220),   (255, 130, 0),   (130, 255, 0),
]


def generate_evidence_clips(
    video_path: str,
    events: List[ConflictEvent],
    trajectories: Dict[int, Trajectory],
    observations: List[TrackObservation],
    output_dir: str = "outputs/clips",
    pre_s: float = 2.0,
    post_s: float = 2.0,
) -> List[str]:
    """
    For each event, extract a short annotated clip.

    Returns list of output clip paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Index observations by frame
    obs_by_frame: Dict[int, List[TrackObservation]] = {}
    for o in observations:
        obs_by_frame.setdefault(o.frame_id, []).append(o)

    paths: List[str] = []
    for ev in events:
        clip_path = str(Path(output_dir) / f"event_{ev.event_id}.mp4")
        _extract_clip(
            video_path, clip_path,
            start_s=max(0.0, ev.start_time - pre_s),
            end_s=ev.end_time + post_s,
            fps=fps, w=w, h=h,
            event=ev,
            trajectories=trajectories,
            obs_by_frame=obs_by_frame,
        )
        paths.append(clip_path)
        log.info("Evidence clip saved: %s", clip_path)

    return paths


def generate_annotated_video(
    video_path: str,
    observations: List[TrackObservation],
    events: List[ConflictEvent],
    trajectories: Dict[int, Trajectory],
    output_path: str = "outputs/annotated_video.mp4",
) -> str:
    """Render full annotated video with tracks, trails, and event overlays."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    obs_by_frame: Dict[int, List[TrackObservation]] = {}
    for o in observations:
        obs_by_frame.setdefault(o.frame_id, []).append(o)

    # Build frame → active events lookup
    event_by_frame: Dict[int, List[ConflictEvent]] = {}
    for ev in events:
        for fi in range(ev.start_frame, ev.end_frame + 1):
            event_by_frame.setdefault(fi, []).append(ev)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _annotate_frame(frame, frame_idx, obs_by_frame,
                                 event_by_frame, trajectories, fps)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    log.info("Annotated video saved: %s", output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_clip(video_path, clip_path, start_s, end_s, fps, w, h,
                  event, trajectories, obs_by_frame):
    cap = cv2.VideoCapture(video_path)
    start_frame = int(start_s * fps)
    end_frame   = int(end_s   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))

    # Dummy event_by_frame for clip window
    event_by_frame = {}
    for fi in range(event.start_frame, event.end_frame + 1):
        event_by_frame[fi] = [event]

    for fi in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = _annotate_frame(frame, fi, obs_by_frame, event_by_frame, trajectories, fps)
        out.write(frame)

    cap.release()
    out.release()


def _annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    obs_by_frame: Dict[int, List[TrackObservation]],
    event_by_frame: Dict[int, List[ConflictEvent]],
    trajectories: Dict[int, Trajectory],
    fps: float,
) -> np.ndarray:
    frame = frame.copy()
    obs_list = obs_by_frame.get(frame_idx, [])

    # Draw trajectory trails (last 30 frames)
    trail_len = int(fps * 1.0)   # 1 second of trail
    for obs in obs_list:
        tid = obs.track_id
        color = PALETTE[tid % len(PALETTE)]
        traj = trajectories.get(tid)
        if traj is not None:
            cur_ts = obs.timestamp
            trail_mask = (traj.timestamps >= cur_ts - trail_len / fps) & (traj.timestamps <= cur_ts)
            trail_pts = traj.positions[trail_mask].astype(int)
            for k in range(1, len(trail_pts)):
                alpha = k / len(trail_pts)
                col = tuple(int(c * alpha) for c in color)
                cv2.line(frame, tuple(trail_pts[k-1]), tuple(trail_pts[k]), col, 2)

    # Draw bounding boxes and labels
    for obs in obs_list:
        tid = obs.track_id
        color = PALETTE[tid % len(PALETTE)]
        x1, y1, x2, y2 = int(obs.x1), int(obs.y1), int(obs.x2), int(obs.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"T{tid}:{obs.class_name[:3]}"
        cv2.putText(frame, label, (x1, max(y1-6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Active event overlay
    active_events = event_by_frame.get(frame_idx, [])
    for ev in active_events:
        _draw_event_banner(frame, ev)

    # Timestamp
    ts = frame_idx / fps if fps > 0 else frame_idx
    cv2.putText(frame, f"t={ts:.2f}s", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def _draw_event_banner(frame: np.ndarray, ev: ConflictEvent) -> None:
    h, w = frame.shape[:2]
    banner_color = (0, 0, 200) if ev.severity >= 0.7 else (0, 140, 255)
    cv2.rectangle(frame, (0, 0), (w, 32), banner_color, -1)
    ttc_str = f"{ev.ttc_min:.1f}s" if ev.ttc_min < math.inf else "—"
    pet_str = f"{ev.pet_min:.1f}s" if ev.pet_min < math.inf else "—"
    label = (f"EVENT {ev.event_id} | {ev.conflict_type} | "
             f"TTC={ttc_str} PET={pet_str} | "
             f"risk={ev.risk_score:.2f}")
    cv2.putText(frame, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
