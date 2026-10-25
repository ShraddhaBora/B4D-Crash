#!/usr/bin/env python3
"""
dry_run.py — Full pipeline dry-run using a synthetic video + mock detections.

Validates every module WITHOUT requiring ultralytics, a GPU, or real video.
Designed to run on local Windows dev machine to confirm the pipeline is wired
correctly before handing off to Nodus/Kaggle.

Usage:
    python scripts/dry_run.py
    python scripts/dry_run.py --frames 300 --tracks 8
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ------------------------------------------------------------------------------
# 1. Synthetic video generator
# ------------------------------------------------------------------------------

def make_synthetic_video(path: str, n_frames: int = 150, fps: float = 15.0,
                          width: int = 640, height: int = 480) -> None:
    """
    Write a short synthetic traffic-scene video.
    Draws moving coloured rectangles on a grey background.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.default_rng(42)

    for i in range(n_frames):
        frame = np.full((height, width, 3), 80, dtype=np.uint8)
        # Draw some lane markings
        for x in range(0, width, 60):
            cv2.line(frame, (x, height // 2), (x + 30, height // 2), (200, 200, 200), 2)
        # Moving rectangles (synthetic "vehicles")
        t = i / fps
        for j in range(5):
            x = int((j * 130 + t * 50 * (1 + j * 0.1)) % width)
            y = int(100 + j * 60 + 10 * math.sin(t + j))
            color = [(0, 100, 255), (0, 200, 0), (200, 0, 0), (200, 200, 0), (0, 200, 200)][j]
            cv2.rectangle(frame, (x, y), (x + 60, y + 30), color, -1)
        out.write(frame)
    out.release()


# ------------------------------------------------------------------------------
# 2. Mock detector (returns synthetic detections without YOLO)
# ------------------------------------------------------------------------------

class MockDetector:
    """
    Produces synthetic Detection objects for dry-run testing.
    Simulates N tracked road-users moving across the scene, including
    one vehicle-pedestrian near-miss scenario.
    """

    def __init__(self, n_tracks: int = 6, fps: float = 15.0,
                 width: int = 640, height: int = 480, seed: int = 42):
        rng = random.Random(seed)
        # Each track: (track_id, class, start_frame, trajectory_fn)
        self._tracks = []
        classes_pool = ["car", "car", "car", "motorcycle", "person", "person"]
        for tid in range(n_tracks):
            cls = classes_pool[tid % len(classes_pool)]
            start = rng.randint(0, 20)
            # Give tracks deterministic motion paths
            speed_x = rng.uniform(3, 8) * (1 if tid % 2 == 0 else -1)
            speed_y = rng.uniform(-1, 1)
            ox = rng.randint(50, width - 150)
            oy = rng.randint(80, height - 100)
            self._tracks.append((tid + 1, cls, start, ox, oy, speed_x, speed_y))
        self._fps = fps
        self._w = width
        self._h = height

    def detect(self, frame_id: int, timestamp: float):
        from src.schema import Detection
        dets = []
        for tid, cls, start, ox, oy, sx, sy in self._tracks:
            if frame_id < start:
                continue
            t = (frame_id - start) / self._fps
            x = (ox + sx * t) % self._w
            y = max(40, min(self._h - 80, oy + sy * t))
            bw = 55 if cls != "person" else 25
            bh = 30 if cls != "person" else 60
            # Inject a near-miss: tracks 1 and 5 converge around frame 70-90
            if tid in (1, 5) and 60 <= frame_id <= 100:
                x += (frame_id - 60) * (2 if tid == 1 else -2)
            dets.append(Detection(
                frame_id=frame_id, timestamp=timestamp,
                class_id={"car": 2, "motorcycle": 3, "person": 0}[cls],
                class_name=cls,
                confidence=round(random.uniform(0.55, 0.95), 2),
                x1=float(x), y1=float(y),
                x2=float(x + bw), y2=float(y + bh),
            ))
        return dets


# ------------------------------------------------------------------------------
# 3. Mock tracker (wraps the real IoU tracker, bypasses ByteTrack import)
# ------------------------------------------------------------------------------

class MockTracker:
    """Uses the real _SimpleIoUTracker without requiring Ultralytics BYTETracker."""

    def __init__(self):
        from src.perception.tracker import _SimpleIoUTracker
        self._t = _SimpleIoUTracker(iou_threshold=0.3, max_age=30)

    def update(self, detections, frame=None):
        from src.schema import TrackObservation
        pairs = self._t.update(detections)
        obs = []
        for tid, det in pairs:
            obs.append(TrackObservation(
                frame_id=det.frame_id, timestamp=det.timestamp,
                track_id=tid, class_id=det.class_id,
                class_name=det.class_name, confidence=det.confidence,
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
            ))
        return obs


# ------------------------------------------------------------------------------
# 4. Full pipeline dry-run
# ------------------------------------------------------------------------------

def run_dry_run(n_frames: int = 150, n_tracks: int = 6,
                fps: float = 15.0, output_dir: str = "outputs/dry_run"):
    t0 = time.time()

    print("=" * 60)
    print("Before the Crash — Pipeline Dry Run")
    print("=" * 60)

    # Config — resolve relative to project root, NOT to the CWD.
    # On Kaggle the CWD is /kaggle/working/ but the config lives inside the repo.
    _cfg_path = str(_ROOT / "config.yaml")
    from src.config_loader import load_config, ensure_output_dirs
    cfg = load_config(_cfg_path)
    cfg["video"]["output_dir"] = output_dir
    ensure_output_dirs(cfg)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from src.logger import configure_root_logger
    configure_root_logger(cfg)

    W, H = 640, 480

    # -- Step 1: Synthetic video -----------------------------------------------
    vid_path = str(Path(output_dir) / "synthetic.mp4")
    print(f"\n[1/8] Generating synthetic video ({n_frames} frames @ {fps} fps)...")
    make_synthetic_video(vid_path, n_frames=n_frames, fps=fps, width=W, height=H)
    print(f"      [OK] {vid_path}")

    # -- Step 2: Ingestion -----------------------------------------------------
    print("\n[2/8] Video ingestion...")
    from src.ingestion.video_reader import VideoReader, get_video_metadata
    meta = get_video_metadata(vid_path)
    print(f"      [OK] {meta.fps:.1f} fps | {meta.frame_count} frames | {meta.width}x{meta.height}")

    # -- Step 3: Detection (mock) ----------------------------------------------
    print(f"\n[3/8] Mock detection + tracking ({n_tracks} synthetic tracks)...")
    detector = MockDetector(n_tracks=n_tracks, fps=fps, width=W, height=H)
    tracker  = MockTracker()
    reader   = VideoReader(vid_path, frame_skip=1)

    all_dets, all_obs = [], []
    for frame, fid, ts, _ in reader:
        dets = detector.detect(fid, ts)
        obs  = tracker.update(dets)
        all_dets.extend(dets)
        all_obs.extend(obs)

    print(f"      [OK] Detections: {len(all_dets)} | Track-observations: {len(all_obs)}")

    # Save
    import pandas as pd
    import math as _math
    Path(output_dir, "detections").mkdir(parents=True, exist_ok=True)
    Path(output_dir, "tracks").mkdir(parents=True, exist_ok=True)
    det_rows = [{"frame_id": d.frame_id, "timestamp": d.timestamp,
                 "class_name": d.class_name, "confidence": d.confidence,
                 "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2} for d in all_dets]
    pd.DataFrame(det_rows).to_csv(f"{output_dir}/detections/detections.csv", index=False)

    trk_rows = [{"frame_id": o.frame_id, "timestamp": o.timestamp,
                 "track_id": o.track_id, "class_name": o.class_name,
                 "foot_x": o.foot_x, "foot_y": o.foot_y} for o in all_obs]
    pd.DataFrame(trk_rows).to_csv(f"{output_dir}/tracks/tracks.csv", index=False)

    # -- Step 4: Trajectories --------------------------------------------------
    print("\n[4/8] Trajectory reconstruction + smoothing + motion estimation...")
    from src.trajectory.builder import build_trajectories
    trajs = build_trajectories(
        all_obs,
        min_track_length=cfg["trajectory"]["min_track_length"],
        max_gap=cfg["trajectory"]["max_gap"],
        max_speed_pixels=cfg["trajectory"]["max_speed_pixels"],
        smoothing_window=cfg["trajectory"]["smoothing_window"],
    )
    good = {tid: t for tid, t in trajs.items() if t.quality == "GOOD"}
    print(f"      [OK] Trajectories: {len(trajs)} total | GOOD: {len(good)}")

    # -- Step 5: Zones ---------------------------------------------------------
    print("\n[5/8] Zone setup...")
    from src.geometry.zones import make_grid_zones
    zone_mgr = make_grid_zones(W, H, grid_size=5)
    print(f"      [OK] {len(zone_mgr.zones)} grid zones")

    # -- Step 6: Candidate pairs -----------------------------------------------
    print("\n[6/8] Candidate interaction generation...")
    from src.safety.candidate_pairs import generate_candidates
    candidates = generate_candidates(
        trajs,
        max_distance=cfg["interaction"]["max_candidate_distance"],
        fps=fps,
    )
    print(f"      [OK] Candidates: {len(candidates)}")

    # -- Step 7: Safety analysis + event merging -------------------------------
    print("\n[7/8] TTC / PET / evasive action / classification / risk scoring...")
    from src.safety.ttc import compute_ttc
    from src.safety.pet import compute_pet
    from src.safety.evasive_action import detect_evasive_actions
    from src.safety.conflict_classifier import classify_conflict
    from src.safety.risk_score import compute_risk_score
    from src.events.event_manager import EventManager
    from src.schema import ConflictClass

    evmgr = EventManager(
        merge_window=cfg["events"]["merge_window"],
        minimum_duration=cfg["events"]["minimum_duration"],
    )
    conflict_radius = cfg.get("safety", {}).get("conflict_radius", 50.0)
    for cand in candidates:
        ta = trajs[cand.track_id_a]
        tb = trajs[cand.track_id_b]
        t_start = cand.start_time
        t_end = cand.end_time
        sample_ts = np.linspace(t_start, t_end, 5) if t_end > t_start else np.array([t_start])
        pet = compute_pet(ta, tb, cand)

        for t_sample in sample_ts:
            t = float(t_sample)
            ttc  = compute_ttc(ta, tb, cand, t_query=t, conflict_radius=conflict_radius)
            evs  = detect_evasive_actions(ta, tb, cand,
                       speed_change_threshold=cfg["safety"]["speed_change_threshold"],
                       heading_change_threshold=cfg["safety"]["heading_change_threshold"],
                       peak_conflict_time=t)
            res  = classify_conflict(cand, ttc, pet, evs,
                       ttc_threshold=cfg["safety"]["ttc_threshold"],
                       pet_threshold=cfg["safety"]["pet_threshold"])
            risk = compute_risk_score(cand, ttc, pet, evs,
                       weights=cfg.get("risk_weights", {}),
                       ttc_threshold=cfg["safety"]["ttc_threshold"],
                       pet_threshold=cfg["safety"]["pet_threshold"])
            mid_x = float(np.interp(t, ta.timestamps, ta.positions[:, 0]))
            mid_y = float(np.interp(t, ta.timestamps, ta.positions[:, 1]))
            zone_id = zone_mgr.zone_id(mid_x, mid_y)
            evmgr.record(cand, ttc, pet, evs, res, risk, zone_id, timestamp=t)

    events = evmgr.finalize()
    high_risk = [e for e in events if e.severity >= 0.7]
    print(f"      [OK] Conflict events: {len(events)} | High-risk: {len(high_risk)}")

    # Save events
    Path(output_dir, "events").mkdir(parents=True, exist_ok=True)
    ev_rows = [{"event_id": e.event_id, "start_time": e.start_time,
                "end_time": e.end_time, "conflict_type": e.conflict_type,
                "zone_id": e.zone_id, "severity": e.severity,
                "risk_score": e.risk_score, "reasons": "|".join(e.reasons)}
               for e in events]
    pd.DataFrame(ev_rows).to_csv(f"{output_dir}/events/conflict_events.csv", index=False)

    # -- Step 8: Evidence + annotated video -----------------------------------
    print("\n[8/8] Generating annotated video + evidence clips...")
    from src.events.evidence import generate_evidence_clips, generate_annotated_video
    Path(output_dir, "clips").mkdir(parents=True, exist_ok=True)
    clips = generate_evidence_clips(
        vid_path, events[:3], trajs, all_obs,
        output_dir=f"{output_dir}/clips",
    )
    ann_path = generate_annotated_video(
        vid_path, all_obs, events, trajs,
        output_path=f"{output_dir}/annotated_video.mp4",
    )

    # -- Hotspots --------------------------------------------------------------
    from src.hotspots.aggregation import aggregate_hotspots
    from src.hotspots.ranking import rank_hotspots
    hotspots = rank_hotspots(aggregate_hotspots(events, meta.duration_s))
    Path(output_dir, "reports").mkdir(parents=True, exist_ok=True)
    hot_rows = [{"zone_id": h.zone_id, "event_count": h.event_count,
                 "avg_risk": h.avg_risk_score} for h in hotspots]
    pd.DataFrame(hot_rows).to_csv(f"{output_dir}/reports/hotspot_summary.csv", index=False)

    elapsed = time.time() - t0

    # -- Summary ---------------------------------------------------------------
    print()
    print("-" * 60)
    print("DRY RUN COMPLETE [OK]")
    print("-" * 60)
    print(f"  Frames processed:       {n_frames}")
    print(f"  Detections:             {len(all_dets)}")
    print(f"  Track observations:     {len(all_obs)}")
    print(f"  Trajectories (GOOD):    {len(good)}")
    print(f"  Candidate interactions: {len(candidates)}")
    print(f"  Conflict events:        {len(events)}")
    print(f"  High-risk events:       {len(high_risk)}")
    print(f"  Evidence clips:         {len(clips)}")
    print(f"  Hotspot zones:          {len(hotspots)}")
    print(f"  Wall-clock time:        {elapsed:.1f} s")
    print(f"  Output dir:             {Path(output_dir).resolve()}")
    print("-" * 60)
    print()
    print("All pipeline modules validated. Ready for Nodus/Kaggle GPU run.")
    print("Next step: copy repo to Nodus, set VIDEO_PATH in nodus_kaggle_runner.py, run.")

    return {"events": len(events), "high_risk": len(high_risk), "elapsed": elapsed}


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Before the Crash — Dry Run")
    parser.add_argument("--frames", type=int, default=150, help="Number of synthetic frames")
    parser.add_argument("--tracks", type=int, default=6,   help="Number of synthetic tracks")
    parser.add_argument("--output", default="outputs/dry_run", help="Output directory")
    args = parser.parse_args()
    run_dry_run(n_frames=args.frames, n_tracks=args.tracks, output_dir=args.output)
