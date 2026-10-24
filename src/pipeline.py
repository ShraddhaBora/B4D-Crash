"""
Before the Crash — Main Pipeline Orchestrator.

Runs the complete vertical slice:
  Video → Detection → Tracking → Trajectories → Candidates →
  TTC/PET → Evasive Actions → Conflict Classification → Risk Scoring →
  Event Merging → Evidence Generation → Hotspot Aggregation → Ranking

All output is written to the configured output_dir.
Designed to run on Nodus Compute or Kaggle GPU.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config_loader import load_config, ensure_output_dirs
from src.logger import get_logger, configure_root_logger
from src.schema import (
    ConflictEvent, Detection, HotspotSummary,
    Trajectory, TrackObservation,
)

log = get_logger(__name__)


class Pipeline:
    """
    End-to-end traffic conflict analysis pipeline.

    Usage:
        pipe = Pipeline("config.yaml")
        result = pipe.run("path/to/video.mp4")
    """

    def __init__(self, config_path: str = "config.yaml", overlay_path: Optional[str] = None):
        self.cfg = load_config(config_path, overlay_path)
        configure_root_logger(self.cfg)
        ensure_output_dirs(self.cfg)
        self._lazy_init()

    def _lazy_init(self):
        """Imports and initialises heavyweight components lazily (for Kaggle compat)."""
        det_cfg = self.cfg["detector"]
        trk_cfg = self.cfg["tracker"]

        from src.perception.detector import YOLODetector
        from src.perception.tracker import ByteTracker

        self.detector = YOLODetector(
            model_path=det_cfg["model"],
            confidence=det_cfg["confidence"],
            iou=det_cfg["iou"],
            class_ids=det_cfg.get("classes"),
        )
        self.tracker = ByteTracker(
            track_high_thresh=trk_cfg["track_high_thresh"],
            track_low_thresh=trk_cfg["track_low_thresh"],
            new_track_thresh=trk_cfg["new_track_thresh"],
            track_buffer=trk_cfg["track_buffer"],
        )

    def run(self, video_path: str) -> dict:
        """
        Execute the full pipeline on `video_path`.
        Returns a summary dict with paths to all outputs.
        """
        t0 = time.time()
        cfg = self.cfg
        out_dir = Path(cfg["video"]["output_dir"])
        vid_cfg = cfg["video"]
        traj_cfg = cfg["trajectory"]
        int_cfg = cfg["interaction"]
        saf_cfg = cfg["safety"]
        evt_cfg = cfg["events"]
        hot_cfg = cfg["hotspots"]

        log.info("=" * 60)
        log.info("Pipeline start | video: %s", video_path)
        log.info("=" * 60)

        # ── Phase 1: Video Ingestion ─────────────────────────────────────────
        from src.ingestion.video_reader import VideoReader, get_video_metadata
        meta = get_video_metadata(video_path)
        reader = VideoReader(
            video_path,
            frame_skip=vid_cfg.get("frame_skip", 1),
            resize_width=vid_cfg.get("resize_width", 0),
        )

        # ── Phase 2+3: Detection + Tracking ─────────────────────────────────
        all_detections: List[Detection] = []
        all_observations: List[TrackObservation] = []
        frames_processed = 0

        for frame, frame_idx, ts, fps in reader:
            detections = self.detector.detect(frame, frame_idx, ts)
            observations = self.tracker.update(detections, frame)
            all_detections.extend(detections)
            all_observations.extend(observations)
            frames_processed += 1

        log.info("Frames processed: %d | Detections: %d | Observations: %d",
                 frames_processed, len(all_detections), len(all_observations))

        # Save detections + tracks
        self._save_detections(all_detections, out_dir / "detections" / "detections.csv")
        self._save_tracks(all_observations, out_dir / "tracks" / "tracks.csv")

        # ── Phase 4+5+6+7: Trajectories ─────────────────────────────────────
        from src.trajectory.builder import build_trajectories
        trajectories = build_trajectories(
            all_observations,
            min_track_length=traj_cfg["min_track_length"],
            max_gap=traj_cfg["max_gap"],
            max_speed_pixels=traj_cfg["max_speed_pixels"],
            smoothing_window=traj_cfg["smoothing_window"],
        )
        self._save_trajectories(trajectories, out_dir / "trajectories" / "trajectories.csv")

        # ── Phase 8: Calibration (optional) ─────────────────────────────────
        from src.geometry.calibration import from_config as cal_from_config
        calibration = cal_from_config(cfg)

        # ── Phase 9: Zones ───────────────────────────────────────────────────
        from src.geometry.zones import make_grid_zones
        effective_w = vid_cfg.get("resize_width") or meta.width
        zone_manager = make_grid_zones(effective_w, meta.height, hot_cfg["grid_size"])

        # ── Phase 10: Candidate Pairs ────────────────────────────────────────
        from src.safety.candidate_pairs import generate_candidates
        candidates = generate_candidates(
            trajectories,
            max_distance=int_cfg["max_candidate_distance"],
            fps=meta.fps,
        )

        # ── Phases 11–15: TTC / PET / Evasive / Classify / Risk ─────────────
        from src.safety.ttc import compute_ttc
        from src.safety.pet import compute_pet
        from src.safety.evasive_action import detect_evasive_actions
        from src.safety.conflict_classifier import classify_conflict
        from src.safety.risk_score import compute_risk_score
        from src.events.event_manager import EventManager

        risk_weights = cfg.get("risk_weights", {})
        event_mgr = EventManager(
            merge_window=evt_cfg["merge_window"],
            minimum_duration=evt_cfg["minimum_duration"],
        )

        for cand in candidates:
            ta = trajectories[cand.track_id_a]
            tb = trajectories[cand.track_id_b]
            t_mid = (cand.start_time + cand.end_time) / 2.0

            ttc  = compute_ttc(ta, tb, cand, t_query=t_mid)
            pet  = compute_pet(ta, tb, cand)
            evs  = detect_evasive_actions(
                ta, tb, cand,
                speed_change_threshold=saf_cfg["speed_change_threshold"],
                heading_change_threshold=saf_cfg["heading_change_threshold"],
            )
            res  = classify_conflict(
                cand, ttc, pet, evs,
                ttc_threshold=saf_cfg["ttc_threshold"],
                pet_threshold=saf_cfg["pet_threshold"],
            )
            risk = compute_risk_score(
                cand, ttc, pet, evs,
                weights=risk_weights,
                ttc_threshold=saf_cfg["ttc_threshold"],
                pet_threshold=saf_cfg["pet_threshold"],
            )

            # Zone lookup at midpoint of both trajectories
            mid_pos = self._mid_position(ta, tb, t_mid)
            zone_id = zone_manager.zone_id(*mid_pos) if mid_pos else None

            event_mgr.record(cand, ttc, pet, evs, res, risk, zone_id)

        # ── Phase 16: Finalize Events ────────────────────────────────────────
        events: List[ConflictEvent] = event_mgr.finalize()
        self._save_events(events, out_dir / "events" / "conflict_events.csv")

        # ── Phase 17: Evidence ───────────────────────────────────────────────
        from src.events.evidence import generate_evidence_clips, generate_annotated_video
        clip_paths = generate_evidence_clips(
            video_path, events, trajectories, all_observations,
            output_dir=str(out_dir / "clips"),
        )
        annotated_path = generate_annotated_video(
            video_path, all_observations, events, trajectories,
            output_path=str(out_dir / "annotated_video.mp4"),
        )

        # ── Phase 18+19: Hotspots ────────────────────────────────────────────
        from src.hotspots.aggregation import aggregate_hotspots
        from src.hotspots.ranking import rank_hotspots
        hotspots = aggregate_hotspots(events, observation_duration_s=meta.duration_s)
        hotspots = rank_hotspots(hotspots)
        self._save_hotspots(hotspots, out_dir / "reports" / "hotspot_summary.csv")

        # ── Summary ──────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        summary = {
            "video": video_path,
            "frames_processed": frames_processed,
            "detections": len(all_detections),
            "tracks": len(trajectories),
            "valid_trajectories": sum(1 for t in trajectories.values() if t.quality == "GOOD"),
            "candidate_interactions": len(candidates),
            "conflict_events": len(events),
            "high_risk_events": sum(1 for e in events if e.severity >= 0.7),
            "hotspots": len(hotspots),
            "processing_time_s": round(elapsed, 2),
            "annotated_video": annotated_path,
            "evidence_clips": clip_paths,
        }
        summary_path = out_dir / "reports" / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)

        self._print_summary(summary)
        return summary

    # ── Save helpers ─────────────────────────────────────────────────────────

    def _save_detections(self, dets: List[Detection], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{
            "frame_id": d.frame_id, "timestamp": d.timestamp,
            "class_id": d.class_id, "class_name": d.class_name,
            "confidence": d.confidence,
            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
        } for d in dets]
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("Detections saved: %s", path)

    def _save_tracks(self, obs: List[TrackObservation], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{
            "frame_id": o.frame_id, "timestamp": o.timestamp,
            "track_id": o.track_id, "class_id": o.class_id,
            "class_name": o.class_name, "confidence": o.confidence,
            "x1": o.x1, "y1": o.y1, "x2": o.x2, "y2": o.y2,
            "center_x": o.center_x, "center_y": o.center_y,
            "foot_x": o.foot_x, "foot_y": o.foot_y,
        } for o in obs]
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("Tracks saved: %s", path)

    def _save_trajectories(self, trajs: Dict[int, Trajectory], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for tid, t in trajs.items():
            for i in range(len(t.timestamps)):
                rows.append({
                    "track_id": tid, "class_name": t.class_name,
                    "quality": t.quality,
                    "timestamp": t.timestamps[i],
                    "pos_x": t.positions[i, 0], "pos_y": t.positions[i, 1],
                    "raw_x": t.raw_positions[i, 0], "raw_y": t.raw_positions[i, 1],
                    "vel_x": t.velocities[i, 0], "vel_y": t.velocities[i, 1],
                    "speed": t.speeds[i], "heading": t.headings[i],
                    "acceleration": t.accelerations[i],
                })
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("Trajectories saved: %s", path)

    def _save_events(self, events: List[ConflictEvent], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{
            "event_id": e.event_id,
            "start_time": e.start_time, "end_time": e.end_time,
            "track_ids": str(e.track_ids), "classes": str(e.classes),
            "conflict_type": e.conflict_type, "zone_id": e.zone_id,
            "ttc_min": e.ttc_min if e.ttc_min < math.inf else None,
            "pet_min": e.pet_min if e.pet_min < math.inf else None,
            "max_speed_change": e.max_speed_change,
            "max_heading_change": e.max_heading_change,
            "severity": e.severity, "confidence": e.confidence,
            "risk_score": e.risk_score, "reasons": "|".join(e.reasons),
        } for e in events]
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("Events saved: %s  (%d events)", path, len(events))

    def _save_hotspots(self, hotspots: List[HotspotSummary], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{
            "zone_id": h.zone_id, "event_count": h.event_count,
            "high_risk_count": h.high_risk_count,
            "avg_risk_score": h.avg_risk_score,
            "min_ttc": h.min_ttc if h.min_ttc < math.inf else None,
            "min_pet": h.min_pet if h.min_pet < math.inf else None,
            "conflict_type_dist": str(h.conflict_type_dist),
            "road_user_dist": str(h.road_user_dist),
            "observation_duration_s": h.observation_duration_s,
            "exposure_based": h.exposure_based,
        } for h in hotspots]
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("Hotspot summary saved: %s", path)

    @staticmethod
    def _mid_position(ta: Trajectory, tb: Trajectory, t_mid: float):
        try:
            ax = float(np.interp(t_mid, ta.timestamps, ta.positions[:, 0]))
            ay = float(np.interp(t_mid, ta.timestamps, ta.positions[:, 1]))
            bx = float(np.interp(t_mid, tb.timestamps, tb.positions[:, 0]))
            by = float(np.interp(t_mid, tb.timestamps, tb.positions[:, 1]))
            return ((ax + bx) / 2, (ay + by) / 2)
        except Exception:
            return None

    @staticmethod
    def _print_summary(s: dict) -> None:
        log.info("-" * 60)
        log.info("Pipeline complete")
        log.info("  Video:                 %s", s["video"])
        log.info("  Frames processed:      %d", s["frames_processed"])
        log.info("  Detections:            %d", s["detections"])
        log.info("  Tracks:                %d", s["tracks"])
        log.info("  Valid trajectories:    %d", s["valid_trajectories"])
        log.info("  Candidate interactions:%d", s["candidate_interactions"])
        log.info("  Conflict events:       %d", s["conflict_events"])
        log.info("  High-risk events:      %d", s["high_risk_events"])
        log.info("  Hotspots:              %d", s["hotspots"])
        log.info("  Processing time:       %.1f s", s["processing_time_s"])
        log.info("-" * 60)
