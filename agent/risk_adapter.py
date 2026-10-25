"""
agent/risk_adapter.py — Adapter: ML risk-engine outputs → RiskAnalysisInput.

Two modes:
  1. From JSON  (current): load dummy_risk_output.json → RiskAnalysisInput
  2. From CSVs  (future):  load conflict_events.csv + hotspot_summary.csv
                            → RiskAnalysisInput (same output)

Agent37 always sees RiskAnalysisInput. Switching from dummy data to the real
pipeline output requires only changes here — not in agent37.py or schemas.py.

CSV column names are chosen to match what src/pipeline.py _save_events() and
_save_hotspots() write. If those methods change their column names, update the
constant mappings below.
"""

from __future__ import annotations

import ast
import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    CameraInfo,
    ConflictEventInput,
    HotspotInput,
    RiskAnalysisInput,
    TrafficStats,
)


# ──────────────────────────────────────────────────────────────────────────────
# CSV column name mappings (mirror src/pipeline.py _save_events/_save_hotspots)
# ──────────────────────────────────────────────────────────────────────────────

# conflict_events.csv columns produced by Pipeline._save_events()
_EVENT_COL = {
    "event_id":           "event_id",
    "start_time":         "start_time",
    "end_time":           "end_time",
    "track_ids":          "track_ids",       # stored as string repr of list
    "classes":            "classes",         # stored as string repr of list
    "conflict_type":      "conflict_type",
    "zone_id":            "zone_id",
    "ttc_min":            "ttc_min",
    "pet_min":            "pet_min",
    "max_speed_change":   "max_speed_change",
    "max_heading_change": "max_heading_change",
    "severity":           "severity",
    "confidence":         "confidence",
    "risk_score":         "risk_score",
    "reasons":            "reasons",         # stored as pipe-separated string
    "peak_conflict_time": "peak_conflict_time",
    "start_frame":        "start_frame",
    "end_frame":          "end_frame",
}

# hotspot_summary.csv columns produced by Pipeline._save_hotspots()
_HOTSPOT_COL = {
    "zone_id":                  "zone_id",
    "event_count":              "event_count",
    "high_risk_count":          "high_risk_count",
    "avg_risk_score":           "avg_risk_score",
    "min_ttc":                  "min_ttc",
    "min_pet":                  "min_pet",
    "conflict_type_dist":       "conflict_type_dist",  # string repr of dict
    "road_user_dist":           "road_user_dist",      # string repr of dict
    "observation_duration_s":   "observation_duration_s",
    "exposure_based":           "exposure_based",
}


# ──────────────────────────────────────────────────────────────────────────────
# From JSON (dummy / structured)
# ──────────────────────────────────────────────────────────────────────────────

def from_json(path: str | Path) -> RiskAnalysisInput:
    """
    Load a RiskAnalysisInput from a structured JSON file.

    The JSON file must conform to the RiskAnalysisInput schema.
    This is the current path (dummy data).  When the real pipeline
    is ready, use from_csv() instead.

    Raises:
        FileNotFoundError if the path does not exist.
        pydantic.ValidationError if the JSON does not match the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Risk engine output not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    return RiskAnalysisInput.model_validate(raw)


# ──────────────────────────────────────────────────────────────────────────────
# From CSVs (real pipeline — future path)
# ──────────────────────────────────────────────────────────────────────────────

def from_csv(
    events_csv: str | Path,
    hotspots_csv: str | Path,
    analysis_id: Optional[str] = None,
    video_path: Optional[str] = None,
    ttc_threshold_s: float = 3.0,
    pet_threshold_s: float = 2.0,
    camera_info: Optional[Dict[str, Any]] = None,
    traffic_stats: Optional[Dict[str, Any]] = None,
) -> RiskAnalysisInput:
    """
    Build a RiskAnalysisInput from the real pipeline CSV outputs.

    Args:
        events_csv:     Path to outputs/events/conflict_events.csv
        hotspots_csv:   Path to outputs/reports/hotspot_summary.csv
        analysis_id:    Optional explicit ID (auto-generated if None).
        video_path:     Optional path to the source video.
        ttc_threshold_s: TTC threshold from config.yaml (default 3.0).
        pet_threshold_s: PET threshold from config.yaml (default 2.0).
        camera_info:    Optional CameraInfo dict.
        traffic_stats:  Optional TrafficStats dict.

    Returns:
        Validated RiskAnalysisInput ready for Agent37.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for from_csv(). Install it: pip install pandas"
        ) from e

    events_path   = Path(events_csv)
    hotspots_path = Path(hotspots_csv)

    for p in (events_path, hotspots_path):
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    ev_df  = pd.read_csv(events_path)
    hot_df = pd.read_csv(hotspots_path)

    # ── Parse events ──────────────────────────────────────────────────────────
    events: List[ConflictEventInput] = []
    for _, row in ev_df.iterrows():
        # Parse list columns stored as string representations
        c_type = str(row.get(_EVENT_COL["conflict_type"], "UNKNOWN"))
        default_classes = ["vehicle", "pedestrian"] if "PED" in c_type.upper() else ["vehicle", "vehicle"]

        try:
            val = row.get(_EVENT_COL["track_ids"])
            track_ids = ast.literal_eval(str(val)) if pd.notna(val) else [0, 1]
            if not isinstance(track_ids, list) or len(track_ids) < 2:
                track_ids = [0, 1]
        except Exception:
            track_ids = [0, 1]

        try:
            val = row.get(_EVENT_COL["classes"])
            classes = ast.literal_eval(str(val)) if pd.notna(val) else default_classes
            if not isinstance(classes, list) or len(classes) < 2:
                classes = default_classes
        except Exception:
            classes = default_classes

        reasons_raw = str(row.get(_EVENT_COL["reasons"], "")) if pd.notna(row.get(_EVENT_COL["reasons"])) else ""
        reasons = [r.strip() for r in reasons_raw.split("|") if r.strip()]

        ttc_raw = row.get(_EVENT_COL["ttc_min"])
        pet_raw = row.get(_EVENT_COL["pet_min"])

        def _safe_float(v, allow_inf: bool = True) -> Optional[float]:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            try:
                f = float(v)
                if math.isinf(f):
                    return None if not allow_inf else None
                return f
            except (TypeError, ValueError):
                return None

        events.append(ConflictEventInput(
            event_id=str(row.get(_EVENT_COL["event_id"], uuid.uuid4().hex[:8])),
            start_time=float(row.get(_EVENT_COL["start_time"], 0.0)),
            end_time=float(row.get(_EVENT_COL["end_time"], 0.0)),
            start_frame=int(row[_EVENT_COL["start_frame"]]) if _EVENT_COL["start_frame"] in row and pd.notna(row[_EVENT_COL["start_frame"]]) else None,
            end_frame=int(row[_EVENT_COL["end_frame"]]) if _EVENT_COL["end_frame"] in row and pd.notna(row[_EVENT_COL["end_frame"]]) else None,
            track_ids=track_ids,
            classes=classes,
            conflict_type=c_type,
            zone_id=str(row[_EVENT_COL["zone_id"]]) if pd.notna(row.get(_EVENT_COL["zone_id"])) else None,
            ttc_min=_safe_float(ttc_raw),
            pet_min=_safe_float(pet_raw),
            max_speed_change=_safe_float(row.get(_EVENT_COL["max_speed_change"])),
            max_heading_change=_safe_float(row.get(_EVENT_COL["max_heading_change"])),
            severity=float(row.get(_EVENT_COL["severity"], 0.0)),
            confidence=float(row.get(_EVENT_COL["confidence"], 0.8)),
            risk_score=float(row.get(_EVENT_COL["risk_score"], 0.0)),
            reasons=reasons,
            peak_conflict_time=_safe_float(row.get(_EVENT_COL["peak_conflict_time"])),
        ))

    # ── Build zone → event_ids index ──────────────────────────────────────────
    zone_event_ids: Dict[str, List[str]] = {}
    for e in events:
        z = e.zone_id or "UNKNOWN"
        zone_event_ids.setdefault(z, []).append(e.event_id)

    # ── Parse hotspots ────────────────────────────────────────────────────────
    hotspots: List[HotspotInput] = []
    for _, row in hot_df.iterrows():
        zone_id = str(row.get(_HOTSPOT_COL["zone_id"], "UNKNOWN"))

        def _parse_dict(val: Any) -> Dict[str, int]:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return {}
            try:
                return ast.literal_eval(str(val))
            except Exception:
                return {}

        avg_risk_val = row.get(_HOTSPOT_COL["avg_risk_score"])
        if pd.isna(avg_risk_val):
            avg_risk_val = row.get("avg_risk", 0.0)
        avg_risk = float(avg_risk_val) if pd.notna(avg_risk_val) else 0.0

        ev_cnt = int(row.get(_HOTSPOT_COL["event_count"], 0)) if pd.notna(row.get(_HOTSPOT_COL["event_count"])) else 0
        hi_cnt = int(row.get(_HOTSPOT_COL["high_risk_count"], 0)) if pd.notna(row.get(_HOTSPOT_COL["high_risk_count"])) else 0
        obs_dur = float(row.get(_HOTSPOT_COL["observation_duration_s"], 0.0)) if pd.notna(row.get(_HOTSPOT_COL["observation_duration_s"])) else 0.0

        hotspots.append(HotspotInput(
            zone_id=zone_id,
            event_count=ev_cnt,
            high_risk_count=hi_cnt,
            avg_risk_score=avg_risk,
            min_ttc=_safe_float(row.get(_HOTSPOT_COL["min_ttc"])),
            min_pet=_safe_float(row.get(_HOTSPOT_COL["min_pet"])),
            conflict_type_dist=_parse_dict(row.get(_HOTSPOT_COL["conflict_type_dist"])),
            road_user_dist=_parse_dict(row.get(_HOTSPOT_COL["road_user_dist"])),
            observation_duration_s=obs_dur,
            exposure_based=bool(row.get(_HOTSPOT_COL["exposure_based"], False)),
            event_ids=zone_event_ids.get(zone_id, []),
        ))

    return RiskAnalysisInput(
        analysis_id=analysis_id or f"ANALYSIS_{uuid.uuid4().hex[:8].upper()}",
        conflict_events=events,
        hotspots=hotspots,
        ttc_threshold_s=ttc_threshold_s,
        pet_threshold_s=pet_threshold_s,
        video_path=video_path,
        camera=CameraInfo(**camera_info) if camera_info else None,
        traffic_stats=TrafficStats(**traffic_stats) if traffic_stats else None,
    )
