"""
src/api_server.py — FastAPI backend serving Before the Crash & Agent37 intelligence.

Exposes clean REST API endpoints consuming:
  - data/dummy_risk_output.json (ML pipeline output)
  - outputs/reports/agent37_safety_intelligence.json (Agent37 intelligence output)
  - outputs/clips/*.mp4 (Evidence video clips)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(
    title="B4DCrash — Road Safety Intelligence API",
    version="1.0.0",
    description="REST API serving near-miss conflict events, spatial hotspots, and Agent37 safety intelligence."
)

# Enable CORS for frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dummy_risk_output.json"
AGENT_REPORT_PATH = BASE_DIR / "outputs" / "reports" / "agent37_safety_intelligence.json"
AGENT_DUMMY_REPORT_PATH = BASE_DIR / "outputs" / "reports" / "agent37_dummy_session_output.json"
CLIPS_DIR = BASE_DIR / "outputs" / "clips"

# Mount static video clips directory if it exists
if CLIPS_DIR.exists():
    app.mount("/static/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")


def load_risk_data() -> Dict[str, Any]:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Risk output data not found at {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_agent_report() -> Dict[str, Any]:
    # Prioritize agent37_safety_intelligence.json, fallback to agent37_dummy_session_output.json
    path = AGENT_REPORT_PATH if AGENT_REPORT_PATH.exists() else AGENT_DUMMY_REPORT_PATH
    if not path.exists():
        raise HTTPException(status_code=500, detail="Agent37 safety report not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "B4DCrash API",
        "version": "1.0.0",
        "data_loaded": DATA_PATH.exists(),
        "agent_report_loaded": AGENT_REPORT_PATH.exists() or AGENT_DUMMY_REPORT_PATH.exists(),
        "clips_dir_exists": CLIPS_DIR.exists()
    }


@app.get("/api/analysis/latest")
def get_latest_analysis() -> Dict[str, Any]:
    """Returns top-level session metadata and aggregated traffic statistics."""
    data = load_risk_data()
    agent_data = load_agent_report()
    
    return {
        "analysis_id": data.get("analysis_id"),
        "session_id": data.get("session_id"),
        "analysis_ts": data.get("analysis_ts"),
        "video_path": data.get("video_path"),
        "camera": data.get("camera"),
        "traffic_stats": data.get("traffic_stats"),
        "ttc_threshold_s": data.get("ttc_threshold_s", 3.0),
        "pet_threshold_s": data.get("pet_threshold_s", 2.0),
        "total_event_count": data.get("total_event_count", len(data.get("conflict_events", []))),
        "high_severity_count": data.get("high_severity_count", 0),
        "peak_risk_score": data.get("peak_risk_score", 0.0),
        "avg_risk_score": data.get("avg_risk_score", 0.0),
        "conflict_type_counts": data.get("conflict_type_counts", {}),
        "overall_assessment": agent_data.get("overall_assessment", {}),
        "agent_version": agent_data.get("agent_version", "37.0"),
        "generated_at": agent_data.get("generated_at")
    }


@app.get("/api/hotspots")
def get_hotspots() -> List[Dict[str, Any]]:
    """Returns spatial conflict hotspots enriched with priority and Agent37 linkages."""
    risk_data = load_risk_data()
    agent_data = load_agent_report()
    
    hotspots = risk_data.get("hotspots", [])
    prioritized_hotspots = {
        h["zone_id"]: h for h in agent_data.get("prioritized_hotspots", [])
    }
    
    enriched = []
    for h in hotspots:
        zid = h.get("zone_id")
        agent_h = prioritized_hotspots.get(zid, {})
        
        # Link findings that reference any events in this hotspot
        zone_events = set(h.get("event_ids", []))
        related_findings = [
            f["finding_id"] for f in agent_data.get("key_findings", [])
            if any(eid in zone_events for eid in f.get("evidence_event_ids", []))
        ]
        related_recommendations = [
            r["recommendation_id"] for r in agent_data.get("recommendations", [])
            if any(eid in zone_events for eid in r.get("evidence_event_ids", []))
        ]
        
        enriched_h = dict(h)
        enriched_h["priority"] = agent_h.get("priority", 99)
        enriched_h["risk_level"] = agent_h.get("risk_level", "LOW")
        enriched_h["reason"] = agent_h.get("reason", "")
        enriched_h["related_finding_ids"] = related_findings
        enriched_h["related_recommendation_ids"] = related_recommendations
        enriched.append(enriched_h)
        
    enriched.sort(key=lambda x: x.get("priority", 99))
    return enriched


@app.get("/api/events")
def get_events(
    zone_id: Optional[str] = None,
    conflict_type: Optional[str] = None,
    min_risk: Optional[float] = None,
    min_severity: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Returns conflict events with optional filtering."""
    data = load_risk_data()
    events = data.get("conflict_events", [])
    
    # Check evidence availability on filesystem
    for ev in events:
        clip_path = ev.get("evidence_clip_path")
        if clip_path:
            filename = Path(clip_path).name
            full_clip = CLIPS_DIR / filename
            ev["evidence_available"] = full_clip.exists()
            ev["evidence_url"] = f"/static/clips/{filename}" if full_clip.exists() else None
        else:
            ev["evidence_available"] = False
            ev["evidence_url"] = None

    filtered = events
    if zone_id:
        filtered = [e for e in filtered if e.get("zone_id") == zone_id]
    if conflict_type:
        filtered = [e for e in filtered if e.get("conflict_type") == conflict_type]
    if min_risk is not None:
        filtered = [e for e in filtered if e.get("risk_score", 0.0) >= min_risk]
    if min_severity is not None:
        filtered = [e for e in filtered if e.get("severity", 0.0) >= min_severity]
        
    return filtered


@app.get("/api/events/{event_id}")
def get_event_detail(event_id: str) -> Dict[str, Any]:
    """Returns single conflict event with full traceability and Agent37 context."""
    data = load_risk_data()
    agent_data = load_agent_report()
    
    event = next((e for e in data.get("conflict_events", []) if e.get("event_id") == event_id), None)
    if not event:
        raise HTTPException(status_code=404, detail=f"Conflict event {event_id} not found")
        
    event_dict = dict(event)
    
    # Evidence clip verification
    clip_path = event_dict.get("evidence_clip_path")
    if clip_path:
        filename = Path(clip_path).name
        full_clip = CLIPS_DIR / filename
        event_dict["evidence_available"] = full_clip.exists()
        event_dict["evidence_url"] = f"/static/clips/{filename}" if full_clip.exists() else None
    else:
        event_dict["evidence_available"] = False
        event_dict["evidence_url"] = None

    # Connect Agent37 findings
    event_dict["related_findings"] = [
        f for f in agent_data.get("key_findings", [])
        if event_id in f.get("evidence_event_ids", [])
    ]
    
    # Connect Agent37 recommendations
    event_dict["related_recommendations"] = [
        r for r in agent_data.get("recommendations", [])
        if event_id in r.get("evidence_event_ids", [])
    ]

    # Connect Agent37 patterns
    event_dict["related_patterns"] = [
        p for p in agent_data.get("patterns", [])
        if event_id in p.get("evidence_event_ids", [])
    ]
    
    return event_dict


@app.get("/api/agent-report")
def get_agent_report() -> Dict[str, Any]:
    """Returns complete structured Agent37 report."""
    return load_agent_report()


@app.get("/api/traceability")
def get_traceability_graph() -> Dict[str, Any]:
    """
    Returns the complete end-to-end evidence graph:
    Hotspot -> Events -> ML Metrics -> Findings -> Recommendations.
    """
    risk_data = load_risk_data()
    agent_data = load_agent_report()
    
    events = {e["event_id"]: e for e in risk_data.get("conflict_events", [])}
    hotspots = {h["zone_id"]: h for h in risk_data.get("hotspots", [])}
    findings = {f["finding_id"]: f for f in agent_data.get("key_findings", [])}
    recommendations = {r["recommendation_id"]: r for r in agent_data.get("recommendations", [])}
    
    chains = []
    for rec in agent_data.get("recommendations", []):
        rec_id = rec.get("recommendation_id")
        rec_events = rec.get("evidence_event_ids", [])
        rec_findings = rec.get("finding_ids", [])
        
        for eid in rec_events:
            ev = events.get(eid, {})
            zid = ev.get("zone_id")
            hotspot = hotspots.get(zid, {})
            
            chains.append({
                "recommendation_id": rec_id,
                "recommendation_action": rec.get("action"),
                "recommendation_priority": rec.get("priority"),
                "recommendation_category": rec.get("category"),
                "finding_ids": rec_findings,
                "event_id": eid,
                "zone_id": zid,
                "hotspot_priority": hotspot.get("priority", 1 if zid == "R1C2" else 2 if zid == "R2C3" else 3),
                "conflict_type": ev.get("conflict_type"),
                "risk_score": ev.get("risk_score"),
                "severity": ev.get("severity"),
                "ttc_min": ev.get("ttc_min"),
                "pet_min": ev.get("pet_min"),
                "reasons": ev.get("reasons", [])
            })
            
    return {
        "analysis_id": risk_data.get("analysis_id"),
        "chains": chains,
        "nodes": {
            "hotspots": hotspots,
            "events": events,
            "findings": findings,
            "recommendations": recommendations,
            "patterns": {p["pattern_id"]: p for p in agent_data.get("patterns", [])}
        }
    }


import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
