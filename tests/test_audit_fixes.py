"""
Deterministic unit tests for the 10 audit fixes.

Tests cover:
  1. Fix #1: TTC closest approach requires predicted minimum separation < conflict_radius.
  2. Fix #2: Multi-timestamp TTC sampling returns min valid TTC and peak conflict timestamp.
  3. Fix #3: EventManager accumulates frame alerts and merges temporally proximate clusters.
  4. Fix #4: Evasive actions require proximity to peak conflict timestamp.
  5. Fix #5: PET marks non-crossing geometries as inapplicable.
  6. Fix #6: Candidate generation requires sustained approach (>=3 steps & >=25% window).
  7. Fix #7: Image-space MVP status without calibration.
  8. Fix #8: Evaluator requires conflict-type compatibility alongside temporal IoU.
  9. Fix #9: Agent37 dynamic next config generation based on empirical metrics.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import (
    CandidateInteraction,
    ConflictClass,
    ConflictEvent,
    ConflictResult,
    PETResult,
    RiskScore,
    TrackQuality,
    Trajectory,
    TTCResult,
)
from src.evaluation.evaluator import GroundTruthEvent


def _make_dummy_traj(
    track_id: int,
    positions: list[list[float]],
    timestamps: list[float] | None = None,
    class_name: str = "car",
) -> Trajectory:
    pos = np.array(positions, dtype=float)
    n = len(pos)
    if timestamps is None:
        ts = np.linspace(0.0, (n - 1) * 0.1, n)
    else:
        ts = np.array(timestamps, dtype=float)

    vels = np.zeros_like(pos)
    if n > 1:
        dt = np.diff(ts)[:, None]
        vels[:-1] = np.diff(pos, axis=0) / np.maximum(dt, 1e-6)
        vels[-1] = vels[-2]
    speeds = np.linalg.norm(vels, axis=1)
    headings = np.zeros(n)
    accels = np.zeros(n)

    return Trajectory(
        track_id=track_id,
        class_name=class_name,
        quality=TrackQuality.GOOD,
        timestamps=ts,
        raw_positions=pos,
        positions=pos,
        velocities=vels,
        speeds=speeds,
        headings=headings,
        accelerations=accels,
    )


# ── Fix #1: TTC closest approach conflict radius ──────────────────────────────
def test_ttc_closest_approach_conflict_radius():
    from src.safety.ttc import compute_ttc

    # Two objects moving parallel in opposite directions on lines separated by 100px.
    # Closest separation will be 100px, which is > default conflict_radius (50px).
    t_start, t_end = 0.0, 5.0
    ts = np.linspace(t_start, t_end, 20)
    # A moves from (0, 0) to (500, 0)
    pos_a = [[float(x), 0.0] for x in np.linspace(0, 500, 20)]
    # B moves from (500, 100) to (0, 100)
    pos_b = [[float(x), 100.0] for x in np.linspace(500, 0, 20)]

    ta = _make_dummy_traj(1, pos_a, list(ts))
    tb = _make_dummy_traj(2, pos_b, list(ts))
    cand = CandidateInteraction(
        track_id_a=1, track_id_b=2,
        class_a="car", class_b="car",
        interaction_type="VEH_VEH",
        start_time=t_start, end_time=t_end,
        start_frame=0, end_frame=20,
    )

    # At t=0.5, they are closing, but min separation will be 100px (> conflict_radius=50px).
    res = compute_ttc(ta, tb, cand, t_query=0.5, conflict_radius=50.0)
    assert not res.valid, "TTC should be invalid when min separation > conflict_radius"
    assert math.isinf(res.value)

    # With a larger conflict radius of 150px, it should be considered a conflict
    res_large_radius = compute_ttc(ta, tb, cand, t_query=0.5, conflict_radius=150.0)
    assert res_large_radius.valid, "TTC should be valid when min separation <= conflict_radius"
    assert res_large_radius.value > 0


# ── Fix #2: Multi-timestamp TTC safety analysis ──────────────────────────────
def test_multi_timestamp_ttc():
    from src.safety.ttc import compute_ttc_multi

    # Head-on collision trajectory meeting at t = 2.5s at (250, 0)
    ts = np.linspace(0.0, 5.0, 51)
    pos_a = [[float(x), 0.0] for x in np.linspace(0, 500, 51)]
    pos_b = [[float(x), 0.0] for x in np.linspace(500, 0, 51)]

    ta = _make_dummy_traj(1, pos_a, list(ts))
    tb = _make_dummy_traj(2, pos_b, list(ts))
    cand = CandidateInteraction(
        track_id_a=1, track_id_b=2,
        class_a="car", class_b="car",
        interaction_type="VEH_VEH",
        start_time=0.0, end_time=2.4,  # window before contact
        start_frame=0, end_frame=24,
    )

    best_ttc, peak_t = compute_ttc_multi(ta, tb, cand, n_samples=10, conflict_radius=50.0)
    assert best_ttc.valid
    assert 0.0 < best_ttc.value < 3.0
    assert peak_t > 1.5


# ── Fix #3: EventManager alert accumulation and temporal merging ─────────────
def test_event_manager_temporal_merging():
    from src.events.event_manager import EventManager

    em = EventManager(merge_window=1.5, minimum_duration=0.2)
    cand = CandidateInteraction(
        track_id_a=10, track_id_b=20,
        class_a="car", class_b="car",
        interaction_type="VEH_VEH",
        start_time=0.0, end_time=10.0,
        start_frame=0, end_frame=100,
    )

    ttc_valid = TTCResult(value=1.5, valid=True, model="closest_approach", confidence=0.9)
    pet_inval = PETResult(value=math.inf, valid=False, conflict_point=None, confidence=0.0, applicable=False)
    risk = RiskScore(track_id_a=10, track_id_b=20, ttc_score=0.9, pet_score=0.0, evasive_score=0.0, vulnerability_score=0.5, overall_score=0.85)
    res = ConflictResult(classification=ConflictClass.HIGH_RISK_CONFLICT, severity=0.85, confidence=0.9, reasons=["Low TTC"], metrics={})

    # Cluster 1: alerts at t=1.0, 1.3, 1.8 (within 1.5s window) -> Event 1
    em.record(cand, ttc_valid, pet_inval, [], res, risk, zone_id="Z1", timestamp=1.0)
    em.record(cand, ttc_valid, pet_inval, [], res, risk, zone_id="Z1", timestamp=1.3)
    em.record(cand, ttc_valid, pet_inval, [], res, risk, zone_id="Z1", timestamp=1.8)

    # Cluster 2: alert at t=7.0, 7.5 (gap > 1.5s from t=1.8) -> Event 2
    em.record(cand, ttc_valid, pet_inval, [], res, risk, zone_id="Z1", timestamp=7.0)
    em.record(cand, ttc_valid, pet_inval, [], res, risk, zone_id="Z1", timestamp=7.5)

    events = em.finalize()
    assert len(events) == 2, f"Expected 2 merged events from two separated clusters, got {len(events)}"
    assert events[0].start_time < 3.0 or events[1].start_time < 3.0
    assert any(e.start_time >= 6.5 for e in events)


# ── Fix #4: Evasive action temporal proximity to peak conflict ───────────────
def test_evasive_action_proximity():
    from src.safety.evasive_action import detect_evasive_actions

    # Vehicle brakes abruptly at t = 1.0s (speed drops from 100 to 10)
    # But peak conflict is at t = 8.0s (far away)
    ts = np.linspace(0.0, 10.0, 101)
    pos_a = []
    x = 0.0
    for t in ts:
        speed = 100.0 if t < 1.0 else 10.0
        x += speed * 0.1
        pos_a.append([x, 0.0])
    pos_b = [[1000.0, 0.0] for _ in ts]

    ta = _make_dummy_traj(1, pos_a, list(ts))
    tb = _make_dummy_traj(2, pos_b, list(ts))
    cand = CandidateInteraction(
        track_id_a=1, track_id_b=2,
        class_a="car", class_b="car",
        interaction_type="VEH_VEH",
        start_time=0.0, end_time=10.0,
        start_frame=0, end_frame=100,
    )

    # With peak_conflict_time = 8.0 and evasive_window_s = 2.0: braking at t=1.0 should NOT count
    evs_dist = detect_evasive_actions(
        ta, tb, cand,
        peak_conflict_time=8.0,
        evasive_window_s=2.0,
    )
    assert len(evs_dist) == 0, "Braking far from peak conflict should not count as evasive action"

    # With peak_conflict_time = 1.2s: braking at t=1.0 SHOULD count
    evs_close = detect_evasive_actions(
        ta, tb, cand,
        peak_conflict_time=1.2,
        evasive_window_s=2.0,
    )
    assert len(evs_close) > 0, "Braking near peak conflict should count as evasive action"


# ── Fix #5: PET crossing-geometry validation ──────────────────────────────────
def test_pet_non_crossing_inapplicable():
    from src.safety.pet import compute_pet

    # Parallel vehicles moving side-by-side (same direction, never cross paths)
    ts = np.linspace(0.0, 5.0, 25)
    pos_a = [[float(x), 0.0] for x in np.linspace(0, 100, 25)]
    pos_b = [[float(x), 20.0] for x in np.linspace(0, 100, 25)]

    ta = _make_dummy_traj(1, pos_a, list(ts))
    tb = _make_dummy_traj(2, pos_b, list(ts))
    cand = CandidateInteraction(
        track_id_a=1, track_id_b=2,
        class_a="car", class_b="car",
        interaction_type="VEH_VEH",
        start_time=0.0, end_time=5.0,
        start_frame=0, end_frame=25,
    )

    pet_res = compute_pet(ta, tb, cand)
    assert not pet_res.applicable, "Parallel motion must have applicable=False for PET"
    assert not pet_res.valid
    assert pet_res.inapplicable_reason == "not_crossing_geometry"


# ── Fix #6: Candidate generation sustained approach ──────────────────────────
def test_sustained_approach_check():
    from src.safety.candidate_pairs import _has_sustained_approach

    # Single noisy fluctuation closing then opening immediately
    noisy_dists = np.array([100.0, 100.0, 97.0, 100.0, 101.0, 102.0])
    assert not _has_sustained_approach(noisy_dists), "Single noisy closing step must not pass sustained approach"

    # Sustained approach: >=3 consecutive decreases and >25% fraction
    sustained_dists = np.array([100.0, 95.0, 90.0, 85.0, 80.0, 82.0])
    assert _has_sustained_approach(sustained_dists), "Sustained closing must pass approach check"


# ── Fix #8: Evaluator conflict type compatibility ────────────────────────────
def test_evaluator_conflict_type_compatibility():
    from src.evaluation.evaluator import _conflict_types_compatible, evaluate

    # Identical types match
    assert _conflict_types_compatible("rear_end", "rear_end")
    assert _conflict_types_compatible("crossing", "crossing")

    # Incompatible types do not match
    assert not _conflict_types_compatible("rear_end", "crossing")
    assert not _conflict_types_compatible("head_on", "rear_end")

    # Evaluator test with incompatible type
    pred = ConflictEvent(
        event_id="p1", start_time=1.0, end_time=3.0,
        start_frame=10, end_frame=30,
        track_ids=[1, 2], classes=["car", "car"],
        conflict_type="rear_end", zone_id=None,
        ttc_min=1.0, pet_min=math.inf,
        max_speed_change=0.2, max_heading_change=10.0,
        severity=0.8, confidence=0.9, risk_score=0.8,
        reasons=["Low TTC"],
    )
    gt = GroundTruthEvent(
        start_time=1.0, end_time=3.0,
        conflict_type="crossing",  # Mismatched type!
        severity="HIGH",
    )

    res = evaluate([pred], [gt], video_duration_s=10.0, iou_threshold=0.3)
    assert res.TP == 0, "Mismatched conflict type must not count as TP"
    assert res.FP == 1


# ── Fix #9: Agent37 dynamic config generation ────────────────────────────────
def test_agent37_dynamic_config():
    import pandas as pd
    from scripts.agent37_analyze import generate_next_config

    best_low_prec = {
        "experiment_id": "exp_run1",
        "precision": 0.3,
        "recall": 0.8,
        "safety.ttc_threshold": 3.0,
        "safety.pet_threshold": 2.0,
    }
    df = pd.DataFrame([best_low_prec])
    cfg_text = generate_next_config(best_low_prec, df)

    assert "ttc_threshold: 2.7" in cfg_text or "Precision-constrained" in cfg_text


# ── Fix #7: Motorcycle interaction classification ────────────────────────────
def test_motorcycle_classification():
    from src.safety.candidate_pairs import interaction_type

    # Motorcycle with vehicles -> VEH_MOTO (distinct from VEH_PED and VEH_VEH)
    assert interaction_type("car", "motorcycle") == "VEH_MOTO"
    assert interaction_type("motorcycle", "truck") == "VEH_MOTO"
    assert interaction_type("bus", "motorcycle") == "VEH_MOTO"
    assert interaction_type("motorcycle", "motorcycle") == "VEH_MOTO"

    # Pedestrian/person interactions -> VEH_PED
    assert interaction_type("person", "car") == "VEH_PED"
    assert interaction_type("car", "person") == "VEH_PED"
    assert interaction_type("person", "motorcycle") == "VEH_PED"
    assert interaction_type("bicycle", "car") == "VEH_PED"

    # Vehicle-to-vehicle interactions -> VEH_VEH
    assert interaction_type("car", "car") == "VEH_VEH"
    assert interaction_type("car", "truck") == "VEH_VEH"
    assert interaction_type("bus", "truck") == "VEH_VEH"


# ── Fix #2 & #3: Temporal sampling & EventManager aggregation ────────────────
def test_pipeline_temporal_sampling_integration():
    from src.events.event_manager import EventManager
    from src.schema import ConflictClass, ConflictResult, EvasiveAction, RiskScore

    em = EventManager(merge_window=2.0, minimum_duration=0.2)
    cand = CandidateInteraction(
        track_id_a=101, track_id_b=102,
        class_a="car", class_b="motorcycle",
        interaction_type="VEH_MOTO",
        start_time=1.0, end_time=4.0,
        start_frame=30, end_frame=120,
    )

    # Simulate temporal sampling across interval [1.0, 4.0] with varying TTC
    # Peak conflict happens at t = 2.5s with TTC = 0.8s
    sample_timestamps = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    ttc_vals = [2.5, 1.8, 1.2, 0.8, 1.4, 2.0, 3.0]

    for t, ttc_val in zip(sample_timestamps, ttc_vals):
        ttc = TTCResult(value=ttc_val, valid=True, model="closest_approach", confidence=0.85)
        pet = PETResult(value=1.5, valid=True, conflict_point=(200, 200), confidence=0.8, applicable=True)
        evs = [EvasiveAction(101, t, "BRAKING", -0.45, 0.0)] if t == 2.5 else []
        res = ConflictResult(
            classification=ConflictClass.HIGH_RISK_CONFLICT if ttc_val < 1.0 else ConflictClass.POTENTIAL_CONFLICT,
            severity=0.8 if ttc_val < 1.0 else 0.5,
            confidence=0.85,
            reasons=["ttc_critical"] if ttc_val < 1.0 else [],
            metrics={},
        )
        risk = RiskScore(
            track_id_a=101, track_id_b=102,
            ttc_score=1.0 - (ttc_val / 3.0),
            pet_score=0.25,
            evasive_score=0.3 if evs else 0.0,
            vulnerability_score=1.0,
            overall_score=0.85 if t == 2.5 else 0.45,
        )
        em.record(cand, ttc, pet, evs, res, risk, zone_id="Z3", timestamp=t)

    events = em.finalize()
    assert len(events) == 1, "Temporal alerts within merge_window must merge into single ConflictEvent"
    ev = events[0]

    # Verify all peak/min metrics are retained
    assert ev.conflict_type == "VEH_MOTO"
    assert ev.ttc_min == pytest.approx(0.8, rel=1e-3)
    assert ev.pet_min == pytest.approx(1.5, rel=1e-3)
    assert ev.max_speed_change == pytest.approx(0.45, rel=1e-3)
    assert ev.risk_score == pytest.approx(0.85, rel=1e-3)
    assert ev.peak_conflict_time == pytest.approx(2.5, rel=1e-3)
    assert ev.peak_conflict_frame is not None
    assert ev.peak_conflict_frame == int(2.5 * 30.0)


# ── Fix #10: Nodus/Kaggle fail-fast without CUDA ──────────────────────────────
def test_runner_cuda_fail_fast(monkeypatch):
    import torch
    from scripts.nodus_kaggle_runner import check_cuda_or_fail

    # Explicit allow_cpu passes without error
    check_cuda_or_fail(allow_cpu=True)

    # When CUDA is unavailable, check_cuda_or_fail must raise RuntimeError
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as exc_info:
        check_cuda_or_fail(allow_cpu=False)
    assert "CUDA GPU accelerator is required" in str(exc_info.value)

