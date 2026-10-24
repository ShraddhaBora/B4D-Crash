"""
Unit tests for the Before the Crash pipeline.

Tests deterministic components: TTC, PET, event merging, risk scoring,
trajectory smoothing, and conflict classification.

Run with:
    python -m pytest tests/ -v
    (from the before-the-crash/ directory)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_trajectory(
    track_id: int,
    class_name: str,
    positions: list,
    t_start: float = 0.0,
    fps: float = 10.0,
) -> "Trajectory":
    from src.schema import Trajectory, TrackQuality
    from src.trajectory.smoothing import smooth_positions
    from src.trajectory.motion import compute_motion

    n = len(positions)
    timestamps = np.array([t_start + i / fps for i in range(n)])
    raw_pos = np.array(positions, dtype=float)
    smooth  = smooth_positions(raw_pos, window=3)
    vels, speeds, headings, accels = compute_motion(smooth, timestamps)
    return Trajectory(
        track_id=track_id,
        class_name=class_name,
        quality=TrackQuality.GOOD,
        timestamps=timestamps,
        raw_positions=raw_pos,
        positions=smooth,
        velocities=vels,
        speeds=speeds,
        headings=headings,
        accelerations=accels,
    )


def _make_candidate(ta, tb, interaction_type="VEH_PED"):
    from src.schema import CandidateInteraction
    return CandidateInteraction(
        track_id_a=ta.track_id,
        track_id_b=tb.track_id,
        class_a=ta.class_name,
        class_b=tb.class_name,
        interaction_type=interaction_type,
        start_time=max(ta.timestamps[0], tb.timestamps[0]),
        end_time=min(ta.timestamps[-1], tb.timestamps[-1]),
        start_frame=0,
        end_frame=len(ta.timestamps) - 1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TTC Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTTC:
    def test_head_on_approach_gives_finite_ttc(self):
        """Two objects approaching each other head-on should give finite TTC."""
        from src.safety.ttc import compute_ttc

        # Object A at x=0 moving right; Object B at x=100 moving left
        ta = _make_trajectory(1, "car",    [[i * 10, 50] for i in range(20)])
        tb = _make_trajectory(2, "car",    [[200 - i * 10, 50] for i in range(20)])
        cand = _make_candidate(ta, tb, "VEH_VEH")
        result = compute_ttc(ta, tb, cand, t_query=0.5)

        assert result.valid, "TTC should be valid for converging trajectories"
        assert result.value > 0, "TTC must be positive"
        assert result.value < 100, "TTC should be reasonably small"

    def test_diverging_objects_give_infinite_ttc(self):
        """Two objects moving apart should give TTC=inf (not approaching)."""
        from src.safety.ttc import compute_ttc

        ta = _make_trajectory(1, "car", [[i * 5, 50] for i in range(20)])
        tb = _make_trajectory(2, "car", [[-(i * 5), 50] for i in range(20)])
        cand = _make_candidate(ta, tb, "VEH_VEH")
        result = compute_ttc(ta, tb, cand, t_query=0.5)

        assert result.value == math.inf or not result.valid, \
            "Diverging objects should have TTC=inf"

    def test_stationary_objects_give_undefined_ttc(self):
        """If both objects are stationary, closing speed = 0, TTC = undefined."""
        from src.safety.ttc import compute_ttc

        ta = _make_trajectory(1, "person", [[50, 50] for _ in range(20)])
        tb = _make_trajectory(2, "car",    [[80, 50] for _ in range(20)])
        cand = _make_candidate(ta, tb)
        result = compute_ttc(ta, tb, cand, t_query=0.5)

        assert not result.valid or result.value == math.inf, \
            "Stationary objects should not produce valid TTC"


# ──────────────────────────────────────────────────────────────────────────────
# PET Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPET:
    def test_crossing_trajectories_give_finite_pet(self):
        """Trajectories that cross a shared region should produce finite PET."""
        from src.safety.pet import compute_pet

        # A crosses horizontally, B crosses vertically; they share region around (100,100)
        ta = _make_trajectory(1, "car",    [[i * 10, 100] for i in range(20)])
        tb = _make_trajectory(2, "person", [[100, i * 10] for i in range(20)], t_start=0.5)
        cand = _make_candidate(ta, tb)
        result = compute_pet(ta, tb, cand, conflict_radius=30.0)

        assert result.valid, "Crossing trajectories should produce valid PET"
        assert result.value >= 0, "PET must be non-negative"

    def test_parallel_paths_give_undefined_pet(self):
        """Parallel paths that never share a conflict region — PET undefined."""
        from src.safety.pet import compute_pet

        ta = _make_trajectory(1, "car", [[i * 5, 10] for i in range(20)])
        tb = _make_trajectory(2, "car", [[i * 5, 500] for i in range(20)])
        cand = _make_candidate(ta, tb, "VEH_VEH")
        result = compute_pet(ta, tb, cand, conflict_radius=20.0)

        assert not result.valid or result.value == math.inf, \
            "Parallel paths far apart should not produce valid PET"


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory Smoothing Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSmoothing:
    def test_smoothing_reduces_noise(self):
        """Smoothed positions should have lower variance than noisy raw positions."""
        from src.trajectory.smoothing import smooth_positions

        rng = np.random.default_rng(42)
        base = np.linspace(0, 100, 50)
        noisy = np.stack([base + rng.normal(0, 5, 50),
                          base + rng.normal(0, 5, 50)], axis=1)
        smoothed = smooth_positions(noisy, window=7)
        assert smoothed.shape == noisy.shape
        # Variance of smoothed should be lower
        assert np.var(smoothed) < np.var(noisy)

    def test_smoothing_handles_short_track(self):
        """Smoothing should not crash on a 3-point track."""
        from src.trajectory.smoothing import smooth_positions

        pos = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        result = smooth_positions(pos, window=5)
        assert result.shape == pos.shape


# ──────────────────────────────────────────────────────────────────────────────
# Conflict Classifier Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestConflictClassifier:
    def _dummy_candidate(self):
        from src.schema import CandidateInteraction
        return CandidateInteraction(
            track_id_a=1, track_id_b=2,
            class_a="car", class_b="person",
            interaction_type="VEH_PED",
            start_time=0.0, end_time=3.0,
            start_frame=0, end_frame=90,
        )

    def test_high_risk_when_ttc_critical_and_evasive(self):
        from src.schema import TTCResult, PETResult, EvasiveAction, ConflictClass
        from src.safety.conflict_classifier import classify_conflict

        cand = self._dummy_candidate()
        ttc = TTCResult(value=0.8, valid=True, model="closest_approach", confidence=0.9)
        pet = PETResult(value=0.5, valid=True, conflict_point=(100, 100), confidence=0.8)
        evs = [EvasiveAction(1, 1.0, "BRAKING", -0.6, 0.0)]
        result = classify_conflict(cand, ttc, pet, evs, ttc_threshold=3.0, pet_threshold=2.0)

        assert result.classification == ConflictClass.HIGH_RISK_CONFLICT
        assert "ttc_critical" in result.reasons
        assert "evasive_braking" in result.reasons

    def test_normal_when_no_critical_indicators(self):
        from src.schema import TTCResult, PETResult, ConflictClass
        from src.safety.conflict_classifier import classify_conflict

        cand = self._dummy_candidate()
        ttc = TTCResult(value=math.inf, valid=False, model="none", confidence=0.0)
        pet = PETResult(value=math.inf, valid=False, conflict_point=None, confidence=0.0)
        result = classify_conflict(cand, ttc, pet, [], ttc_threshold=3.0, pet_threshold=2.0)

        assert result.classification == ConflictClass.NORMAL
        assert result.severity == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Risk Score Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRiskScore:
    def _dummy_candidate(self, class_a="car", class_b="person"):
        from src.schema import CandidateInteraction
        return CandidateInteraction(
            track_id_a=1, track_id_b=2,
            class_a=class_a, class_b=class_b,
            interaction_type="VEH_PED",
            start_time=0.0, end_time=3.0,
            start_frame=0, end_frame=90,
        )

    def test_higher_danger_gives_higher_score(self):
        from src.schema import TTCResult, PETResult, EvasiveAction
        from src.safety.risk_score import compute_risk_score

        cand = self._dummy_candidate()
        # Dangerous: short TTC, short PET, evasive action
        ttc_d = TTCResult(0.5, True, "closest_approach", 0.9)
        pet_d = PETResult(0.3, True, (100, 100), 0.9)
        evs_d = [EvasiveAction(1, 1.0, "BRAKING", -0.8, 0.0),
                 EvasiveAction(1, 1.1, "SWERVE",   0.0, 35.0)]
        risk_d = compute_risk_score(cand, ttc_d, pet_d, evs_d)

        # Safe: large TTC, large PET, no evasion
        ttc_s = TTCResult(math.inf, False, "none", 0.0)
        pet_s = PETResult(math.inf, False, None, 0.0)
        risk_s = compute_risk_score(cand, ttc_s, pet_s, [])

        assert risk_d.overall_score > risk_s.overall_score, \
            "Dangerous scenario must score higher than safe one"

    def test_invalid_metrics_dont_fabricate_high_score(self):
        from src.schema import TTCResult, PETResult
        from src.safety.risk_score import compute_risk_score

        cand = self._dummy_candidate(class_a="car", class_b="car")
        ttc = TTCResult(math.inf, False, "none", 0.0)
        pet = PETResult(math.inf, False, None, 0.0)
        risk = compute_risk_score(cand, ttc, pet, [])

        # Without VRU and without valid metrics, score should stay modest
        assert risk.overall_score < 0.5, \
            "No valid metrics + no VRU should not produce high risk score"


# ──────────────────────────────────────────────────────────────────────────────
# Event Manager Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEventManager:
    def _make_conflict_result(self, cls, severity=0.8):
        from src.schema import ConflictResult
        return ConflictResult(
            classification=cls, severity=severity, confidence=0.85,
            reasons=["ttc_critical"], metrics={},
        )

    def test_single_candidate_becomes_one_event(self):
        from src.schema import (CandidateInteraction, TTCResult, PETResult,
                                 RiskScore, ConflictClass)
        from src.events.event_manager import EventManager

        mgr = EventManager(merge_window=2.0, minimum_duration=0.1)
        cand = CandidateInteraction(1, 2, "car", "person", "VEH_PED",
                                    0.0, 2.0, 0, 60)
        ttc = TTCResult(1.0, True, "closest_approach", 0.8)
        pet = PETResult(0.5, True, (100, 100), 0.8)
        res = self._make_conflict_result(ConflictClass.POTENTIAL_CONFLICT)
        risk = RiskScore(1, 2, 0.7, 0.6, 0.3, 1.0, 0.65)
        mgr.record(cand, ttc, pet, [], res, risk, zone_id="R2C2")
        events = mgr.finalize()
        assert len(events) == 1

    def test_independent_pairs_give_separate_events(self):
        from src.schema import (CandidateInteraction, TTCResult, PETResult,
                                 RiskScore, ConflictClass)
        from src.events.event_manager import EventManager

        mgr = EventManager(merge_window=2.0, minimum_duration=0.1)
        for pair_id, (tid_a, tid_b) in enumerate([(1, 2), (3, 4)]):
            cand = CandidateInteraction(tid_a, tid_b, "car", "person",
                                        "VEH_PED", 0.0, 2.0, 0, 60)
            ttc  = TTCResult(1.0, True, "closest_approach", 0.8)
            pet  = PETResult(0.5, True, (100, 100), 0.8)
            res  = self._make_conflict_result(ConflictClass.POTENTIAL_CONFLICT)
            risk = RiskScore(tid_a, tid_b, 0.7, 0.6, 0.3, 1.0, 0.65)
            mgr.record(cand, ttc, pet, [], res, risk)

        events = mgr.finalize()
        assert len(events) == 2, "Independent pairs should produce separate events"


# ──────────────────────────────────────────────────────────────────────────────
# Zone Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestZones:
    def test_point_in_correct_grid_cell(self):
        from src.geometry.zones import make_grid_zones

        zm = make_grid_zones(frame_width=500, frame_height=500, grid_size=5)
        # Point (50, 50) should land in row=0, col=0 (each cell is 100x100)
        zone = zm.find_zone(50, 50)
        assert zone is not None
        assert zone.id == "R0C0"

    def test_all_pixels_assigned_to_a_zone(self):
        from src.geometry.zones import make_grid_zones

        zm = make_grid_zones(frame_width=640, frame_height=480, grid_size=4)
        for x, y in [(0, 0), (639, 479), (320, 240), (100, 100)]:
            z = zm.find_zone(x, y)
            assert z is not None, f"Point ({x},{y}) should be in a zone"
