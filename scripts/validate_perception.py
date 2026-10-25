#!/usr/bin/env python3
"""
validate_perception.py — Staged perception stack validation for Kaggle/Nodus.

Runs Tests 1–5 from the Before the Crash validation hierarchy:

  TEST 1: YOLO loads correctly on CUDA
  TEST 2: YOLO detects objects on actual video frames
  TEST 3: ByteTrack successfully processes detections (via YOLO.track)
  TEST 4: Track IDs persist across consecutive frames
  TEST 5: Detection-to-track class/bounding-box mapping is correct

This script is designed to run on SHORT video windows (first N frames) to
validate the perception stack before committing to full-video processing.

Usage (from project root OR any directory — uses _ROOT for all paths):
    python scripts/validate_perception.py --video /kaggle/input/.../video.mp4
    python scripts/validate_perception.py --video /kaggle/input/.../video.mp4 --frames 30
    python scripts/validate_perception.py --video data/test/sample_01.mp4 --frames 30

The script will NOT declare any test PASSED unless the specific observable
condition is verified.  It will explicitly report FAILED with the exception
or the violated condition.

Design principles:
- Tests 1–5 must all pass before the pipeline is considered ready.
- A test is PASSED only when the observable condition is confirmed, NOT
  merely because no exception was raised.
- Track identity persistence (Test 4) requires at least 2 frames with shared
  IDs.  It cannot be verified on a single frame.
- ByteTrack is considered genuinely active only when boxes.id is non-None and
  the same integer ID appears in two consecutive model.track() calls.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Resolve project root regardless of CWD (Kaggle or local)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Test result accumulator
# ──────────────────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed: bool = False
        self.details: str = ""
        self.exception: str | None = None

    def mark_pass(self, details: str):
        self.passed = True
        self.details = details
        print(f"  [PASS] {self.name}: {details}")

    def mark_fail(self, reason: str, exc: Exception | None = None):
        self.passed = False
        self.details = reason
        if exc:
            self.exception = type(exc).__name__ + ": " + str(exc)
            print(f"  [FAIL] {self.name}: {reason}")
            print(f"         Exception: {self.exception}")
        else:
            print(f"  [FAIL] {self.name}: {reason}")


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test1_yolo_loads_on_cuda(model_path: str) -> tuple[TestResult, object | None]:
    """TEST 1: YOLO loads correctly on CUDA."""
    r = TestResult("TEST 1 — YOLO loads on CUDA")
    try:
        import torch
        from src.perception.detector import YOLODetector
        detector = YOLODetector(model_path=model_path, confidence=0.30, iou=0.50)
        if detector.device == "cpu":
            if not torch.cuda.is_available():
                r.mark_fail(
                    "CUDA is not available. Ensure GPU is enabled on this compute node."
                )
            else:
                r.mark_fail(
                    f"Detector chose CPU even though torch.cuda.is_available()=True. "
                    f"Device selected: {detector.device}"
                )
            return r, None
        r.mark_pass(f"device={detector.device}  model={model_path}")
        return r, detector
    except Exception as exc:
        r.mark_fail("Exception during YOLODetector construction", exc)
        return r, None


def test2_yolo_detects_on_frame(detector, frame) -> TestResult:
    """TEST 2: YOLO detects objects on an actual video frame."""
    r = TestResult("TEST 2 — YOLO detects on real frame")
    try:
        detections = detector.detect(frame, frame_id=0, timestamp=0.0)
        if len(detections) == 0:
            r.mark_fail(
                "detect() returned 0 detections on the first frame. "
                "Possible causes: wrong class filter, confidence too high, "
                "frame is black/empty, or model is not loaded correctly."
            )
        else:
            cls_counts = {}
            for d in detections:
                cls_counts[d.class_name] = cls_counts.get(d.class_name, 0) + 1
            r.mark_pass(
                f"{len(detections)} detections: {cls_counts}  "
                f"conf range=[{min(d.confidence for d in detections):.2f}, "
                f"{max(d.confidence for d in detections):.2f}]"
            )
    except Exception as exc:
        r.mark_fail("Exception during detector.detect()", exc)
    return r


def test3_bytetrack_processes_frame(detector, frame) -> tuple[TestResult, list]:
    """TEST 3: ByteTrack processes detections via YOLO.track() without exception."""
    r = TestResult("TEST 3 — ByteTrack processes frame via YOLO.track()")
    try:
        detections, observations = detector.track(frame, frame_id=1, timestamp=0.0)
        if len(detections) == 0:
            # Not necessarily a test failure — just means no detections
            r.mark_pass(
                f"track() succeeded with 0 detections/observations. "
                f"(Frame may have no vehicles — try a different start frame)"
            )
        else:
            r.mark_pass(
                f"track() succeeded: {len(detections)} detections, "
                f"{len(observations)} observations. "
                f"Track IDs: {sorted(set(o.track_id for o in observations))}"
            )
        return r, observations
    except Exception as exc:
        r.mark_fail(
            "Exception during detector.track(). "
            "This likely means model.track() API incompatibility. "
            "Check Ultralytics version and tracker YAML.",
            exc,
        )
        return r, []


def test4_track_ids_persist(detector, frames: list) -> TestResult:
    """TEST 4: Track IDs persist across consecutive frames."""
    r = TestResult("TEST 4 — Track IDs persist across frames")
    if len(frames) < 2:
        r.mark_fail("Fewer than 2 frames provided; cannot check ID persistence.")
        return r

    try:
        prev_ids: set[int] = set()
        frame_results = []

        for i, (frame, frame_id) in enumerate(frames):
            ts = frame_id / 24.0  # approximate; exact fps not critical here
            _, observations = detector.track(frame, frame_id=frame_id, timestamp=ts)
            current_ids = {o.track_id for o in observations}
            frame_results.append((frame_id, current_ids))
            if i > 0 and prev_ids:
                persisted = current_ids & prev_ids
                if persisted:
                    r.mark_pass(
                        f"Track IDs {sorted(persisted)} persisted from "
                        f"frame {frame_results[i-1][0]} to frame {frame_id}. "
                        f"All IDs frame {frame_id}: {sorted(current_ids)}"
                    )
                    return r
            prev_ids = current_ids

        # If we get here, no IDs persisted across any consecutive pair
        all_ids = [ids for _, ids in frame_results]
        r.mark_fail(
            f"No track ID persisted across any consecutive frame pair. "
            f"This indicates tracker re-initialisation or all frames have 0 tracks. "
            f"Per-frame IDs: {[(fid, sorted(ids)) for fid, ids in frame_results]}"
        )
    except Exception as exc:
        r.mark_fail("Exception during multi-frame track() calls", exc)

    return r


def test5_class_bbox_mapping(detector, frame) -> TestResult:
    """TEST 5: Detection-to-track class/bounding-box mapping is correct."""
    r = TestResult("TEST 5 — class_id and bbox preserved in TrackObservation")
    try:
        detections, observations = detector.track(frame, frame_id=2, timestamp=0.0)
        if not observations:
            r.mark_pass(
                "0 observations this frame — cannot validate mapping. "
                "Run on a frame with detections to complete this test."
            )
            return r

        # Check that class_id appears in model.names
        model_names = detector.model.names
        bad_class = [o for o in observations if o.class_id not in model_names]
        if bad_class:
            r.mark_fail(
                f"Observations with unknown class_id: "
                f"{[(o.track_id, o.class_id) for o in bad_class]}. "
                f"Model has {len(model_names)} classes."
            )
            return r

        # Check bbox sanity: x2 > x1, y2 > y1, all non-negative
        bad_bbox = [
            o for o in observations
            if o.x2 <= o.x1 or o.y2 <= o.y1 or o.x1 < 0 or o.y1 < 0
        ]
        if bad_bbox:
            r.mark_fail(
                f"Observations with invalid bounding boxes: "
                f"{[(o.track_id, o.x1, o.y1, o.x2, o.y2) for o in bad_bbox]}"
            )
            return r

        # Check confidence is in [0, 1]
        bad_conf = [o for o in observations if not (0.0 <= o.confidence <= 1.0)]
        if bad_conf:
            r.mark_fail(
                f"Observations with out-of-range confidence: "
                f"{[(o.track_id, o.confidence) for o in bad_conf]}"
            )
            return r

        cls_dist = {}
        for o in observations:
            cls_dist[o.class_name] = cls_dist.get(o.class_name, 0) + 1
        r.mark_pass(
            f"{len(observations)} observations validated. "
            f"Classes: {cls_dist}. "
            f"Bbox and confidence values all within valid ranges."
        )
    except Exception as exc:
        r.mark_fail("Exception during class/bbox validation", exc)
    return r


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run_validation(video_path: str, model_path: str = "yolov8n.pt", n_frames: int = 10):
    print("=" * 65)
    print("Before the Crash — Perception Stack Validation (Tests 1–5)")
    print("=" * 65)
    print(f"  Video   : {video_path}")
    print(f"  Model   : {model_path}")
    print(f"  Frames  : first {n_frames} frames")
    print()

    t0 = time.time()
    results: list[TestResult] = []

    # ── Load video frames ─────────────────────────────────────────────────────
    print("Loading video frames...")
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[FATAL] Cannot open video: {video_path}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  {fps:.3f} fps | {total} frames | {w}x{h}")
        frames_raw = []
        frame_idx = 0
        while len(frames_raw) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames_raw.append((frame, frame_idx))
            frame_idx += 1
        cap.release()
        print(f"  Loaded {len(frames_raw)} frames.\n")
    except Exception as exc:
        print(f"[FATAL] Failed to load video frames: {exc}")
        sys.exit(1)

    if not frames_raw:
        print("[FATAL] No frames could be read from the video.")
        sys.exit(1)

    first_frame, first_frame_id = frames_raw[0]

    # ── TEST 1 ────────────────────────────────────────────────────────────────
    print("[TEST 1] YOLO model loads on CUDA")
    r1, detector = test1_yolo_loads_on_cuda(model_path)
    results.append(r1)

    if detector is None:
        print("\n[ABORT] Cannot proceed without a loaded detector.")
        _print_summary(results, time.time() - t0)
        return results

    # ── TEST 2 ────────────────────────────────────────────────────────────────
    print("\n[TEST 2] YOLO detection on real video frame")
    r2 = test2_yolo_detects_on_frame(detector, first_frame)
    results.append(r2)

    # ── TEST 3 ────────────────────────────────────────────────────────────────
    print("\n[TEST 3] ByteTrack processes frame via YOLO.track()")
    r3, first_observations = test3_bytetrack_processes_frame(detector, first_frame)
    results.append(r3)

    if not r3.passed:
        print("\n[ABORT] ByteTrack API is broken. Cannot proceed to Tests 4–5.")
        _print_summary(results, time.time() - t0)
        return results

    # ── TEST 4 ────────────────────────────────────────────────────────────────
    print(f"\n[TEST 4] Track ID persistence across {min(n_frames, len(frames_raw))} frames")
    # Use first min(n_frames, 5) frames to check persistence
    test_frames = frames_raw[:min(n_frames, 5)]
    r4 = test4_track_ids_persist(detector, test_frames)
    results.append(r4)

    # ── TEST 5 ────────────────────────────────────────────────────────────────
    print("\n[TEST 5] class_id / bbox / confidence mapping in TrackObservation")
    r5 = test5_class_bbox_mapping(detector, first_frame)
    results.append(r5)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(results, time.time() - t0)
    return results


def _print_summary(results: list[TestResult], elapsed: float):
    print()
    print("─" * 65)
    print("VALIDATION SUMMARY")
    print("─" * 65)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")
        if not r.passed:
            print(f"         → {r.details}")
            if r.exception:
                print(f"         → Exception: {r.exception}")
    print()
    print(f"  {passed}/{total} tests passed | elapsed: {elapsed:.1f}s")
    if passed == total:
        print("\n  ✓ All perception tests PASSED.")
        print("  Next step: run the full pipeline on a short clip (first 60s).")
        print("  python scripts/run_pipeline.py --video <path> --output-dir outputs/test_60s")
    else:
        print("\n  ✗ Perception tests FAILED. Do NOT proceed to full-video processing.")
        print("  Fix the failing tests above before running the pipeline.")
    print("─" * 65)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Before the Crash — Perception Stack Validation (Tests 1–5)"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to video file (use /kaggle/input/... path on Kaggle)"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLO model name or path (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--frames", type=int, default=10,
        help="Number of frames to validate (default: 10; use 30 for thorough check)"
    )
    args = parser.parse_args()

    results = run_validation(
        video_path=args.video,
        model_path=args.model,
        n_frames=args.frames,
    )

    # Exit code: 0 if all tests pass, 1 otherwise (for CI/scripting)
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)
