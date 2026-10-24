"""
Phase 20 — Conflict Evaluation.

Computes TP/FP/FN, Precision, Recall, F1 when ground-truth labels are available.
Keeps evaluation separate from the inference pipeline.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from src.logger import get_logger
from src.schema import ConflictEvent

log = get_logger(__name__)


@dataclass
class GroundTruthEvent:
    """Minimal ground-truth annotation for one conflict event."""
    start_time: float
    end_time: float
    conflict_type: str      # "VEH_PED" | "VEH_VEH"
    severity: str           # "HIGH" | "MEDIUM" | "LOW"
    source: str = ""        # annotation file / annotator ID


@dataclass
class EvaluationResult:
    TP: int
    FP: int
    FN: int
    precision: float
    recall: float
    f1: float
    false_alerts_per_minute: float
    events_detected: int
    events_confirmed: int
    video_duration_s: float


def evaluate(
    predictions: List[ConflictEvent],
    ground_truth: List[GroundTruthEvent],
    video_duration_s: float,
    iou_threshold: float = 0.3,   # temporal IoU for matching
    severity_filter: Optional[str] = None,
) -> EvaluationResult:
    """
    Match predictions to ground truth using temporal IoU.

    Args:
        predictions:       Pipeline ConflictEvent list.
        ground_truth:      GroundTruthEvent list from dataset.
        video_duration_s:  Total video duration for false-alert rate.
        iou_threshold:     Minimum temporal IoU to count as TP.
        severity_filter:   If set, only evaluate gt events of this severity.
    """
    if severity_filter:
        ground_truth = [g for g in ground_truth if g.severity.upper() == severity_filter.upper()]

    matched_gt:   set = set()
    matched_pred: set = set()

    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):
            if gi in matched_gt:
                continue
            # FIX #8 — require compatible conflict type in addition to temporal IoU
            if not _conflict_types_compatible(pred.conflict_type, gt.conflict_type):
                continue
            tiou = _temporal_iou(pred.start_time, pred.end_time,
                                  gt.start_time, gt.end_time)
            if tiou >= iou_threshold:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break


    TP = len(matched_gt)
    FP = len(predictions) - len(matched_pred)
    FN = len(ground_truth) - len(matched_gt)

    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    fa_rate   = FP / max(video_duration_s / 60.0, 1e-6)

    result = EvaluationResult(
        TP=TP, FP=FP, FN=FN,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_alerts_per_minute=round(fa_rate, 4),
        events_detected=len(predictions),
        events_confirmed=TP,
        video_duration_s=video_duration_s,
    )
    log.info(
        "Evaluation | TP=%d FP=%d FN=%d | P=%.3f R=%.3f F1=%.3f | FA/min=%.2f",
        TP, FP, FN, precision, recall, f1, fa_rate,
    )
    return result


def save_evaluation(result: EvaluationResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(asdict(result), fh, indent=2)
    log.info("Evaluation saved: %s", path)


def _temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / max(union, 1e-9)


def _conflict_types_compatible(pred_type: str, gt_type: str) -> bool:
    """
    FIX #8 — A VEH_PED prediction should not count as a TP for a VEH_VEH
    ground-truth event, and vice versa.

    Matching rules:
      - Exact match always compatible.
      - Unknown/empty type in either prediction or GT is treated as wildcard
        (to handle annotations that lack type labels).
    """
    if not pred_type or not gt_type:
        return True          # missing label → no penalty
    return pred_type.upper() == gt_type.upper()

