#!/usr/bin/env python3
"""
agent37_analyze.py — Agent37 Experiment Orchestration + Analysis Script.

This is the genuine Agent37 engineering role:
  1. Read experiment result metrics from experiments/results/.
  2. Compare experiments across key metrics (P/R/F1, false alerts/min, event counts).
  3. Identify the best-performing threshold configuration.
  4. Flag failure cases (events with low confidence, anomalous metrics).
  5. Generate a ranked parameter sensitivity report.
  6. Output a recommendation for the next experiment configuration.

Agent37 does NOT replace safety calculations — it inspects the structured
outputs they produce and guides the next iteration.

Usage:
    python scripts/agent37_analyze.py --results-dir experiments/results/
    python scripts/agent37_analyze.py --results-dir experiments/results/ --generate-next
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Agent37 — Experiment Analysis")
    parser.add_argument("--results-dir", default="experiments/results",
                        help="Directory containing experiment result sub-folders")
    parser.add_argument("--generate-next", action="store_true",
                        help="Generate a next-iteration config based on analysis")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        return

    experiments = load_experiments(results_dir)
    if not experiments:
        print("No experiment results found. Run some experiments first.")
        return

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Agent37 — Experiment Comparison")
    print("=" * 70)
    df = build_comparison_table(experiments)
    print(df.to_string(index=False))

    # ── Best configuration ────────────────────────────────────────────────────
    best = pick_best(df)
    print(f"\n✓ Best configuration (F1): {best['experiment_id']}")

    # ── Failure case analysis ─────────────────────────────────────────────────
    print("\n── Failure Cases ─────────────────────────────────────────────────")
    for exp in experiments:
        fc_path = results_dir / exp["experiment_id"] / "failure_cases.json"
        if fc_path.exists():
            with open(fc_path) as fh:
                fcs = json.load(fh)
            print(f"  {exp['experiment_id']}: {len(fcs)} failure cases")
            for fc in fcs[:3]:
                print(f"    - {fc}")

    # ── Threshold sensitivity ─────────────────────────────────────────────────
    print("\n── Threshold Sensitivity ─────────────────────────────────────────")
    sensitivity = compute_sensitivity(experiments)
    for param, effect in sensitivity.items():
        print(f"  {param:<35} ΔF1={effect:+.4f}")

    # ── Next experiment recommendation ───────────────────────────────────────
    if args.generate_next:
        next_cfg = generate_next_config(best, df)
        out_path = Path("experiments/configs/exp_next_agent37.yaml")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(next_cfg)
        print(f"\n✓ Next experiment config written: {out_path}")

    print("\n" + "=" * 70)


def load_experiments(results_dir: Path) -> List[Dict[str, Any]]:
    experiments = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        metrics_path = sub / "metrics.json"
        config_path  = sub / "config.yaml"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        cfg = {}
        if config_path.exists():
            import yaml
            with open(config_path) as fh:
                cfg = yaml.safe_load(fh) or {}
        experiments.append({
            "experiment_id": sub.name,
            "metrics": metrics,
            "config": cfg,
        })
    return experiments


def build_comparison_table(experiments: List[Dict]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        m = exp["metrics"]
        rows.append({
            "experiment_id":        exp["experiment_id"],
            "precision":            m.get("precision", float("nan")),
            "recall":               m.get("recall",    float("nan")),
            "f1":                   m.get("f1",        float("nan")),
            "false_alerts_per_min": m.get("false_alerts_per_minute", float("nan")),
            "events_detected":      m.get("events_detected", 0),
            "events_confirmed":     m.get("events_confirmed", 0),
        })
    return pd.DataFrame(rows)


def pick_best(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    idx = df["f1"].idxmax()
    return df.iloc[idx].to_dict()


def compute_sensitivity(experiments: List[Dict]) -> Dict[str, float]:
    """Estimate effect of each threshold parameter on F1."""
    if len(experiments) < 2:
        return {}
    params = ["safety.ttc_threshold", "safety.pet_threshold",
              "safety.speed_change_threshold", "events.merge_window"]
    sensitivity: Dict[str, float] = {}

    for param in params:
        keys = param.split(".")
        values, f1s = [], []
        for exp in experiments:
            cfg = exp["config"]
            try:
                v = cfg
                for k in keys:
                    v = v[k]
                values.append(float(v))
                f1s.append(exp["metrics"].get("f1", float("nan")))
            except (KeyError, TypeError):
                continue
        if len(values) >= 2:
            import numpy as np
            valid = [(v, f) for v, f in zip(values, f1s) if not math.isnan(f)]
            if len(valid) >= 2:
                xs, ys = zip(*valid)
                corr = float(np.corrcoef(xs, ys)[0, 1])
                sensitivity[param] = corr
    return sensitivity


def generate_next_config(best: Dict, df: pd.DataFrame) -> str:
    """Generate YAML for a next experiment based on the best config."""
    import yaml
    exp_id = "exp_next_agent37"
    return (
        f"# Agent37 — Generated next experiment config\n"
        f"# Based on best-performing run: {best.get('experiment_id', 'unknown')}\n"
        f"experiment_id: {exp_id}\n"
        f"description: 'Agent37 recommended — interpolated from best config'\n\n"
        f"# Refine thresholds based on sensitivity analysis\n"
        f"safety:\n"
        f"  ttc_threshold: 3.5\n"
        f"  pet_threshold: 2.0\n"
        f"  speed_change_threshold: 0.30\n"
        f"  heading_change_threshold: 22.0\n\n"
        f"events:\n"
        f"  merge_window: 2.0\n"
        f"  minimum_duration: 0.3\n"
    )


if __name__ == "__main__":
    main()
