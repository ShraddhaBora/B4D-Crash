#!/usr/bin/env python3
"""
run_pipeline.py — CLI entry point for the Before the Crash pipeline.

Usage:
    python scripts/run_pipeline.py --video data/samples/clip.mp4
    python scripts/run_pipeline.py --video data/samples/clip.mp4 \\
        --config config.yaml --experiment experiments/configs/exp_001.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path (supports running from scripts/ or from root)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Before the Crash — Traffic Conflict Pipeline")
    parser.add_argument("--video",      required=True,           help="Path to input video file")
    parser.add_argument("--config",     default="config.yaml",   help="Base config path")
    parser.add_argument("--experiment", default=None,            help="Experiment overlay config path")
    parser.add_argument("--output-dir", default=None,            help="Override output directory")
    args = parser.parse_args()

    from src.pipeline import Pipeline
    pipe = Pipeline(config_path=args.config, overlay_path=args.experiment)

    if args.output_dir:
        pipe.cfg["video"]["output_dir"] = args.output_dir

    result = pipe.run(args.video)

    print("\n── Pipeline Summary ──────────────────────────────────")
    for k, v in result.items():
        if not isinstance(v, list):
            print(f"  {k:<28} {v}")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
