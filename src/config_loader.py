"""
Config loader — reads config.yaml and optional experiment overlay.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(base_path: str = "config.yaml", overlay_path: str | None = None) -> dict:
    """
    Load the base config and optionally merge an experiment overlay on top.
    Overlay values take precedence; nested dicts are deep-merged.
    """
    base = _read_yaml(base_path)
    if overlay_path:
        overlay = _read_yaml(overlay_path)
        base = _deep_merge(base, overlay)
    return base


def _read_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def ensure_output_dirs(cfg: dict) -> None:
    """Create all output directories referenced in config."""
    root = Path(cfg.get("video", {}).get("output_dir", "outputs"))
    for sub in ("detections", "tracks", "trajectories", "events", "clips", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    Path("experiments/logs").mkdir(parents=True, exist_ok=True)
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
