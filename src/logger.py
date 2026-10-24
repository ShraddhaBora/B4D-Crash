"""
Centralised logging setup for Before the Crash.

Usage:
    from src.logger import get_logger
    log = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Return a configured logger. Safe to call multiple times for the same name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_FORMATTER)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(_FORMATTER)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def configure_root_logger(cfg: dict) -> None:
    """Configure the root logger from config dict (logging section)."""
    level = cfg.get("logging", {}).get("level", "INFO")
    log_file = cfg.get("logging", {}).get("log_file", None)
    get_logger("btc", level=level, log_file=log_file)
