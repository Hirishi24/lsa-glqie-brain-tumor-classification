"""Logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure console and file logging."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lsa_glqie")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / "experiment.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

