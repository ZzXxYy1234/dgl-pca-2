"""Logging helpers."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "dgl_pca", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stream logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"))
    logger.addHandler(handler)
    return logger

