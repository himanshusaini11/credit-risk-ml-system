"""Logging helpers for the service and pipelines."""

from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging defaults."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with optional name."""
    return logging.getLogger(name)
