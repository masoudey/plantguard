"""Structured logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

_LOG_DIR = Path("logs")
_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(sys.stderr, level="INFO")
    logger.add(
        _LOG_DIR / "plantguard.log",
        rotation="10 MB",
        retention="10 days",
        level="INFO",
    )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    _configure()
    return logger.bind(module=name or "plantguard")
