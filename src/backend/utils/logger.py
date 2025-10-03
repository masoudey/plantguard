"""Structured logging configuration."""
from __future__ import annotations

import logging
from typing import Optional

from loguru import logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger.remove()
    logger.add(
        "logs/plantguard.log",
        rotation="10 MB",
        retention="10 days",
        level="INFO",
    )
    return logger.bind(module=name or "plantguard")
