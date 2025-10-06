"""Utility collection for PlantGuard backend with lazy imports."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = ["logger", "metrics"]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return import_module(f"{__name__}.{name}")


if TYPE_CHECKING:
    from . import logger, metrics  # noqa: F401
