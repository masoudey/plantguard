"""Explainability utilities for the PlantGuard models."""
from __future__ import annotations

from typing import Any, Dict


def generate_gradcam_overlay(image_bytes: bytes) -> Dict[str, Any]:
    """Placeholder implementation returning a mocked heatmap descriptor."""
    return {
        "message": "Grad-CAM overlay not yet implemented",
    }


def format_attention_weights(weights) -> Dict[str, Any]:
    """Placeholder for attention visualisation."""
    return {
        "attention": [],
        "note": "Attention visualisation to be implemented",
    }
