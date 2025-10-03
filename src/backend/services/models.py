"""Model loading helpers with lightweight mock implementations."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..config import settings
from .fusion_model import MultimodalFusionHead
from .vision import VisionClassifier
from .audio.model import AudioClassifier
from .audio.transcriber import SpeechTranscriber
from .text.qa_pipeline import get_qa_pipeline
from .text.retriever import KnowledgeBase, load_retriever_config


@dataclass(slots=True)
class Prediction:
    label: str
    score: float


def _ensure_model_dir() -> Path:
    path = Path(settings.model_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def get_vision_model() -> VisionClassifier:
    model_dir = _ensure_model_dir()
    checkpoint = model_dir / "vision/plantguard_resnet50.pt"
    return VisionClassifier(device=get_device(), checkpoint=checkpoint if checkpoint.exists() else None)


@lru_cache(maxsize=1)
def get_audio_model() -> AudioClassifier:
    model_dir = _ensure_model_dir()
    checkpoint = model_dir / "audio/plantguard_cnn_lstm.pt"
    class_names = None
    num_classes = 4
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        class_names = state.get("class_names")
        if class_names:
            num_classes = len(class_names)
    return AudioClassifier(
        device=get_device(),
        num_classes=num_classes,
        class_names=class_names,
        checkpoint=checkpoint if checkpoint.exists() else None,
    )


@lru_cache(maxsize=1)
def get_fusion_model() -> MultimodalFusionHead:
    model_dir = _ensure_model_dir()
    checkpoint = model_dir / "fusion/plantguard_mlp.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        metadata = state.get("metadata", {})
        return MultimodalFusionHead(
            device=get_device(),
            image_dim=metadata.get("image_dim", 4),
            text_dim=metadata.get("text_dim", 4),
            audio_dim=metadata.get("audio_dim"),
            num_classes=metadata.get("num_classes", 4),
            class_names=metadata.get("class_names"),
            checkpoint=checkpoint,
        )
    return MultimodalFusionHead(device=get_device())


@lru_cache(maxsize=1)
def get_transcriber() -> SpeechTranscriber:
    return SpeechTranscriber()


@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeBase | None:
    base_dir = Path(settings.knowledge_base_dir)
    if not base_dir.exists():
        logger.warning("Knowledge base directory not found", directory=str(base_dir))
        return None
    try:
        config = load_retriever_config(base_dir, settings.knowledge_embedding_model)
        return KnowledgeBase(config)
    except Exception as exc:  # pragma: no cover - safeguard for optional dependency issues
        logger.error("Failed to initialise knowledge base", error=str(exc))
        return None


def format_predictions(logits: np.ndarray | torch.Tensor, labels: list[str] | None = None) -> dict:
    """Return top-k predictions with labels and probabilities."""
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    top_indices = np.argsort(probabilities)[::-1][:3]
    label_list = labels or [f"class_{idx}" for idx in range(probabilities.shape[-1])]
    return {
        "top_k": [
            {
                "label": label_list[idx] if idx < len(label_list) else f'class_{idx}',
                "confidence": float(probabilities[idx]),
            }
            for idx in top_indices
        ]
    }


__all__ = [
    "get_vision_model",
    "get_audio_model",
    "get_fusion_model",
    "get_transcriber",
    "get_retriever",
    "get_device",
    "format_predictions",
    "get_qa_pipeline",
]
logger = logging.getLogger(__name__)
