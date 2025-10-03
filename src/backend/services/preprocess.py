"""Shared preprocessing utilities for PlantGuard."""
from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .audio import features as audio_features

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def transform_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Convert raw bytes into a normalized tensor suitable for the vision model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    return tensor


def transform_audio_bytes(audio_bytes: bytes, pad_to: int | None = None) -> torch.Tensor:
    """Convert raw WAV/MP3 bytes into MFCC features tensor."""
    features = audio_features.mfcc_from_bytes(audio_bytes)
    if pad_to is not None:
        features = audio_features.pad_features(features, max_length=pad_to)
    tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)
    return tensor.float()


def transform_text(question: str, context: Optional[str] = None) -> dict:
    """Placeholder for text preprocessing (tokenization handled elsewhere)."""
    return {"question": question, "context": context or ""}
