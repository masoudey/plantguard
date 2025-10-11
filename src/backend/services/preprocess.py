"""Shared preprocessing utilities for PlantGuard."""
from __future__ import annotations

import io
from typing import Optional

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


def transform_audio_bytes(
    audio_bytes: bytes,
    *,
    sample_rate: int = 16000,
    max_length: int | None = None,
    filename: str | None = None,
    content_type: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw audio bytes into padded waveform tensor and lengths."""
    waveform, _ = audio_features.load_waveform_from_bytes(
        audio_bytes,
        target_sr=sample_rate,
        filename=filename,
        content_type=content_type,
    )
    original_length = waveform.numel()
    if max_length is not None:
        waveform, original_length = audio_features.pad_or_trim_waveform(waveform, max_length)
    waveforms = waveform.unsqueeze(0)
    lengths = torch.tensor([original_length], dtype=torch.long)
    return waveforms.float(), lengths


def transform_text(question: str, context: Optional[str] = None) -> dict:
    """Placeholder for text preprocessing (tokenization handled elsewhere)."""
    return {"question": question, "context": context or ""}
