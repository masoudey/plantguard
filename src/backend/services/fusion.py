"""Fusion service orchestrating multimodal embeddings."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..config import settings
from . import models, preprocess


def _fit_dimension(tensor: torch.Tensor, target_dim: Optional[int]) -> torch.Tensor:
    if target_dim is None:
        return tensor
    if tensor.shape[-1] == target_dim:
        return tensor
    if tensor.shape[-1] > target_dim:
        return tensor[..., :target_dim]
    pad_width = target_dim - tensor.shape[-1]
    return F.pad(tensor, (0, pad_width))


def run_fusion(image_bytes: bytes, audio_bytes: bytes | None, question: str | None) -> dict:
    """Aggregate modality predictions into a unified response."""
    device = models.get_device()
    vision_model = models.get_vision_model()
    audio_model = models.get_audio_model()
    fusion_model = models.get_fusion_model()
    retriever = models.get_retriever()

    image_tensor = preprocess.transform_image_bytes(image_bytes).to(device)
    with torch.no_grad():
        image_features = vision_model.forward(image_tensor)
    image_embed = _fit_dimension(
        image_features,
        getattr(fusion_model, "image_dim", image_features.shape[-1]),
    )

    audio_embed = None
    transcript = None
    audio_predictions = None
    if audio_bytes and audio_model is not None:
        pad_to = getattr(audio_model, "pad_to", None)
        features = preprocess.transform_audio_bytes(audio_bytes, pad_to=pad_to).to(device)
        with torch.no_grad():
            audio_logits = audio_model(features)
        audio_predictions = models.format_predictions(
            audio_logits.squeeze(0), labels=audio_model.class_names
        )
        audio_embed = _fit_dimension(
            audio_logits,
            getattr(fusion_model, "audio_dim", audio_logits.shape[-1]),
        )
        transcript = models.get_transcriber().transcribe(audio_bytes)

    retriever_results = []
    text_dim = getattr(fusion_model, "text_dim", image_embed.shape[-1])
    text_embed = torch.zeros((image_embed.shape[0], text_dim), device=device)

    if question and retriever:
        retriever_results = retriever.search(question, top_k=settings.retriever_top_k)
        context_text = retriever_results[0]["text"] if retriever_results else None
        query_vec = retriever.build_query_embedding(question, context_text)
        query_tensor = torch.from_numpy(query_vec).to(device)
        text_embed = _fit_dimension(query_tensor, text_dim)
    elif question:
        text_embed = torch.randn_like(text_embed)

    fused = fusion_model.predict(
        image_embed=image_embed,
        text_embed=text_embed,
        audio_embed=audio_embed,
    )

    return {
        "vision": models.format_predictions(
            image_features.squeeze(0), labels=vision_model.class_names
        ),
        "audio": audio_predictions,
        "transcript": transcript,
        "fusion": models.format_predictions(
            fused, labels=getattr(fusion_model, "class_names", None)
        ),
        "retrieval": retriever_results,
    }
