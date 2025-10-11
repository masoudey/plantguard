"""Fusion service orchestrating multimodal embeddings."""
from __future__ import annotations

from typing import Optional

import io

import torch
import torch.nn.functional as F
import soundfile as sf

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


def run_fusion(
    image_bytes: bytes,
    audio_bytes: bytes | None,
    question: str | None,
    *,
    audio_filename: str | None = None,
    audio_content_type: str | None = None,
) -> dict:
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
        sample_rate = getattr(audio_model, "sample_rate", 16000)
        max_length = getattr(audio_model, "max_length", None)
        waveforms, lengths = preprocess.transform_audio_bytes(
            audio_bytes,
            sample_rate=sample_rate,
            max_length=max_length,
            filename=audio_filename,
            content_type=audio_content_type,
        )
        waveforms = waveforms.to(device)
        lengths = lengths.to(device)
        with torch.no_grad():
            audio_logits = audio_model(waveforms, lengths)
        audio_predictions = models.format_predictions(
            audio_logits.squeeze(0), labels=audio_model.class_names
        )
        audio_embed = _fit_dimension(
            audio_logits,
            getattr(fusion_model, "audio_dim", audio_logits.shape[-1]),
        )
        try:
            buffer = io.BytesIO()
            sf.write(
                buffer,
                waveforms.squeeze(0).cpu().numpy(),
                sample_rate,
                format="WAV",
            )
            transcript_bytes = buffer.getvalue()
        except Exception:
            transcript_bytes = b""
        if transcript_bytes:
            transcript = models.get_transcriber().transcribe(transcript_bytes)
        else:
            transcript = ""

    retriever_results = []
    text_predictions = None
    text_dim = getattr(fusion_model, "text_dim", image_embed.shape[-1])
    text_embed = torch.zeros((image_embed.shape[0], text_dim), device=device)

    if question and retriever:
        retriever_results = retriever.search(question, top_k=settings.retriever_top_k)
        context_text = retriever_results[0]["text"] if retriever_results else None
        query_vec = retriever.build_query_embedding(question, context_text)
        query_tensor = torch.from_numpy(query_vec).to(device)
        text_embed = _fit_dimension(query_tensor, text_dim)
        if retriever_results:
            scores = torch.tensor([item.get("score", 0.0) for item in retriever_results], dtype=torch.float32)
            weights = torch.softmax(-scores, dim=0).tolist()
            text_predictions = {
                "top_k": [
                    {
                        "label": item.get("source") or f"chunk_{item.get('chunk_id', idx)}",
                        "confidence": float(weights[idx]),
                        "text": item.get("text", ""),
                        "score": float(item.get("score", 0.0)),
                    }
                    for idx, item in enumerate(retriever_results)
                ],
            }
    elif question:
        text_embed = torch.randn_like(text_embed)

    if text_predictions is None and question:
        text_predictions = {
            "top_k": [
                {
                    "label": "user_question",
                    "confidence": 1.0,
                    "text": question,
                    "score": 0.0,
                }
            ],
        }

    text_answers = []
    weight_lookup = {}
    if text_predictions and text_predictions.get("top_k"):
        weight_lookup = {
            entry.get("label"): float(entry.get("confidence", 0.0))
            for entry in text_predictions["top_k"]
        }
    if question:
        qa = models.get_qa_pipeline()
        candidates = retriever_results if retriever_results else [
            {"text": "", "source": "no_context", "score": 0.0, "chunk_id": ""}
        ]
        for idx, item in enumerate(candidates):
            context_text = item.get("text", "")
            try:
                qa_result = qa(question=question, context=context_text)
            except Exception as exc:  # pragma: no cover - safety net
                qa_result = {"answer": "", "confidence": 0.0}
                models.logger.warning("QA pipeline failed: %s", exc)
            text_answers.append(
                {
                    "label": item.get("source") or f"chunk_{item.get('chunk_id', idx)}",
                    "answer": qa_result.get("answer", ""),
                    "confidence": float(qa_result.get("confidence") or 0.0),
                    "context": context_text,
                    "score": float(item.get("score", 0.0)),
                    "weight": weight_lookup.get(item.get("source") or f"chunk_{item.get('chunk_id', idx)}"),
                }
            )
        text_answers.sort(key=lambda entry: entry["confidence"], reverse=True)

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
        "text": {
            "top_k": text_answers if text_answers else text_predictions.get("top_k") if text_predictions else [],
        },
        "fusion": models.format_predictions(
            fused, labels=getattr(fusion_model, "class_names", None)
        ),
        "retrieval": retriever_results,
    }
