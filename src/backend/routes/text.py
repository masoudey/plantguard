"""Endpoints for text-based question answering."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status

from ..config import settings
from ..services import models
from ..services.text import qa_pipeline

router = APIRouter()


@router.post("/answer", status_code=status.HTTP_200_OK)
async def answer_question(payload: dict) -> dict:
    """Return an answer enriched by retrieval-augmented context."""
    question = (payload.get("question") or "").strip()
    user_context = (payload.get("context") or "").strip()
    top_k = int(payload.get("top_k") or settings.retriever_top_k)

    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    retriever = models.get_retriever()

    candidate_contexts: List[Dict[str, Any]] = []

    if retriever:
        for item in retriever.search(question, top_k=top_k):
            candidate_contexts.append(
                {
                    "text": item["text"],
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "retrieval_score": item["score"],
                    "type": "retrieved",
                }
            )

    if user_context:
        candidate_contexts.insert(
            0,
            {
                "text": user_context,
                "source": "user-provided",
                "chunk_id": "context",
                "retrieval_score": 1.0,
                "type": "user",
            },
        )

    if not candidate_contexts:
        candidate_contexts.append(
            {
                "text": "",
                "source": "",
                "chunk_id": "",
                "retrieval_score": 0.0,
                "type": "empty",
            }
        )

    best_answer: Dict[str, Any] = {
        "answer": "",
        "confidence": 0.0,
        "start": None,
        "end": None,
        "context": "",
        "source": None,
    }

    for context_item in candidate_contexts:
        context_text = context_item["text"]
        response = qa_pipeline.answer_question(question, context_text)
        score = float(response.get("confidence") or 0.0)
        if score >= best_answer["confidence"]:
            best_answer = {
                "answer": response.get("answer", ""),
                "confidence": score,
                "start": response.get("start"),
                "end": response.get("end"),
                "context": context_text,
                "source": {
                    "source": context_item["source"],
                    "chunk_id": context_item["chunk_id"],
                    "retrieval_score": context_item["retrieval_score"],
                    "type": context_item["type"],
                },
            }

    return {
        "answer": best_answer["answer"],
        "confidence": best_answer["confidence"],
        "start": best_answer["start"],
        "end": best_answer["end"],
        "context": best_answer["context"],
        "source": best_answer["source"],
        "sources": candidate_contexts,
    }
