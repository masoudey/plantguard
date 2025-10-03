"""Question answering utilities leveraging the fine-tuned QA model."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_DIR = Path("models/text/plantguard_qa_head")
BASE_MODEL = "distilbert-base-uncased-distilled-squad"


@lru_cache(maxsize=1)
def _load_tokenizer():
    source = MODEL_DIR if MODEL_DIR.exists() else BASE_MODEL
    return AutoTokenizer.from_pretrained(source, local_files_only=MODEL_DIR.exists())


@lru_cache(maxsize=1)
def _load_model():
    source = MODEL_DIR if MODEL_DIR.exists() else BASE_MODEL
    model = AutoModelForQuestionAnswering.from_pretrained(source, local_files_only=MODEL_DIR.exists())
    model.eval()
    return model


def answer_question(question: str, context: str) -> Dict[str, float | str | int | None]:
    """Infer answer span and confidence without using the HF pipeline helper."""
    if not question.strip():
        return {"answer": "", "confidence": 0.0, "start": None, "end": None}

    tokenizer = _load_tokenizer()
    model = _load_model()

    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offset_mapping = encoded.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = model(**encoded)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start_prob = torch.softmax(start_logits, dim=-1)
    end_prob = torch.softmax(end_logits, dim=-1)

    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()
    if end_idx < start_idx:
        end_idx = start_idx

    start_char, _ = offset_mapping[start_idx].tolist()
    _, end_char = offset_mapping[end_idx].tolist()

    tokens = encoded["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()

    confidence = float((start_prob[start_idx] * end_prob[end_idx]).sqrt().item())

    return {
        "answer": answer,
        "confidence": confidence,
        "start": int(start_char),
        "end": int(end_char),
    }
