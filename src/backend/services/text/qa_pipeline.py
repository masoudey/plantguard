"""QA pipeline helper using Hugging Face Transformers."""
from __future__ import annotations

from functools import lru_cache

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

MODEL_NAME = "distilbert-base-uncased-distilled-squad"


@lru_cache(maxsize=1)
def get_qa_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)
