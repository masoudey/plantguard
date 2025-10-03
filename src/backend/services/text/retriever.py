"""LangChain-powered retrieval utilities for the PlantGuard RAG pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


@dataclass(frozen=True)
class RetrieverConfig:
    model_name: str
    index_path: Path
    index_name: str


class KnowledgeBase:
    """Thin wrapper around LangChain's FAISS vector store."""

    def __init__(self, config: RetrieverConfig) -> None:
        if not config.index_path.exists():
            raise FileNotFoundError(f"Knowledge base directory {config.index_path} not found")

        self.embedder = HuggingFaceEmbeddings(model_name=config.model_name)
        self.vectorstore = FAISS.load_local(
            str(config.index_path),
            self.embedder,
            index_name=config.index_name,
            allow_dangerous_deserialization=True,
        )

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if not query.strip():
            return []
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        results: List[dict] = []
        for document, score in docs_with_scores:
            results.append(
                {
                    "score": float(score),
                    "text": document.page_content,
                    "source": document.metadata.get("source", ""),
                    "chunk_id": document.metadata.get("chunk_id"),
                }
            )
        return results

    @lru_cache(maxsize=128)
    def build_query_embedding(self, question: str, context: Optional[str] = None) -> np.ndarray:
        payload = question if context is None else f"{question}\n{context}"
        vector = self.embedder.embed_query(payload)
        return np.asarray(vector, dtype="float32")[None, :]


def load_retriever_config(base_dir: Path, default_model: str) -> RetrieverConfig:
    config_path = base_dir / "config.json"
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = data.get("embedding_model", default_model)
        index_name = data.get("index_name", "knowledge_index")
    else:
        model_name = default_model
        index_name = "knowledge_index"
    return RetrieverConfig(model_name=model_name, index_path=base_dir, index_name=index_name)
