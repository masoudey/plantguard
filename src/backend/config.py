"""Configuration management for PlantGuard backend."""
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = Field("development", alias="PLANTGUARD_ENV")
    cors_origins_raw: str = Field(
        "http://localhost:8501,http://localhost:3000", alias="CORS_ORIGINS"
    )
    model_dir: str = Field("models", alias="PLANTGUARD_MODEL_DIR")
    confidence_threshold: float = 0.6
    knowledge_base_dir: str = Field(
        "models/text/knowledge_base", alias="PLANTGUARD_KNOWLEDGE_DIR"
    )
    knowledge_embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="PLANTGUARD_KNOWLEDGE_MODEL"
    )
    retriever_top_k: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
