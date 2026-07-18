"""Central configuration for Nidus.

All settings are read from the environment (or a local ``.env`` file) using the
``NIDUS_`` prefix. The design goal is *config-driven extensibility*: the chat
model, embeddings and vector store are all selected via configuration rather
than code, so the same codebase targets OpenAI, Doubao/Ark, DeepSeek, Ollama,
vLLM or any other OpenAI-compatible endpoint.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="NIDUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Chat model (OpenAI-compatible) -----------------------------------
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL of the OpenAI-compatible chat endpoint.",
    )
    llm_api_key: str = Field(
        default="sk-no-key",
        description="API key for the chat endpoint.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Chat model name/deployment.",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(
        default=1024,
        gt=0,
        description=(
            "Max tokens per generation. Caps output so weaker models that "
            "degenerate into repetition loops fail fast instead of hanging."
        ),
    )
    llm_frequency_penalty: float = Field(
        default=0.3,
        ge=-2.0,
        le=2.0,
        description=(
            "Discourages token repetition. A small positive value keeps "
            "smaller open models from collapsing into repeated text."
        ),
    )
    llm_request_timeout: float = Field(
        default=60.0,
        gt=0,
        description="Per-request timeout (seconds) for LLM calls.",
    )

    # -- Embeddings (OpenAI-compatible) -----------------------------------
    # Falls back to the chat credentials when the embedding-specific values
    # are left unset, which is the common single-provider case.
    embed_base_url: str | None = Field(default=None)
    embed_api_key: str | None = Field(default=None)
    embed_model: str = Field(default="text-embedding-3-small")
    embed_dim: int = Field(
        default=1536,
        gt=0,
        description="Embedding vector dimensionality; must match the model.",
    )

    # -- Qdrant vector store ----------------------------------------------
    # Leave ``qdrant_url`` empty to use the zero-infra embedded mode, which
    # persists to ``qdrant_path`` on local disk.
    qdrant_url: str | None = Field(default=None)
    qdrant_api_key: str | None = Field(default=None)
    qdrant_path: str = Field(default="./.nidus/qdrant")
    qdrant_collection: str = Field(default="nidus")

    # -- Retrieval / ingestion --------------------------------------------
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    retriever_k: int = Field(default=4, gt=0)

    # -- Agent graph -------------------------------------------------------
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Max query-rewrite / regeneration loops before giving up.",
    )
    enable_direct_answer: bool = Field(
        default=True,
        description="Allow the router to answer chit-chat without retrieval.",
    )

    # -- API server --------------------------------------------------------
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, gt=0, lt=65536)

    @property
    def resolved_embed_base_url(self) -> str:
        return self.embed_base_url or self.llm_base_url

    @property
    def resolved_embed_api_key(self) -> str:
        return self.embed_api_key or self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()
