"""Model factory: build OpenAI-compatible chat and embedding clients.

Everything is driven by :class:`nidus.config.Settings`, so pointing Nidus at a
different provider (Doubao/Ark, DeepSeek, Ollama, vLLM, ...) is purely a matter
of changing ``NIDUS_LLM_BASE_URL`` / ``NIDUS_LLM_MODEL`` etc.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from nidus.config import Settings, get_settings


@lru_cache
def get_chat_model(temperature: float | None = None) -> BaseChatModel:
    """Return a cached OpenAI-compatible chat model.

    Applies portability-oriented stability controls: a ``max_tokens`` cap and a
    small ``frequency_penalty`` so weaker open models can't degenerate into
    unbounded repetition loops, plus a per-request timeout.
    """

    settings: Settings = get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        frequency_penalty=settings.llm_frequency_penalty,
        timeout=settings.llm_request_timeout,
    )


@lru_cache
def get_embeddings() -> Embeddings:
    """Return a cached OpenAI-compatible embeddings client."""

    settings: Settings = get_settings()
    return OpenAIEmbeddings(
        base_url=settings.resolved_embed_base_url,
        api_key=settings.resolved_embed_api_key,
        model=settings.embed_model,
    )
