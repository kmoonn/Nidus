"""LLM client factory for Nidus Modular RAG.

Modular 阶段用 langchain-openai 的 ChatOpenAI / OpenAIEmbeddings 接 SiliconFlow
（OpenAI 兼容）。节点共享单例，便于测试时 mock，也避免每个节点重复建客户端。

Reranker 仍用 urllib 直接打 SiliconFlow `/rerank`（rerank 非 OpenAI 兼容接口，
langchain-openai 无对应封装），见 src/nodes/rank.py。
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import get_config

_CHAT_LLM: ChatOpenAI | None = None
_EMBEDDINGS: OpenAIEmbeddings | None = None


def get_chat_llm(**overrides) -> ChatOpenAI:
    """Return a singleton ChatOpenAI configured from config.llm.

    Extra kwargs (temperature, model, etc.) override the config defaults for
    callers that need a different temperature — without constructing a new
    persistent client each call.
    """
    global _CHAT_LLM
    if _CHAT_LLM is None:
        llm_cfg = get_config().llm
        _CHAT_LLM = ChatOpenAI(
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg["api_key"],
            model=llm_cfg["model"],
        )
    if overrides:
        return _CHAT_LLM.bind(**overrides)  # type: ignore[return-value]
    return _CHAT_LLM


def get_embeddings() -> OpenAIEmbeddings:
    """Return a singleton OpenAIEmbeddings configured from config.embedding."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        emb_cfg = get_config().embedding
        _EMBEDDINGS = OpenAIEmbeddings(
            base_url=emb_cfg["base_url"],
            api_key=emb_cfg["api_key"],
            model=emb_cfg["model"],
        )
    return _EMBEDDINGS


def reset_llm_cache() -> None:
    """Clear the cached LLM/embedding singletons (used by tests + re-index)."""
    global _CHAT_LLM, _EMBEDDINGS
    _CHAT_LLM = None
    _EMBEDDINGS = None
