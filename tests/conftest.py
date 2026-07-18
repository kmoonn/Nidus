"""Shared pytest fixtures and lightweight fakes.

The fakes are duck-typed: nodes call ``bind`` / ``invoke`` on the chat model and
``invoke`` on the retriever, so we avoid subclassing the real ``BaseChatModel``
machinery. Tests run fully offline with no API keys.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, SystemMessage

from nidus import prompts

# A decision reply is either a fixed string or a callable of the messages.
Reply = "str | Callable[[Sequence[Any]], str]"


def _detect_kind(messages: Sequence[Any]) -> str:
    """Classify what the graph is asking, by matching the system prompt.

    Keying off the imported ``prompts`` constants (rather than hardcoded text)
    keeps the fake robust to prompt wording changes.
    """

    system = next(
        (m.content for m in messages if isinstance(m, SystemMessage)), None
    )
    return {
        prompts.ROUTER_SYSTEM: "route",
        prompts.DOC_GRADER_SYSTEM: "doc_grade",
        prompts.HALLUCINATION_SYSTEM: "hallucination",
        prompts.ANSWER_GRADER_SYSTEM: "answer_grade",
        prompts.REWRITE_SYSTEM: "rewrite",
        prompts.DIRECT_SYSTEM: "direct",
    }.get(system, "generate")  # generation has no system message


class FakeChatModel:
    """Minimal chat model stand-in driven by per-decision replies.

    Each decision kind (``route``, ``doc_grade``, ``hallucination``,
    ``answer_grade``, ``rewrite``, ``direct``, ``generate``) resolves to either
    a fixed string or a ``callable(messages) -> str``, mirroring how a real
    model returns a single classification word or free text.
    """

    def __init__(
        self,
        *,
        route: Any = "vectorstore",
        doc_grade: Any = "yes",
        hallucination: Any = "yes",
        answer_grade: Any = "yes",
        rewrite: Any = "rewritten query",
        answer: Any = "FAKE ANSWER",
    ) -> None:
        self._replies = {
            "route": route,
            "doc_grade": doc_grade,
            "hallucination": hallucination,
            "answer_grade": answer_grade,
            "rewrite": rewrite,
            "direct": answer,
            "generate": answer,
        }
        self.calls: list[str] = []

    def bind(self, **_kwargs: Any) -> "FakeChatModel":
        # Nodes cap decision length via bind(max_tokens=...); we just record it.
        self.calls.append("bind")
        return self

    def invoke(self, messages: Sequence[Any]) -> AIMessage:
        kind = _detect_kind(messages)
        self.calls.append(kind)
        reply = self._replies[kind]
        text = reply(messages) if callable(reply) else reply
        return AIMessage(content=text)



class FakeRetriever:
    """Returns a fixed document list regardless of the query."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.calls = 0

    def invoke(self, _query: str) -> list[Document]:
        self.calls += 1
        return list(self.docs)


class FakeEmbeddings(Embeddings):
    """Deterministic, offline embeddings.

    Identical text always maps to the identical unit vector, so a query equal
    to a stored document scores a perfect cosine match — enough to assert
    retrieval behaviour without a real embedding provider.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        values = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset all lru_cache singletons so env overrides take effect per-test."""

    def _clear():
        from nidus import config, models
        from nidus.graph import builder

        config.get_settings.cache_clear()
        models.get_chat_model.cache_clear()
        models.get_embeddings.cache_clear()
        builder.get_graph.cache_clear()
        try:
            from nidus import vectorstore

            # Close the cached embedded client before dropping it so its
            # on-disk lock is released and the next test can reopen the path.
            cached = vectorstore.get_client.cache_info().currsize
            if cached:
                try:
                    vectorstore.get_client().close()
                except Exception:
                    pass
            vectorstore.get_client.cache_clear()
        except Exception:
            pass

    _clear()
    yield
    _clear()


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Nidus is a light extensible agentic RAG platform.",
            metadata={"source": "doc1"},
        ),
        Document(
            page_content="LangGraph orchestrates stateful multi-step agents.",
            metadata={"source": "doc2"},
        ),
    ]
