"""Qdrant vector store wiring.

Two zero-config modes, selected purely by configuration:

* **Embedded** (default) — ``NIDUS_QDRANT_URL`` unset. Qdrant runs in-process
  and persists to ``NIDUS_QDRANT_PATH`` on local disk. No Docker required.
  Note: the embedded on-disk store allows only a single client process at a
  time, so run either the API server *or* a CLI command, not both at once.
* **Server** — ``NIDUS_QDRANT_URL`` set (e.g. ``http://localhost:6333``).
  Connects over the network; supports concurrent clients and scaling.
"""

from __future__ import annotations

import atexit
from functools import lru_cache

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from nidus.config import Settings, get_settings
from nidus.models import get_embeddings


@lru_cache
def get_client() -> QdrantClient:
    """Return a cached Qdrant client for the configured mode.

    The embedded (local) client holds an on-disk lock that it releases in
    ``__del__``. Relying on garbage collection at interpreter shutdown raises a
    noisy ``ImportError: sys.meta_path is None`` because modules are already
    torn down. We register an ``atexit`` hook to close it deterministically
    while the interpreter is still healthy.
    """

    settings: Settings = get_settings()
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    client = QdrantClient(path=settings.qdrant_path)
    atexit.register(_close_client, client)
    return client


def _close_client(client: QdrantClient) -> None:
    try:
        client.close()
    except Exception:  # pragma: no cover - best-effort shutdown
        pass


def vectorstore_mode() -> str:
    """Human-readable description of the active vector-store backend."""

    settings = get_settings()
    return f"qdrant@{settings.qdrant_url}" if settings.qdrant_url else "qdrant-embedded"


def ensure_collection() -> None:
    """Create the configured collection if it does not already exist."""

    settings = get_settings()
    client = get_client()
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embed_dim, distance=Distance.COSINE
            ),
        )


def get_vectorstore() -> QdrantVectorStore:
    """Return a :class:`QdrantVectorStore` bound to the configured collection.

    The collection is created on demand so first-run ingestion just works.
    """

    ensure_collection()
    settings = get_settings()
    return QdrantVectorStore(
        client=get_client(),
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
        retrieval_mode=RetrievalMode.DENSE,
    )


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Return a retriever over the configured collection."""

    settings = get_settings()
    return get_vectorstore().as_retriever(
        search_kwargs={"k": k or settings.retriever_k}
    )
