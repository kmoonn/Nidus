"""Retriever for Nidus RAG — combines embedding + vector store search."""

from .chunker import Chunk
from .config import get_config
from .embedder import Embedder
from .store import Store


class Retriever:
    """Retrieve relevant chunks for a given query."""

    def __init__(self):
        config = get_config()
        self._top_k = config.retriever["top_k"]
        self._embedder = Embedder()
        self._store = Store()

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: User query string.
            top_k: Override number of results (uses config default if None).

        Returns:
            List of Chunk objects sorted by relevance.
        """
        k = top_k or self._top_k
        query_embedding = self._embedder.embed_query(query)
        return self._store.query(query_embedding, top_k=k)
