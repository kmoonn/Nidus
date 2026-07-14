"""Vector store for Nidus RAG — ChromaDB persistence layer."""

from pathlib import Path

import chromadb

from .chunker import Chunk
from .config import get_config


class Store:
    """ChromaDB-backed vector store for document chunks."""

    def __init__(self):
        config = get_config()
        store_cfg = config.store

        persist_dir = Path(store_cfg["persist_directory"])
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection_name = store_cfg["collection_name"]
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Number of documents in the store."""
        return self._collection.count()

    def add_documents(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add document chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects.
            embeddings: Corresponding embedding vectors.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) != number of embeddings ({len(embeddings)})"
            )

        if not chunks:
            return

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[Chunk]:
        """Query the store for similar chunks.

        Args:
            query_embedding: Embedding vector of the query.
            top_k: Number of results to return.

        Returns:
            List of Chunk objects sorted by similarity (most similar first).
        """
        if self.count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None
                metadata["distance"] = distance
                chunks.append(Chunk(text=doc, metadata=metadata))

        return chunks

    def reset(self) -> None:
        """Delete the collection and recreate it (for re-indexing)."""
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
