"""Embedding generator for Nidus RAG — uses SiliconFlow Embedding API."""

from openai import OpenAI

from .config import get_config


class Embedder:
    """Generate text embeddings via SiliconFlow API (OpenAI-compatible)."""

    def __init__(self):
        config = get_config()
        emb_cfg = config.embedding
        self._model = emb_cfg["model"]
        self._client = OpenAI(
            base_url=emb_cfg["base_url"],
            api_key=emb_cfg["api_key"],
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        # OpenAI embeddings API supports batch input
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        # Sort by index to ensure order matches input
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query string.

        Args:
            text: Query text to embed.

        Returns:
            Single embedding vector.
        """
        results = self.embed_texts([text])
        return results[0]
