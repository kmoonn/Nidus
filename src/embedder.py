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
            List of embedding vectors (one per input text). On per-item API
            errors (SiliconFlow occasionally returns 400 for individual inputs
            due to transient rate-limiting), the failing item is retried in
            isolation; if it still fails, a zero vector is returned for that
            slot so the batch never aborts indexing.
        """
        if not texts:
            return []

        try:
            return self._embed_batch(texts)
        except Exception:
            # Batch failed — fall back to embedding items one by one so a
            # single problematic text doesn't abort the whole index build.
            return [self._embed_one_safe(t) for t in texts]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch; raises on API error."""
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]

    def _embed_one_safe(self, text: str) -> list[float]:
        """Embed a single text, returning a zero vector on failure."""
        if not text.strip():
            dim = 1024
            return [0.0] * dim
        for attempt in range(2):
            try:
                return self._embed_batch([text])[0]
            except Exception:
                if attempt == 1:
                    # Give up after retry; keep indexing going with a placeholder.
                    print(f"  [warn] embedding failed for text (len={len(text)}), using zero vector")
                    return [0.0] * 1024
        return [0.0] * 1024

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query string.

        Args:
            text: Query text to embed.

        Returns:
            Single embedding vector.
        """
        results = self.embed_texts([text])
        return results[0]
