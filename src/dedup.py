"""Deduplicator for Nidus Advanced RAG — post-retrieval, pure-logic, no LLM.

检索结果可能含高度重复的 chunk（同源不同页含重复段落，或向量+BM25 召回近似重复）。
去重在 rerank 之前移除高度相似的 chunk，既减少 rerank 候选量，又避免重复上下文干扰生成。

完全相同的 chunk 已由 RRF 融合处理；这里处理"高度相似但不完全相同"的近似重复，
基于字符 n-gram 集合的 Jaccard 相似度判断（纯 Python，零新依赖，零额外 API 调用）。

解决：检索结果内容高度重复，挤占上下文、干扰回答。
"""

from .chunker import Chunk


def _ngram_set(text: str, n: int = 3) -> set[str]:
    """Build the set of character n-grams for a text (n defaults to 3)."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two n-gram sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class Deduplicator:
    """Remove near-duplicate chunks by character n-gram Jaccard similarity."""

    def __init__(self, similarity_threshold: float = 0.85, ngram_size: int = 3):
        self._threshold = similarity_threshold
        self._n = ngram_size

    def dedup(self, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunks with near-duplicates removed (keeps the first of each cluster).

        Order is preserved; a chunk is dropped only if it is highly similar
        (Jaccard >= threshold) to an already-kept chunk.

        Args:
            chunks: Retrieved chunks, best-first.

        Returns:
            Deduplicated list (subset, original order).
        """
        if not chunks:
            return []

        kept: list[Chunk] = []
        kept_grams: list[set[str]] = []

        for chunk in chunks:
            grams = _ngram_set(chunk.text, self._n)
            is_dup = any(_jaccard(grams, g) >= self._threshold for g in kept_grams)
            if is_dup:
                continue
            kept.append(chunk)
            kept_grams.append(grams)

        return kept
