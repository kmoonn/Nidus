"""BM25 + RRF fusion for Nidus Advanced RAG — pure-Python, zero dependencies.

ChromaDB 的内置全文检索会触发其默认 384 维嵌入模型，与本项目 1024 维冲突，
因此这里从零实现 BM25（Okapi BM25 算法），保持 Nidus "不依赖重型框架" 的风格。

混合检索 = 向量检索（语义）+ BM25（精确关键词），用 Reciprocal Rank Fusion (RRF)
融合两路结果，两者优势互补：向量懂同义词，BM25 懂精确术语命中。

解决：纯向量检索精确关键词匹配弱（如具体术语、人名、数字）。
"""

import math
import re
from collections import Counter

from .chunker import Chunk


# 中文按字符 + 英文按词的简易分词。从零实现，不引入 jieba 等分词依赖。
def _tokenize(text: str) -> list[str]:
    """Tokenize mixed CJK + latin text without external dependencies."""
    tokens: list[str] = []
    # Extract latin/digit runs as whole tokens
    for match in re.findall(r"[A-Za-z0-9]+", text):
        tokens.append(match.lower())
    # Treat each CJK character as a token (unigram)
    for ch in text:
        if "一" <= ch <= "鿿":
            tokens.append(ch)
    return tokens


class BM25Index:
    """Okapi BM25 over an in-memory corpus of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._doc_tokens = [_tokenize(c.text) for c in chunks]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

        # term frequency per doc + document frequency across corpus
        self._tf: list[Counter] = [Counter(toks) for toks in self._doc_tokens]
        self._df: Counter = Counter()
        for tf in self._tf:
            for term in tf:
                self._df[term] += 1
        self._n = len(chunks)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Return (chunk, score) pairs ranked by BM25 score, top_k entries."""
        if self._n == 0:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        scores: list[float] = [0.0] * self._n
        for term in set(q_terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            # Inverse document frequency (Okapi variant, always positive)
            idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            for i in range(self._n):
                f = self._tf[i].get(term, 0)
                if f == 0:
                    continue
                dl = self._doc_len[i]
                denom = f + self._k1 * (1 - self._b + self._b * dl / (self._avgdl or 1.0))
                scores[i] += idf * (f * (self._k1 + 1)) / denom

        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(self._chunks[i], s) for i, s in ranked[:top_k]]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    k: float = 60.0,
) -> list[Chunk]:
    """Fuse multiple ranked chunk lists into one via Reciprocal Rank Fusion.

    Each list is already sorted best-first. RRF score = sum(1 / (k + rank)).
    Returns chunks sorted by fused score; duplicates (same Chunk.text) merged,
    keeping the highest score.

    Args:
        ranked_lists: List of ranked chunk lists (e.g. [vector_results, bm25_results]).
        k: RRF constant (default 60).

    Returns:
        Fused, de-duplicated list of chunks sorted by RRF score desc.
    """
    scores: dict[str, float] = {}
    best_chunk: dict[str, Chunk] = {}

    for lst in ranked_lists:
        for rank, chunk in enumerate(lst):
            # Dedup by text content (vector & BM25 may return the same chunk)
            key = chunk.text
            contribution = 1.0 / (k + rank + 1)
            scores[key] = scores.get(key, 0.0) + contribution
            if key not in best_chunk:
                best_chunk[key] = chunk
            else:
                # Preserve rerank/distance metadata from the highest-ranked occurrence
                existing = best_chunk[key]
                for mk, mv in chunk.metadata.items():
                    existing.metadata.setdefault(mk, mv)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for key, score in ordered:
        chunk = best_chunk[key]
        chunk.metadata["rrf_score"] = score
        result.append(chunk)
    return result
