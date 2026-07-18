"""Rank node — dedup + Cross-Encoder rerank.

Post-retrieval：先用 Deduplicator（字符 3-gram Jaccard）去重减少候选量，再用
SiliconFlow `/rerank`（Cross-Encoder）重排，保留 top_n。rerank 非 OpenAI 兼容接口，
沿用 dev/advanced 的 urllib 直连方式。
"""

import json
import urllib.request

from ..chunker import Chunk
from ..dedup import Deduplicator


def _call_rerank_api(
    base_url: str, api_key: str, model: str, query: str, documents: list[str]
) -> list[float]:
    """Call SiliconFlow /rerank; return one score per document (input order)."""
    payload = json.dumps(
        {
            "model": model,
            "query": query,
            "documents": documents,
            "return_documents": False,
            "top_n": len(documents),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/rerank",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    scores = [0.0] * len(documents)
    for item in data.get("results", []):
        idx = item["index"]
        if 0 <= idx < len(documents):
            scores[idx] = item["relevance_score"]
    return scores


def _rerank(query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
    """Rerank chunks via SiliconFlow Cross-Encoder; annotate rerank_score."""
    from ..config import get_config

    if not chunks:
        return []
    if len(chunks) == 1:
        chunks[0].metadata["rerank_score"] = 1.0
        return chunks

    cfg = get_config()
    reranker_cfg = cfg.reranker
    api_key = reranker_cfg.get("api_key")
    if not api_key:
        # No reranker key — skip rerank, keep order.
        for c in chunks:
            c.metadata.setdefault("rerank_score", None)
        return chunks[:top_n]

    base_url = reranker_cfg.get("base_url", "https://api.siliconflow.cn/v1")
    model = reranker_cfg.get("model", "BAAI/bge-reranker-v2-m3")
    try:
        scores = _call_rerank_api(
            base_url, api_key, model, query, [c.text for c in chunks]
        )
    except Exception:
        # Rerank failure must not block — keep original order, top_n only.
        return chunks[:top_n]

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    result = []
    for chunk, score in ranked[:top_n]:
        chunk.metadata["rerank_score"] = score
        result.append(chunk)
    return result


def rank_node(state: dict) -> dict:
    """Dedup then rerank the retrieved chunks (complex path)."""
    from ..config import get_config

    chunks: list[Chunk] = state.get("retrieved_chunks") or []
    if not chunks:
        return {"ranked_chunks": []}

    config = get_config()
    mod = config.modular
    query = state["query"]

    if mod.get("dedup", {}).get("enabled", True):
        threshold = mod.get("dedup", {}).get("similarity_threshold", 0.85)
        chunks = Deduplicator(similarity_threshold=threshold).dedup(chunks)

    if mod.get("reranking", {}).get("enabled", True):
        top_n = mod.get("reranking", {}).get("top_n", config.retriever.get("top_k", 5))
        chunks = _rerank(query, chunks, top_n)

    return {"ranked_chunks": chunks}
