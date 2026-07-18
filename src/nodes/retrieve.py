"""Retrieve node — vector + optional BM25 hybrid retrieval with RRF fusion.

complex 路径下 query_transform 产出多个子查询，每条查询做向量（+可选 BM25）召回，
再用 RRF 融合成一路结果。simple 路径单查询直接检索。

复用 Store（ChromaDB）/ Embedder / bm25（BM25Index + reciprocal_rank_fusion）。
检索相关度（distance）写入 chunk.metadata，供下游相关度门控与 Rank 使用。
"""

from ..bm25 import BM25Index, reciprocal_rank_fusion
from ..chunker import Chunk
from ..embedder import Embedder
from ..store import Store


def _search_one(
    query: str,
    embedder: Embedder,
    store: Store,
    top_k: int,
    candidate_k: int,
    bm25: BM25Index | None,
) -> list[Chunk]:
    """Retrieve for a single query: vector (+ BM25) → RRF fusion if hybrid."""
    q_emb = embedder.embed_query(query)
    vector_chunks = store.query(q_emb, top_k=candidate_k)

    if bm25 is None:
        return vector_chunks

    bm25_results = [c for c, _ in bm25.search(query, top_k=candidate_k)]
    return reciprocal_rank_fusion([vector_chunks, bm25_results])


def retrieve_node(state: dict) -> dict:
    """Retrieve chunks for all queries; multi-query RRF fusion when >1 query."""
    from ..config import get_config

    queries: list[str] = state.get("queries") or [state["query"]]
    config = get_config()
    mod = config.modular
    top_k = config.retriever.get("top_k", 5)
    candidate_k = max(top_k * 3, top_k + 5)

    store = Store()
    embedder = Embedder()

    bm25: BM25Index | None = None
    if mod.get("hybrid_search", {}).get("enabled", True) and store.count > 0:
        bm25 = BM25Index(store.all_chunks())

    if store.count == 0:
        return {"retrieved_chunks": []}

    ranked_lists: list[list[Chunk]] = []
    for q in queries:
        ranked_lists.append(
            _search_one(q, embedder, store, top_k, candidate_k, bm25)
        )

    # Multi-query fusion (only when >1 distinct ranked list)
    if len(ranked_lists) > 1:
        chunks = reciprocal_rank_fusion(ranked_lists)[:top_k]
    else:
        chunks = ranked_lists[0][:top_k]

    return {"retrieved_chunks": chunks}
