"""Unit tests for the pure-Python BM25 index and RRF fusion.

These need no API key — BM25 and RRF are implemented from scratch with no
network calls.
"""

from src.bm25 import BM25Index, reciprocal_rank_fusion, _tokenize
from src.chunker import Chunk


def _chunk(text: str, src: str = "doc.pdf", page: int = 1) -> Chunk:
    return Chunk(text=text, metadata={"source": src, "page": page})


# ── Tokenizer ────────────────────────────────────────────────


def test_tokenize_cjk_unigrams():
    """Chinese text is tokenized into single characters (unigram model)."""
    tokens = _tokenize("碳达峰目标")
    assert tokens == ["碳", "达", "峰", "目", "标"]


def test_tokenize_latin_runs():
    """Latin/digit runs are kept as whole lowercase tokens."""
    tokens = _tokenize("DeepSeek 2030年")
    assert "deepseek" in tokens
    assert "2030" in tokens
    # 年 is a CJK char -> unigram
    assert "年" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []
    assert _tokenize("   ") == []


# ── BM25 ranking ─────────────────────────────────────────────


def test_bm25_ranks_relevant_doc_first():
    """The doc containing query terms should rank above unrelated docs."""
    chunks = [
        _chunk("中国承诺在2030年前实现碳达峰。"),
        _chunk("今天天气真不错，适合出门。"),
        _chunk("碳中和是指净零排放。"),
    ]
    bm25 = BM25Index(chunks)
    results = bm25.search("碳达峰", top_k=3)
    assert results[0][0].text.startswith("中国承诺")
    assert results[0][1] > 0


def test_bm25_no_match_returns_empty():
    chunks = [_chunk("今天天气不错")]
    bm25 = BM25Index(chunks)
    assert bm25.search("碳达峰", top_k=3) == []


def test_bm25_empty_corpus():
    bm25 = BM25Index([])
    assert bm25.search("anything", top_k=3) == []


def test_bm25_top_k_limits_results():
    chunks = [_chunk(f"碳达峰内容{i}") for i in range(5)]
    bm25 = BM25Index(chunks)
    results = bm25.search("碳达峰", top_k=2)
    assert len(results) == 2


def test_bm25_term_with_higher_df_scores_lower():
    """A rare term should score higher than a ubiquitous term for a matching doc."""
    chunks = [
        _chunk("碳达峰 碳达峰 碳达峰"),  # term A in many docs
        _chunk("碳达峰 稀有术语"),
        _chunk("碳达峰 碳达峰 碳达峰"),
    ]
    bm25 = BM25Index(chunks)
    # Searching for the rare term should surface doc 1
    results = bm25.search("稀有术语", top_k=3)
    assert results
    assert "稀有术语" in results[0][0].text


# ── RRF fusion ───────────────────────────────────────────────


def test_rrf_merges_and_dedups():
    """A chunk appearing in both lists ranks above one in a single list."""
    a = _chunk("共同相关内容")
    b = _chunk("仅向量召回")
    c = _chunk("仅BM25召回")
    vector_list = [a, b]
    bm25_list = [a, c]

    fused = reciprocal_rank_fusion([vector_list, bm25_list])

    # 'a' appears rank 0 in both lists -> highest RRF score -> first
    assert fused[0].text == "共同相关内容"
    assert "rrf_score" in fused[0].metadata
    # fused length == unique chunks (3)
    assert len(fused) == 3


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list():
    chunks = [_chunk("x"), _chunk("y")]
    fused = reciprocal_rank_fusion([chunks])
    assert [c.text for c in fused] == ["x", "y"]
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]
