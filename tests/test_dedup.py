"""Tests for the Deduplicator — pure logic, no LLM."""

from src.dedup import Deduplicator, _ngram_set, _jaccard
from src.chunker import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(text=text, metadata={"source": "d.pdf", "page": 1})


def test_ngram_set_basic():
    grams = _ngram_set("abcdef", n=3)
    assert grams == {"abc", "bcd", "cde", "def"}


def test_ngram_set_short_text():
    assert _ngram_set("ab", n=3) == {"ab"}
    assert _ngram_set("", n=3) == set()


def test_jaccard_identical():
    s = _ngram_set("碳达峰目标", 3)
    assert _jaccard(s, s) == 1.0


def test_jaccard_disjoint():
    a = {"abc", "def"}
    b = {"ghi", "jkl"}
    assert _jaccard(a, b) == 0.0


def test_jaccard_both_empty():
    assert _jaccard(set(), set()) == 1.0


def test_dedup_removes_near_duplicates():
    """Two chunks differing only slightly should collapse to one."""
    dd = Deduplicator(similarity_threshold=0.85)
    a = _chunk("中国承诺在2030年前实现碳达峰，努力争取2060年前实现碳中和。")
    b = _chunk("中国承诺在2030年前实现碳达峰，努力争取2060年前实现碳中和")  # missing trailing punctuation
    c = _chunk("证券投资分析报告包含风险评估内容。")  # unrelated, kept

    result = dd.dedup([a, b, c])
    assert len(result) == 2
    assert result[0].text == a.text  # keeps first
    assert result[1].text == c.text


def test_dedup_keeps_all_when_distinct():
    dd = Deduplicator(similarity_threshold=0.85)
    chunks = [
        _chunk("碳达峰是指二氧化碳排放达到峰值。"),
        _chunk("碳中和是指净零排放。"),
        _chunk("能源转型涉及电力系统改革。"),
    ]
    assert len(dd.dedup(chunks)) == 3


def test_dedup_preserves_order():
    dd = Deduplicator(similarity_threshold=0.85)
    a = _chunk("内容甲完全不同。")
    b = _chunk("内容乙完全不同。")
    c = _chunk("内容丙完全不同。")
    result = dd.dedup([a, b, c])
    assert [c.text for c in result] == ["内容甲完全不同。", "内容乙完全不同。", "内容丙完全不同。"]


def test_dedup_empty():
    dd = Deduplicator()
    assert dd.dedup([]) == []


def test_dedup_threshold_respected():
    """Lower threshold removes more; higher threshold removes less."""
    a = _chunk("碳达峰是指排放达到峰值之后逐步下降。")
    b = _chunk("碳达峰是指排放达到峰值之后逐步下降的目标。")
    assert len(Deduplicator(similarity_threshold=0.95).dedup([a, b])) == 2
    assert len(Deduplicator(similarity_threshold=0.5).dedup([a, b])) == 1


def test_dedup_single_chunk():
    dd = Deduplicator()
    result = dd.dedup([_chunk("唯一一个chunk")])
    assert len(result) == 1
