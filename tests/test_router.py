"""Tests for the Router node — LLM-based query classification.

Mocks ChatOpenAI so no real API call is made. Verifies route labels and the
web_search gating (time_sensitive only allowed when web_search.enabled).
"""

import json

import src.config as cfg_mod
from src.config import Config


def _set_config(web_search: bool = False):
    import tempfile

    import yaml

    d = {
        "llm": {"model": "m", "base_url": "u", "api_key": "${SILICONFLOW_API_KEY}"},
        "embedding": {"model": "m", "base_url": "u", "api_key": "${SILICONFLOW_API_KEY}"},
        "reranker": {"model": "r", "base_url": "u", "api_key": "${SILICONFLOW_API_KEY}"},
        "chunker": {"chunk_size": 500, "chunk_overlap": 50},
        "retriever": {"top_k": 5, "relevance_threshold": 0.5},
        "store": {"type": "chromadb", "persist_directory": "x", "collection_name": "c"},
        "modular": {"router": {"enabled": True}},
        "web_search": {"enabled": web_search, "max_results": 5},
    }
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(d, f)
    f.close()
    cfg_mod._config = Config(f.name)


class _FakeLLM:
    """Fake ChatOpenAI: returns a configured structured decision."""

    def __init__(self, parsed):
        self._parsed = parsed

    def bind(self, **kwargs):
        return self  # ignore temperature overrides

    def with_structured_output(self, schema):
        outer = self

        class _Bound:
            def invoke(self, messages):
                return outer._parsed

        return _Bound()


def _patch_llm(monkeypatch, decision: dict):
    import src.llm as llm_mod
    import src.nodes.router as router_mod
    from src.nodes.router import RouteDecision

    parsed = RouteDecision(route=decision["route"], reason=decision["reason"])
    fake = _FakeLLM(parsed)
    monkeypatch.setattr(llm_mod, "_CHAT_LLM", fake)
    monkeypatch.setattr(router_mod, "get_chat_llm", lambda **kw: fake)


def test_router_classifies_simple(monkeypatch):
    _set_config(web_search=False)
    _patch_llm(monkeypatch, {"route": "simple", "reason": "单一事实查询"})
    from src.nodes.router import router_node

    state = router_node({"query": "碳达峰的目标年份？"})
    assert state["route"] == "simple"
    assert "事实" in state["reason"]


def test_router_classifies_complex(monkeypatch):
    _set_config(web_search=False)
    _patch_llm(monkeypatch, {"route": "complex", "reason": "需要对比"})
    from src.nodes.router import router_node

    state = router_node({"query": "对比三份报告能源转型的异同"})
    assert state["route"] == "complex"


def test_router_demotes_time_sensitive_when_web_disabled(monkeypatch):
    """web_search disabled → time_sensitive label normalized to complex."""
    _set_config(web_search=False)
    _patch_llm(monkeypatch, {"route": "time_sensitive", "reason": "实时"})
    from src.nodes.router import router_node

    state = router_node({"query": "今天有什么新闻"})
    # Without web_search, time_sensitive must not survive — routed to complex.
    assert state["route"] == "complex"


def test_router_keeps_time_sensitive_when_web_enabled(monkeypatch):
    _set_config(web_search=True)
    _patch_llm(monkeypatch, {"route": "time_sensitive", "reason": "实时"})
    from src.nodes.router import router_node

    state = router_node({"query": "今天有什么新闻"})
    assert state["route"] == "time_sensitive"


def test_router_falls_back_to_simple_on_error(monkeypatch):
    _set_config(web_search=False)

    class _BrokenLLM:
        def bind(self, **kw):
            return self

        @property
        def with_structured_output(self):
            raise RuntimeError("LLM unavailable")

    import src.llm as llm_mod
    import src.nodes.router as router_mod

    monkeypatch.setattr(llm_mod, "_CHAT_LLM", _BrokenLLM())
    monkeypatch.setattr(router_mod, "get_chat_llm", lambda **kw: _BrokenLLM())
    from src.nodes.router import router_node

    state = router_node({"query": "anything"})
    assert state["route"] == "simple"
