"""Tests for the LangGraph orchestration — conditional-edge branch selection.

Tests the routing functions (_route_after_router, _route_after_retrieve)
directly: these are the heart of Modular RAG's runtime dynamic routing.
Also exercises the full compiled graph with all heavy nodes monkey-patched
to stubs, verifying each route traverses the expected node sequence.
"""

from src.chunker import Chunk


# ── _route_after_router ──────────────────────────────────────


def _make_state(**kw):
    return kw


def test_route_after_router_simple():
    from src.graph import _route_after_router

    assert _route_after_router({"route": "simple"}) == "simple"


def test_route_after_router_complex():
    from src.graph import _route_after_router

    assert _route_after_router({"route": "complex"}) == "complex"


def test_route_after_router_time_sensitive():
    from src.graph import _route_after_router

    assert _route_after_router({"route": "time_sensitive"}) == "time_sensitive"


def test_route_after_router_defaults_to_simple():
    from src.graph import _route_after_router

    assert _route_after_router({}) == "simple"


# ── _route_after_retrieve (relevance gate) ────────────────────


def _chunk(distance=None):
    return Chunk(text="x", metadata={"distance": distance} if distance is not None else {})


def _set_config(threshold=0.5):
    import tempfile

    import yaml

    import src.config as cfg_mod
    from src.config import Config

    d = {
        "llm": {"model": "m", "base_url": "u", "api_key": "${SILICONFLOW_API_KEY}"},
        "embedding": {"model": "m", "base_url": "u", "api_key": "${SILICONFLOW_API_KEY}"},
        "reranker": {"model": "r", "base_url": "u", "api_key": "${SILICONFLOW_API_KEY}"},
        "chunker": {"chunk_size": 500, "chunk_overlap": 50},
        "retriever": {"top_k": 5, "relevance_threshold": threshold},
        "store": {"type": "chromadb", "persist_directory": "x", "collection_name": "c"},
        "modular": {"router": {"enabled": True}},
    }
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(d, f)
    f.close()
    cfg_mod._config = Config(f.name)


def test_relevance_gate_empty_goes_to_generate():
    _set_config()
    from src.graph import _route_after_retrieve

    assert _route_after_retrieve({"retrieved_chunks": []}) == "generate"


def test_relevance_gate_irrelevant_goes_to_generate():
    _set_config(threshold=0.5)
    from src.graph import _route_after_retrieve

    # distance 0.9 > 0.5 → irrelevant
    state = {"retrieved_chunks": [_chunk(0.9)], "route": "complex"}
    assert _route_after_retrieve(state) == "generate"


def test_relevance_gate_relevant_simple_goes_passthrough():
    _set_config(threshold=0.5)
    from src.graph import _route_after_retrieve

    state = {"retrieved_chunks": [_chunk(0.2)], "route": "simple"}
    assert _route_after_retrieve(state) == "context_passthrough"


def test_relevance_gate_relevant_complex_goes_to_rank():
    _set_config(threshold=0.5)
    from src.graph import _route_after_retrieve

    state = {"retrieved_chunks": [_chunk(0.2)], "route": "complex"}
    assert _route_after_retrieve(state) == "rank"


# ── Full graph traversal with stubbed nodes ─────────────────


def test_graph_traverses_complex_path(monkeypatch):
    """complex route: router → query_transform → retrieve → rank → context_process → generate."""
    import src.graph as graph_mod

    visited: list[str] = []

    def _wrap(name, ret):
        def _node(state):
            visited.append(name)
            return ret
        return _node

    monkeypatch.setattr(graph_mod, "router_node", _wrap("router", {"route": "complex", "reason": "r"}))
    monkeypatch.setattr(graph_mod, "query_transform_node", _wrap("query_transform", {"queries": ["q1", "q2"]}))
    c = Chunk(text="doc", metadata={"distance": 0.1, "source": "a", "page": 1})
    monkeypatch.setattr(graph_mod, "retrieve_node", _wrap("retrieve", {"retrieved_chunks": [c]}))
    monkeypatch.setattr(graph_mod, "rank_node", _wrap("rank", {"ranked_chunks": [c]}))
    monkeypatch.setattr(graph_mod, "context_process_node", _wrap("context_process", {"context": [c]}))
    monkeypatch.setattr(graph_mod, "generate_node", _wrap("generate", {"answer": "A", "sources": []}))
    # Rebuild graph with the patched nodes
    monkeypatch.setattr(graph_mod, "_GRAPH", None)
    _set_config()

    g = graph_mod.build_graph()
    result = g.invoke({"query": "对比三份报告能源转型的异同"})

    assert visited == [
        "router",
        "query_transform",
        "retrieve",
        "rank",
        "context_process",
        "generate",
    ]
    assert result["answer"] == "A"


def test_graph_traverses_simple_path(monkeypatch):
    """simple route: router → retrieve → context_passthrough → generate."""
    import src.graph as graph_mod

    visited: list[str] = []

    def _wrap(name, ret):
        def _node(state):
            visited.append(name)
            return ret
        return _node

    monkeypatch.setattr(graph_mod, "router_node", _wrap("router", {"route": "simple", "reason": "r"}))
    c = Chunk(text="doc", metadata={"distance": 0.1, "source": "a", "page": 1})
    monkeypatch.setattr(graph_mod, "retrieve_node", _wrap("retrieve", {"retrieved_chunks": [c]}))
    monkeypatch.setattr(graph_mod, "_context_passthrough", _wrap("context_passthrough", {"context": [c]}))
    monkeypatch.setattr(graph_mod, "generate_node", _wrap("generate", {"answer": "A", "sources": []}))
    monkeypatch.setattr(graph_mod, "_GRAPH", None)
    _set_config()

    g = graph_mod.build_graph()
    result = g.invoke({"query": "碳达峰的目标年份？"})

    assert visited == ["router", "retrieve", "context_passthrough", "generate"]
    assert result["answer"] == "A"


def test_graph_traverses_web_path(monkeypatch):
    """time_sensitive route: router → web_search → generate."""
    import src.graph as graph_mod

    visited: list[str] = []

    def _wrap(name, ret):
        def _node(state):
            visited.append(name)
            return ret
        return _node

    monkeypatch.setattr(graph_mod, "router_node", _wrap("router", {"route": "time_sensitive", "reason": "r"}))
    monkeypatch.setattr(graph_mod, "web_search_node", _wrap("web_search", {"context": []}))
    monkeypatch.setattr(graph_mod, "generate_node", _wrap("generate", {"answer": "A", "sources": []}))
    monkeypatch.setattr(graph_mod, "_GRAPH", None)
    _set_config()

    g = graph_mod.build_graph()
    result = g.invoke({"query": "今天有什么新闻"})

    assert visited == ["router", "web_search", "generate"]
    assert result["answer"] == "A"


def test_graph_relevance_gate_skips_rank_on_irrelevant(monkeypatch):
    """simple route + irrelevant results → free-chat generate, no rank."""
    import src.graph as graph_mod

    visited: list[str] = []

    def _wrap(name, ret):
        def _node(state):
            visited.append(name)
            return ret
        return _node

    monkeypatch.setattr(graph_mod, "router_node", _wrap("router", {"route": "complex", "reason": "r"}))
    c = Chunk(text="doc", metadata={"distance": 0.9, "source": "a", "page": 1})
    monkeypatch.setattr(graph_mod, "retrieve_node", _wrap("retrieve", {"retrieved_chunks": [c]}))
    monkeypatch.setattr(graph_mod, "rank_node", _wrap("rank", {"ranked_chunks": []}))  # should NOT be visited
    monkeypatch.setattr(graph_mod, "generate_node", _wrap("generate", {"answer": "freechat", "sources": []}))
    monkeypatch.setattr(graph_mod, "_GRAPH", None)
    _set_config(threshold=0.5)

    g = graph_mod.build_graph()
    result = g.invoke({"query": "今天天气怎么样"})

    assert "rank" not in visited
    assert visited[-1] == "generate"
    assert result["answer"] == "freechat"
