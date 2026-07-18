"""LangGraph orchestration for Nidus Modular RAG.

图结构：
    query → Router →(条件边 by route)
        ├─ simple        → Retrieve →(相关度门控) → Generate
        ├─ complex        → QueryTransform → Retrieve → Rank → ContextProcess → Generate
        └─ time_sensitive → WebSearch → Generate

动态路由（Router 运行时分类）+ 相关度门控（Retrieve 后判 distance）是 Modular
相对 Advanced（静态开关线性流水线）的核心区别。
"""

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .nodes.context_process import context_process_node
from .nodes.generate import generate_node
from .nodes.query_transform import query_transform_node
from .nodes.rank import rank_node
from .nodes.retrieve import retrieve_node
from .nodes.router import (
    ROUTE_COMPLEX,
    ROUTE_SIMPLE,
    ROUTE_TIME_SENSITIVE,
    router_node,
)
from .nodes.web_search import web_search_node

# 相关度门控：最佳结果 distance 超过此阈值视为与文档无关 → 跳过 Rank，空 context 直达 Generate。
DEFAULT_RELEVANCE_THRESHOLD = 0.50


class GraphState(TypedDict, total=False):
    """State flowing through the Modular RAG graph."""

    query: str
    route: str
    reason: str
    queries: list[str]
    retrieved_chunks: list
    ranked_chunks: list
    context: list
    answer: str
    sources: list


@dataclass
class Answer:
    """Result of a RAG query (mirrors dev/advanced Pipeline Answer)."""

    query: str
    answer: str
    route: str = ROUTE_SIMPLE
    reason: str = ""
    sources: list = field(default_factory=list)


def _route_after_router(state: GraphState) -> str:
    """Conditional edge: pick path by router's route label."""
    route = state.get("route", ROUTE_SIMPLE)
    if route == ROUTE_COMPLEX:
        return ROUTE_COMPLEX
    if route == ROUTE_TIME_SENSITIVE:
        return ROUTE_TIME_SENSITIVE
    return ROUTE_SIMPLE


def _route_after_retrieve(state: GraphState) -> str:
    """Relevance gate: empty/too-distant results → skip rank, free-chat.

    simple 路径：相关 → context_process（仅截断，无压缩）→ generate；不相关 → generate。
    complex 路径：相关 → rank；不相关 → generate。
    """
    from .config import get_config

    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        return "generate"

    threshold = get_config().retriever.get(
        "relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD
    )
    best_distance = chunks[0].metadata.get("distance", 1.0)
    if best_distance is None or best_distance > threshold:
        return "generate"
    # 相关 → 按路由决定后续：complex 走深度后处理，simple 直接组装 context。
    if state.get("route") == ROUTE_COMPLEX:
        return "rank"
    return "context_passthrough"


def _context_passthrough(state: GraphState) -> GraphState:
    """simple 路径：检索结果直接作为 context（不做 rerank/压缩）。"""
    chunks = state.get("retrieved_chunks") or []
    return {"context": list(chunks)}


def build_graph():
    """Build and compile the Modular RAG LangGraph."""
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("query_transform", query_transform_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rank", rank_node)
    graph.add_node("context_process", context_process_node)
    graph.add_node("context_passthrough", _context_passthrough)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("router")

    # Router → path selection
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            ROUTE_SIMPLE: "retrieve",
            ROUTE_COMPLEX: "query_transform",
            ROUTE_TIME_SENSITIVE: "web_search",
        },
    )

    # complex path: query_transform → retrieve
    graph.add_edge("query_transform", "retrieve")

    # simple/complex converge at retrieve → relevance gate
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {
            "rank": "rank",
            "context_passthrough": "context_passthrough",
            "generate": "generate",
        },
    )

    # complex post-retrieval: rank → context_process → generate
    graph.add_edge("rank", "context_process")
    graph.add_edge("context_process", "generate")

    # simple 路径相关结果：context_passthrough 组装 context 后直达 generate
    graph.add_edge("context_passthrough", "generate")

    # web path: web_search → generate
    graph.add_edge("web_search", "generate")

    graph.add_edge("generate", END)

    return graph.compile()


# Compiled graph singleton (rebuilt on re-index via reset_graph_cache)
_GRAPH = None


def get_graph():
    """Return the compiled graph singleton."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def reset_graph_cache() -> None:
    """Drop the compiled graph + LLM singletons (used on re-index / in tests)."""
    from .llm import reset_llm_cache

    global _GRAPH
    _GRAPH = None
    reset_llm_cache()


def ask(query: str) -> Answer:
    """Run the full Modular RAG graph for a query and return an Answer."""
    result = get_graph().invoke({"query": query})
    return Answer(
        query=query,
        answer=result.get("answer", ""),
        route=result.get("route", ROUTE_SIMPLE),
        reason=result.get("reason", ""),
        sources=result.get("sources") or [],
    )
