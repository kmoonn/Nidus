"""Assemble the agentic RAG graph from nodes and edges.

Graph shape (self-correcting / adaptive RAG)::

    START -> route
      route --direct-------> generate_direct -> END
      route --vectorstore--> retrieve -> grade_documents -> decide:
          has relevant docs         -> generate
          none & retries < max      -> transform_query -> retrieve (loop)
          none & retries exhausted  -> generate (best effort)
    transform_query -> retrieve
    generate -> grade_generation:
          not_supported (hallucination) -> generate
          not_useful (off-topic)        -> transform_query
          useful                        -> END
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from nidus.config import get_settings
from nidus.graph import edges, nodes
from nidus.graph.state import AgentState


def build_graph(
    *,
    chat_model: BaseChatModel,
    retriever: BaseRetriever,
    max_retries: int = 2,
    enable_direct: bool = True,
):
    """Build and compile the agentic RAG graph.

    Dependencies are injected so the same builder serves production (real LLM +
    Qdrant) and tests (fakes), with no hidden globals.
    """

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("route", nodes.make_route_node(chat_model, enable_direct=enable_direct))
    graph.add_node("retrieve", nodes.make_retrieve_node(retriever))
    graph.add_node("grade_documents", nodes.make_grade_documents_node(chat_model))
    graph.add_node("transform_query", nodes.make_transform_query_node(chat_model))
    graph.add_node("generate", nodes.make_generate_node(chat_model))
    graph.add_node("generate_direct", nodes.make_generate_direct_node(chat_model))

    # Routing out of START
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        edges.route_after_routing,
        {"vectorstore": "retrieve", "direct": "generate_direct"},
    )

    # Retrieval -> grading -> decision
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        edges.make_decide_to_generate(max_retries),
        {"generate": "generate", "transform_query": "transform_query"},
    )
    graph.add_edge("transform_query", "retrieve")

    # Generation grading -> loop or finish
    graph.add_conditional_edges(
        "generate",
        edges.make_grade_generation(chat_model, max_retries),
        {
            "useful": END,
            "not_supported": "generate",
            "not_useful": "transform_query",
        },
    )
    graph.add_edge("generate_direct", END)

    return graph.compile()


@lru_cache
def get_graph():
    """Return the cached production graph wired to config-driven deps."""

    # Imported lazily so importing the graph package doesn't require Qdrant.
    from nidus.models import get_chat_model
    from nidus.vectorstore import get_retriever

    settings = get_settings()
    return build_graph(
        chat_model=get_chat_model(),
        retriever=get_retriever(),
        max_retries=settings.max_retries,
        enable_direct=settings.enable_direct_answer,
    )
