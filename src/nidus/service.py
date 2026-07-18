"""High-level query service shared by the API and CLI.

Wraps the compiled graph, turning ``question`` in / ``QueryResponse`` out and
providing a token-streaming helper for chat endpoints.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from nidus.graph.builder import get_graph
from nidus.schemas import QueryResponse, Source


def _state_to_response(state: dict[str, Any]) -> QueryResponse:
    documents = state.get("documents", []) or []
    sources = [
        Source(content=doc.page_content, metadata=doc.metadata)
        for doc in documents
    ]
    return QueryResponse(
        answer=state.get("generation") or "",
        route=state.get("route", "vectorstore"),
        sources=sources,
        retries=state.get("retries", 0),
        trace=state.get("trace", []),
    )


def _initial_state(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "original_question": question,
        "retries": 0,
        "gen_attempts": 0,
        "documents": [],
        "trace": [],
    }


# Hard ceiling on total node steps, as a last-resort guard against loops.
_RECURSION_LIMIT = 25


def answer_question(question: str, graph=None) -> QueryResponse:
    """Run the graph to completion and return a structured answer."""

    graph = graph or get_graph()
    final = graph.invoke(
        _initial_state(question), config={"recursion_limit": _RECURSION_LIMIT}
    )
    return _state_to_response(final)


async def astream_answer(question: str, graph=None) -> AsyncIterator[str]:
    """Yield answer tokens as they are produced by the generation nodes.

    Uses LangGraph's ``messages`` stream mode, which surfaces LLM token chunks
    tagged with the emitting node so we can stream only the user-facing answer
    (``generate`` / ``generate_direct``) and skip grader/router chatter.
    """

    graph = graph or get_graph()
    answer_nodes = {"generate", "generate_direct"}
    async for chunk, metadata in graph.astream(
        _initial_state(question),
        stream_mode="messages",
        config={"recursion_limit": _RECURSION_LIMIT},
    ):
        node = metadata.get("langgraph_node")
        text = getattr(chunk, "content", "")
        if node in answer_nodes and text:
            yield text
