"""Shared graph state for the agentic RAG workflow."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """State threaded through every node of the graph.

    Only ``question`` is required as input; the rest are populated as the
    graph runs and are surfaced in the final :class:`QueryResponse`.
    """

    question: str          # current (possibly rewritten) question
    original_question: str  # the user's initial question, never mutated
    documents: list[Document]
    generation: str
    route: str
    retries: int        # query-rewrite attempts (bounds the retrieval loop)
    gen_attempts: int   # generate calls (bounds the hallucination re-gen loop)
    trace: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
