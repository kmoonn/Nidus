"""Conditional-edge logic for the agentic RAG graph.

These functions inspect :class:`AgentState` and return the *name* of the next
node. The retry counter (bounded by ``max_retries``) guarantees termination of
the query-rewrite / regenerate loops.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from nidus import prompts
from nidus.graph.nodes import _format_docs, classify
from nidus.graph.state import AgentState

Router = Callable[[AgentState], str]


def route_after_routing(state: AgentState) -> str:
    """Send the question to retrieval or a direct answer."""

    return "vectorstore" if state.get("route") == "vectorstore" else "direct"


def make_decide_to_generate(max_retries: int) -> Router:
    """After grading docs: generate, rewrite the query, or give up."""

    def decide_to_generate(state: AgentState) -> str:
        documents = state.get("documents", [])
        if documents:
            return "generate"
        if state.get("retries", 0) < max_retries:
            return "transform_query"
        # Out of retries with no relevant docs: answer honestly anyway.
        return "generate"

    return decide_to_generate


def make_grade_generation(chat_model: BaseChatModel, max_retries: int) -> Router:
    """Check the generation for grounding and answer-relevance.

    Returns one of ``"useful"`` (END), ``"not_supported"`` (regenerate) or
    ``"not_useful"`` (rewrite query and retrieve again).
    """

    def grade_generation(state: AgentState) -> str:
        # No documents (or retries exhausted): accept the best-effort answer to
        # avoid looping forever.
        documents = state.get("documents", [])
        if not documents or state.get("retries", 0) >= max_retries:
            return "useful"

        # Bound the regeneration ("not_supported") loop independently of the
        # query-rewrite loop: allow at most max_retries + 1 generate calls.
        gen_exhausted = state.get("gen_attempts", 0) > max_retries

        facts = _format_docs(documents)
        grounded = classify(
            chat_model,
            [
                SystemMessage(content=prompts.HALLUCINATION_SYSTEM),
                HumanMessage(
                    content=(
                        f"Set of facts:\n{facts}\n\n"
                        f"LLM generation: {state.get('generation', '')}"
                    )
                ),
            ],
            choices=["yes", "no"],
            default="yes",  # when unsure, assume grounded and stop looping
        )
        if grounded != "yes":
            return "useful" if gen_exhausted else "not_supported"

        addresses = classify(
            chat_model,
            [
                SystemMessage(content=prompts.ANSWER_GRADER_SYSTEM),
                HumanMessage(
                    content=(
                        f"User question: {state['question']}\n\n"
                        f"LLM generation: {state.get('generation', '')}"
                    )
                ),
            ],
            choices=["yes", "no"],
            default="yes",
        )
        return "useful" if addresses == "yes" else "not_useful"

    return grade_generation
