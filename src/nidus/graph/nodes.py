"""Graph nodes for the agentic RAG workflow.

Each node is produced by a ``make_*`` factory bound to its dependencies (a chat
model and/or a retriever). This keeps nodes free of global state and lets tests
inject fakes without monkeypatching module internals.

Decision nodes (router, graders) use :func:`classify`, a provider-portable
single-word classifier that caps the response length and parses leniently, so
it degrades gracefully instead of raising on models that lack strict
structured-output support.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from nidus import prompts
from nidus.graph.state import AgentState

Node = Callable[[AgentState], dict[str, Any]]


def _format_docs(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(documents)
    )


def classify(
    chat_model: BaseChatModel,
    messages: Sequence[BaseMessage],
    choices: Sequence[str],
    default: str,
) -> str:
    """Return whichever of ``choices`` the model's reply matches.

    Portable across providers: it caps generation length when the model
    supports it and matches leniently (case-insensitive substring), falling
    back to ``default`` if nothing matches or the call fails. This avoids the
    unbounded-generation / parse errors seen with strict structured output on
    smaller OpenAI-compatible models.
    """

    model = chat_model
    # Cap output for these one-word decisions when the client supports it.
    bind = getattr(chat_model, "bind", None)
    if callable(bind):
        try:
            model = chat_model.bind(max_tokens=8)
        except Exception:
            model = chat_model

    try:
        raw = model.invoke(list(messages))
        text = (getattr(raw, "content", "") or "").strip().lower()
    except Exception:
        return default

    for choice in choices:
        if choice.lower() in text:
            return choice
    return default


def make_route_node(chat_model: BaseChatModel, *, enable_direct: bool) -> Node:
    """Route the question to the vectorstore or a direct answer."""

    def route_question(state: AgentState) -> dict[str, Any]:
        if not enable_direct:
            return {"route": "vectorstore", "trace": ["route -> vectorstore (forced)"]}
        route = classify(
            chat_model,
            [
                SystemMessage(content=prompts.ROUTER_SYSTEM),
                HumanMessage(content=state["question"]),
            ],
            choices=["vectorstore", "direct"],
            default="vectorstore",  # when unsure, prefer grounded retrieval
        )
        return {"route": route, "trace": [f"route -> {route}"]}

    return route_question


def make_retrieve_node(retriever) -> Node:
    """Fetch candidate documents for the current question."""

    def retrieve(state: AgentState) -> dict[str, Any]:
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "trace": [f"retrieve -> {len(docs)} docs"]}

    return retrieve


def make_grade_documents_node(chat_model: BaseChatModel) -> Node:
    """Keep only documents the grader deems relevant to the question."""

    def grade_documents(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        kept: list[Document] = []
        for doc in state.get("documents", []):
            verdict = classify(
                chat_model,
                [
                    SystemMessage(content=prompts.DOC_GRADER_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Retrieved document:\n{doc.page_content}\n\n"
                            f"User question: {question}"
                        )
                    ),
                ],
                choices=["yes", "no"],
                default="yes",  # when unsure, keep the doc rather than drop it
            )
            if verdict == "yes":
                kept.append(doc)
        return {
            "documents": kept,
            "trace": [f"grade_documents -> {len(kept)} relevant"],
        }

    return grade_documents


def make_transform_query_node(chat_model: BaseChatModel) -> Node:
    """Rewrite the question to improve retrieval, counting the attempt."""

    def transform_query(state: AgentState) -> dict[str, Any]:
        result = chat_model.invoke(
            [
                SystemMessage(content=prompts.REWRITE_SYSTEM),
                HumanMessage(content=f"Initial question:\n{state['question']}"),
            ]
        )
        rewritten = (result.content or "").strip() or state["question"]
        retries = state.get("retries", 0) + 1
        return {
            "question": rewritten,
            "retries": retries,
            "trace": [f"transform_query (retry {retries}) -> {rewritten!r}"],
        }

    return transform_query


def make_generate_node(chat_model: BaseChatModel) -> Node:
    """Generate a grounded answer from the retrieved context."""

    def generate(state: AgentState) -> dict[str, Any]:
        documents = state.get("documents", [])
        prompt = prompts.GENERATE_SYSTEM.format(
            question=state["question"], context=_format_docs(documents)
        )
        result = chat_model.invoke([HumanMessage(content=prompt)])
        attempts = state.get("gen_attempts", 0) + 1
        return {
            "generation": result.content,
            "gen_attempts": attempts,
            "trace": [f"generate (attempt {attempts})"],
        }

    return generate


def make_generate_direct_node(chat_model: BaseChatModel) -> Node:
    """Answer without retrieval (greetings / general chit-chat)."""

    def generate_direct(state: AgentState) -> dict[str, Any]:
        result = chat_model.invoke(
            [
                SystemMessage(content=prompts.DIRECT_SYSTEM),
                HumanMessage(content=state["question"]),
            ]
        )
        return {"generation": result.content, "trace": ["generate_direct"]}

    return generate_direct
