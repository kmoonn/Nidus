"""Tests for the agentic RAG graph: routing, grading loops and retry bounds."""

from __future__ import annotations

from nidus.graph.builder import build_graph
from nidus.service import answer_question
from tests.conftest import FakeChatModel, FakeRetriever


def _run(graph, question: str) -> dict:
    return graph.invoke(
        {
            "question": question,
            "original_question": question,
            "retries": 0,
            "documents": [],
            "trace": [],
        }
    )


def test_vectorstore_route_happy_path(sample_docs):
    chat = FakeChatModel(answer="Grounded answer.")
    retriever = FakeRetriever(sample_docs)
    graph = build_graph(chat_model=chat, retriever=retriever, max_retries=2)

    state = _run(graph, "What is Nidus?")

    assert state["generation"] == "Grounded answer."
    assert state["route"] == "vectorstore"
    assert len(state["documents"]) == 2
    assert retriever.calls == 1  # no rewrite loop needed
    assert any("retrieve" in t for t in state["trace"])


def test_direct_route_skips_retrieval():
    chat = FakeChatModel(route="direct", answer="Hi there!")
    retriever = FakeRetriever([])
    graph = build_graph(chat_model=chat, retriever=retriever, max_retries=2)

    state = _run(graph, "hello")

    assert state["route"] == "direct"
    assert state["generation"] == "Hi there!"
    assert retriever.calls == 0
    assert any("generate_direct" in t for t in state["trace"])


def test_irrelevant_docs_trigger_rewrite_then_stop(sample_docs):
    """Grader says 'no' to every doc -> rewrite loop -> bounded by max_retries."""

    chat = FakeChatModel(doc_grade="no", answer="Best-effort answer.")
    retriever = FakeRetriever(sample_docs)
    graph = build_graph(chat_model=chat, retriever=retriever, max_retries=2)

    state = _run(graph, "unanswerable question")

    # Initial retrieve + 2 rewrite-driven retrieves = 3 total, then it stops.
    assert retriever.calls == 3
    assert state["retries"] == 2
    assert state["generation"] == "Best-effort answer."


def test_rewrite_updates_question_but_keeps_original(sample_docs):
    calls = {"n": 0}

    def doc_grade(_messages):
        # Irrelevant on the first pass, relevant after one rewrite.
        calls["n"] += 1
        return "no" if calls["n"] <= 2 else "yes"

    chat = FakeChatModel(doc_grade=doc_grade, rewrite="rewritten query")
    retriever = FakeRetriever(sample_docs)
    graph = build_graph(chat_model=chat, retriever=retriever, max_retries=3)

    state = _run(graph, "original question")

    assert state["original_question"] == "original question"
    assert state["question"] == "rewritten query"
    assert state["retries"] == 1


def test_hallucination_loop_is_bounded(sample_docs):
    """A grader that always says 'not grounded' must not loop forever."""

    # hallucination='no' would send generate -> not_supported -> generate...
    # The gen_attempts bound (max_retries + 1) must break out to 'useful'.
    chat = FakeChatModel(hallucination="no", answer="Ungroundable answer.")
    retriever = FakeRetriever(sample_docs)
    graph = build_graph(chat_model=chat, retriever=retriever, max_retries=2)

    state = graph.invoke(
        {
            "question": "q",
            "original_question": "q",
            "retries": 0,
            "gen_attempts": 0,
            "documents": [],
            "trace": [],
        },
        config={"recursion_limit": 25},
    )

    # generate runs at most max_retries + 1 = 3 times, then accepts.
    assert state["gen_attempts"] <= 3
    assert state["generation"] == "Ungroundable answer."


def test_answer_question_service_shapes_response(sample_docs):
    chat = FakeChatModel(answer="Service answer.")
    retriever = FakeRetriever(sample_docs)
    graph = build_graph(chat_model=chat, retriever=retriever, max_retries=1)

    resp = answer_question("What is LangGraph?", graph=graph)

    assert resp.answer == "Service answer."
    assert resp.route == "vectorstore"
    assert len(resp.sources) == 2
    assert resp.sources[0].metadata["source"] in {"doc1", "doc2"}
    assert resp.trace  # decision trace populated


def test_max_retries_zero_never_rewrites(sample_docs):
    chat = FakeChatModel(doc_grade="no")
    retriever = FakeRetriever(sample_docs)
    graph = build_graph(chat_model=chat, retriever=retriever, max_retries=0)

    state = _run(graph, "q")

    assert retriever.calls == 1  # no rewrite attempts at all
    assert state["retries"] == 0


def test_classify_falls_back_to_default_on_error(sample_docs):
    """A model that raises on decision calls should not crash the graph."""

    class ExplodingChat(FakeChatModel):
        def invoke(self, messages):
            from tests.conftest import _detect_kind

            kind = _detect_kind(messages)
            if kind in {"route", "doc_grade", "hallucination", "answer_grade"}:
                raise RuntimeError("provider blew up on structured decision")
            return super().invoke(messages)

    chat = ExplodingChat(answer="Recovered answer.")
    retriever = FakeRetriever(sample_docs)
    graph = build_graph(chat_model=chat, retriever=retriever, max_retries=1)

    state = _run(graph, "What is Nidus?")

    # route defaults to 'vectorstore', doc_grade defaults to 'yes' (keep docs).
    assert state["route"] == "vectorstore"
    assert state["generation"] == "Recovered answer."
    assert len(state["documents"]) == 2
