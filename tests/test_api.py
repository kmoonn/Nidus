"""Tests for the FastAPI endpoints using dependency fakes."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from nidus.api.app import create_app
from nidus.schemas import QueryResponse, Source


def _client() -> TestClient:
    return TestClient(create_app())


def test_health(monkeypatch):
    monkeypatch.delenv("NIDUS_QDRANT_URL", raising=False)
    resp = _client().get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["vectorstore"] == "qdrant-embedded"
    assert "version" in body


def test_query_route(monkeypatch):
    from nidus.api import routes

    fake = QueryResponse(
        answer="42",
        route="vectorstore",
        sources=[Source(content="ctx", metadata={"source": "d"})],
        retries=1,
        trace=["route -> vectorstore"],
    )
    monkeypatch.setattr(routes, "answer_question", lambda q: fake.model_copy(deep=True))

    resp = _client().post("/query", json={"question": "meaning of life?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "42"
    assert body["sources"][0]["content"] == "ctx"
    # include_trace defaults to False -> trace stripped.
    assert body["trace"] == []


def test_query_route_can_include_trace(monkeypatch):
    from nidus.api import routes

    fake = QueryResponse(answer="a", route="direct", trace=["route -> direct"])
    monkeypatch.setattr(routes, "answer_question", lambda q: fake.model_copy(deep=True))

    resp = _client().post(
        "/query", json={"question": "hi", "include_trace": True}
    )
    assert resp.json()["trace"] == ["route -> direct"]


def test_ingest_text_route(monkeypatch):
    from nidus.api import routes

    monkeypatch.setattr(routes, "ingest_text", lambda text, meta: 3)
    resp = _client().post("/ingest/text", json={"text": "hello world"})
    assert resp.status_code == 200
    assert resp.json()["chunks"] == 3


def test_ingest_path_missing_returns_400(monkeypatch):
    from nidus.api import routes

    def _raise(_path):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(routes, "ingest_path", _raise)
    resp = _client().post("/ingest/path", json={"path": "/no/such"})
    assert resp.status_code == 400


def test_chat_streams_sse(monkeypatch):
    from nidus.api import routes

    async def fake_stream(question):
        for tok in ["Hel", "lo"]:
            yield tok

    monkeypatch.setattr(routes, "astream_answer", fake_stream)

    with _client().stream("POST", "/chat", json={"question": "hi"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "Hel" in body and "lo" in body
    assert "done" in body
