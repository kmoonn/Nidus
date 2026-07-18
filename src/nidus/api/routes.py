"""FastAPI route handlers for Nidus."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from nidus import __version__
from nidus.config import get_settings
from nidus.ingest import ingest_path, ingest_text
from nidus.schemas import (
    ChatRequest,
    HealthResponse,
    IngestPathRequest,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
)
from nidus.service import answer_question, astream_answer
from nidus.vectorstore import vectorstore_mode

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        collection=settings.qdrant_collection,
        vectorstore=vectorstore_mode(),
        llm_model=settings.llm_model,
    )


@router.post("/ingest/text", response_model=IngestResponse)
def ingest_text_route(req: IngestTextRequest) -> IngestResponse:
    chunks = ingest_text(req.text, req.metadata)
    return IngestResponse(
        chunks=chunks, documents=1, collection=get_settings().qdrant_collection
    )


@router.post("/ingest/path", response_model=IngestResponse)
def ingest_path_route(req: IngestPathRequest) -> IngestResponse:
    try:
        docs, chunks = ingest_path(req.path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(
        chunks=chunks, documents=docs, collection=get_settings().qdrant_collection
    )


@router.post("/query", response_model=QueryResponse)
def query_route(req: QueryRequest) -> QueryResponse:
    resp = answer_question(req.question)
    if not req.include_sources:
        resp.sources = []
    if not req.include_trace:
        resp.trace = []
    return resp


@router.post("/chat")
async def chat_route(req: ChatRequest) -> EventSourceResponse:
    """Stream the answer token-by-token as Server-Sent Events.

    Emits ``token`` events for each chunk and a final ``done`` event.
    """

    async def event_generator():
        try:
            async for token in astream_answer(req.question):
                yield {"event": "token", "data": json.dumps({"token": token})}
        except Exception as exc:  # surface errors to the SSE client
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}
        else:
            yield {"event": "done", "data": json.dumps({"done": True})}

    return EventSourceResponse(event_generator())
