"""Pydantic schemas shared across the API, CLI and graph boundaries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single retrieved chunk surfaced as an answer citation."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class IngestTextRequest(BaseModel):
    """Ingest raw text passed directly in the request body."""

    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestPathRequest(BaseModel):
    """Ingest a local file or directory reachable by the server."""

    path: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    chunks: int
    documents: int
    collection: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    include_sources: bool = True
    include_trace: bool = False


class QueryResponse(BaseModel):
    answer: str
    route: str
    sources: list[Source] = Field(default_factory=list)
    retries: int = 0
    trace: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class HealthResponse(BaseModel):
    status: str
    version: str
    collection: str
    vectorstore: str
    llm_model: str
