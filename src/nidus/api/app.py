"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from nidus import __version__
from nidus.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the vector store / collection on startup so the first request is
    # not penalised. Kept best-effort: a missing LLM key shouldn't stop the
    # server from booting (health + ingest may still be useful).
    try:
        from nidus.vectorstore import ensure_collection

        ensure_collection()
    except Exception:  # pragma: no cover - startup is best-effort
        pass
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Nidus",
        description="Light Extensible Agentic RAG built on LangGraph",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
