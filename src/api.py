"""FastAPI server for Nidus RAG — REST API for indexing and Q&A."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .chunker import chunk_documents
from .config import get_config
from .embedder import Embedder
from .loader import load_directory
from .pipeline import Pipeline
from .store import Store


# ── Shared state ──────────────────────────────────────────────
_pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    global _pipeline
    _pipeline = Pipeline()
    yield


app = FastAPI(
    title="Nidus RAG",
    description="Light Extensible Agentic RAG — API",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceInfo(BaseModel):
    source: str
    page: int | str
    distance: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo] = []


class IndexResponse(BaseModel):
    pages: int
    chunks: int
    indexed: int


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int


# ── Endpoints ─────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns status and number of indexed chunks."""
    store = Store()
    return HealthResponse(status="ok", chunks_indexed=store.count)


@app.post("/index", response_model=IndexResponse)
async def build_index():
    """Build or rebuild the document index from PDFs in docs/."""
    config = get_config()

    # Load
    documents = load_directory(config.docs_dir)
    if not documents:
        return IndexResponse(pages=0, chunks=0, indexed=0)

    # Chunk
    chunker_cfg = config.chunker
    chunks = chunk_documents(
        documents,
        chunk_size=chunker_cfg["chunk_size"],
        chunk_overlap=chunker_cfg["chunk_overlap"],
    )

    # Embed
    embedder = Embedder()
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = embedder.embed_texts(texts)
        all_embeddings.extend(embeddings)

    # Store
    store = Store()
    store.reset()
    store.add_documents(chunks, all_embeddings)

    return IndexResponse(
        pages=len(documents),
        chunks=len(chunks),
        indexed=store.count,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Ask a question and get an answer from indexed documents."""
    result = _pipeline.ask(req.question)
    return AskResponse(
        question=result.query,
        answer=result.answer,
        sources=[
            SourceInfo(
                source=s["source"],
                page=s["page"],
                distance=s.get("distance"),
            )
            for s in result.sources
        ],
    )
