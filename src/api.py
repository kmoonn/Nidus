"""FastAPI server for Nidus Modular RAG — REST API for indexing and Q&A."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .chunker import chunk_documents
from .config import get_config
from .embedder import Embedder
from .graph import ask as graph_ask, reset_graph_cache
from .loader import load_directory
from .store import Store


# ── Shared state ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup (graph is lazy)."""
    yield


app = FastAPI(
    title="Nidus Modular RAG",
    description="Light Extensible Agentic RAG — LangGraph 动态路由模块化编排",
    version="0.2.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None  # reserved — Modular 路由决定检索深度


class SourceInfo(BaseModel):
    source: str
    page: int | str
    distance: float | None = None
    rerank_score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    route: str | None = None
    sources: list[SourceInfo] = []


class IndexRequest(BaseModel):
    strategy: str = "fixed"  # Modular 仅支持 fixed（语义分块见后续演进）


class IndexResponse(BaseModel):
    pages: int
    chunks: int
    indexed: int
    strategy: str


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
async def build_index(req: IndexRequest | None = None):
    """Build or rebuild the document index from PDFs in docs/."""
    config = get_config()

    # Load
    documents = load_directory(config.docs_dir)
    if not documents:
        return IndexResponse(pages=0, chunks=0, indexed=0, strategy="fixed")

    # Chunk (fixed-size, Modular 阶段仅此策略)
    chunker_cfg = config.chunker
    chunks = chunk_documents(
        documents,
        chunk_size=chunker_cfg["chunk_size"],
        chunk_overlap=chunker_cfg["chunk_overlap"],
    )

    # Embed (batched)
    embedder = Embedder()
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        all_embeddings.extend(embedder.embed_texts(texts))

    # Store
    store = Store()
    store.reset()
    store.add_documents(chunks, all_embeddings)

    # Drop the compiled graph + LLM singletons so the next /ask rebuilds them
    # with fresh collection references (reset() deleted the old collection).
    reset_graph_cache()

    return IndexResponse(
        pages=len(documents),
        chunks=len(chunks),
        indexed=store.count,
        strategy="fixed",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Ask a question — routed dynamically by the Modular RAG graph."""
    result = graph_ask(req.question)
    return AskResponse(
        question=result.query,
        answer=result.answer,
        route=result.route,
        sources=[
            SourceInfo(
                source=s["source"],
                page=s["page"],
                distance=s.get("distance"),
                rerank_score=s.get("rerank_score"),
            )
            for s in result.sources
        ],
    )


# ── Static files (chat UI) ───────────────────────────────────
app.mount(
    "/",
    StaticFiles(
        directory=str(Path(__file__).parent.parent / "static"), html=True
    ),
    name="static",
)
