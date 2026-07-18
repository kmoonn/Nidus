# Nidus

**Light Extensible Agentic RAG** — a self-correcting retrieval & question-answering
platform built on [LangGraph](https://github.com/langchain-ai/langgraph).

Nidus routes each question, retrieves from a [Qdrant](https://qdrant.tech) vector
store, **grades** the retrieved documents, **rewrites** the query when results are
weak, generates an answer, then **checks the answer** for hallucinations and
relevance — looping until it is confident or a retry budget is exhausted.

Everything is **config-driven**: the LLM and embeddings are any OpenAI-compatible
endpoint (OpenAI, Doubao/Ark, DeepSeek, Ollama, vLLM…), and Qdrant runs embedded
(zero infra) or as a server.

---

## Architecture

```
                       ┌─────────────┐
      question ───────▶│    route    │
                       └──────┬──────┘
              direct │        │ vectorstore
             ┌────────▼──┐    ▼
             │ generate_ │  ┌──────────┐   ┌────────────────┐
             │  direct   │  │ retrieve │──▶│ grade_documents│
             └─────┬─────┘  └────▲─────┘   └───────┬────────┘
                   │             │ loop            │ relevant? / retries
                   │      ┌──────┴────────┐  none  │
                   │      │ transform_    │◀───────┤
                   │      │   query       │        │ yes / exhausted
                   │      └───────────────┘        ▼
                   │                          ┌──────────┐
                   │        not_supported ◀───│ generate │
                   │        not_useful ──────▶ (rewrite) └────┬─────┘
                   │                                          │ useful
                   └──────────────────────────────────────▶ END
```

| Module | Responsibility |
| --- | --- |
| `nidus.config` | `Settings` from env / `.env` (prefix `NIDUS_`) |
| `nidus.models` | OpenAI-compatible chat + embeddings factories |
| `nidus.vectorstore` | Qdrant client, collection, retriever (embedded/server) |
| `nidus.ingest` | load (txt/md/pdf/dir) → split → embed → upsert |
| `nidus.graph` | LangGraph state, nodes, edges, builder |
| `nidus.service` | invoke graph → `QueryResponse`; token streaming |
| `nidus.api` | FastAPI app + routes (`/health`, `/ingest`, `/query`, `/chat`) |
| `nidus.cli` | Typer CLI (`init-db`, `ingest`, `ask`, `serve`, `info`) |

## Install

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # install runtime + dev dependencies
cp .env.example .env          # then edit .env with your provider details
```

## Configure

Edit `.env` (see `.env.example` for all keys and provider presets). Minimum:

```bash
NIDUS_LLM_BASE_URL=https://api.openai.com/v1
NIDUS_LLM_API_KEY=sk-...
NIDUS_LLM_MODEL=gpt-4o-mini
NIDUS_EMBED_MODEL=text-embedding-3-small
NIDUS_EMBED_DIM=1536          # MUST match the embedding model's dimensionality
```

> **`NIDUS_EMBED_DIM` must match your embedding model** (e.g. `text-embedding-3-small`
> = 1536, `-3-large` = 3072, SiliconFlow `BAAI/bge-m3` = 1024, Ollama
> `nomic-embed-text` = 768). It defines the Qdrant collection's vector size.

### Stability controls (smaller open models)

Weaker open models (7B-class, served via vLLM/SGLang) can degenerate into
repetition loops or ignore strict JSON-schema output. Nidus is built to stay
robust on them:

- **Single-word classification** — the router and graders ask for one word
  (`yes`/`no`, `vectorstore`/`direct`) and parse leniently, rather than relying
  on strict structured output. Failures fall back to a safe default.
- **`NIDUS_LLM_MAX_TOKENS`** / **`NIDUS_LLM_FREQUENCY_PENALTY`** /
  **`NIDUS_LLM_REQUEST_TIMEOUT`** — cap output, penalise repetition, and bound
  latency so a degenerating model fails fast instead of hanging.
- **Bounded loops** — the query-rewrite loop is capped by `NIDUS_MAX_RETRIES`,
  the regeneration loop by an independent generate-attempt counter, and the
  whole graph by a hard `recursion_limit`. No prompt can cause an infinite loop.

Qdrant defaults to **embedded** mode, persisting to `./.nidus/qdrant` — no Docker
needed. For **server** mode:

```bash
docker compose up -d          # starts Qdrant on :6333
# then in .env:
NIDUS_QDRANT_URL=http://localhost:6333
```

> Embedded on-disk Qdrant allows a single process at a time — run either the API
> server *or* a CLI command, not both concurrently. Use server mode for concurrency.

## Usage — CLI

```bash
uv run nidus info                        # show resolved config
uv run nidus init-db                     # create the collection
uv run nidus ingest ./docs               # index a file or directory
uv run nidus ask "What is Nidus?"        # ask a question
uv run nidus ask "..." --stream          # stream tokens
uv run nidus ask "..." --sources --trace # show citations + decision trace
uv run nidus serve                       # run the API server
```

## Usage — REST API

```bash
uv run nidus serve       # http://127.0.0.1:8000  (docs at /docs)
```

| Method & path | Body | Description |
| --- | --- | --- |
| `GET /health` | — | Status, version, backend, model |
| `POST /ingest/text` | `{"text": "...", "metadata": {}}` | Index raw text |
| `POST /ingest/path` | `{"path": "./docs"}` | Index a server-side file/dir |
| `POST /query` | `{"question": "...", "include_sources": true, "include_trace": false}` | Full answer + sources |
| `POST /chat` | `{"question": "..."}` | Token stream (SSE) |

```bash
curl -s localhost:8000/health
curl -s -XPOST localhost:8000/ingest/path -H 'content-type: application/json' \
     -d '{"path": "./docs"}'
curl -s -XPOST localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "What is Nidus?", "include_trace": true}'
curl -N -XPOST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question": "Summarise the docs"}'   # SSE stream
```

## Test

```bash
uv run pytest
```

Tests run **fully offline** — no API keys required. They use a fake chat model,
a fake retriever, and deterministic offline embeddings against an embedded Qdrant,
covering config loading, ingestion round-trips, and the graph's routing / grading /
retry-bound behaviour.

## Extend

- **New provider** → change `NIDUS_LLM_*` / `NIDUS_EMBED_*` (no code).
- **New document format** → add a loader in `nidus/ingest.py`.
- **New graph behaviour** → add a node factory in `nidus/graph/nodes.py` and wire it
  in `nidus/graph/builder.py`. Nodes take injected dependencies, so they stay
  unit-testable. For new yes/no or multi-way decisions, reuse the portable
  `classify()` helper in `nidus/graph/nodes.py`.
- **Hybrid/sparse retrieval** → swap `RetrievalMode` in `nidus/vectorstore.py`
  (`langchain-qdrant` supports dense / sparse / hybrid).
