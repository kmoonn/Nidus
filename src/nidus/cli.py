"""Typer command-line interface for Nidus.

Commands:
    nidus init-db            Create the Qdrant collection.
    nidus ingest PATH        Index a file or directory.
    nidus ask "QUESTION"     Ask a question (``--stream`` for token streaming).
    nidus serve              Run the FastAPI server.
    nidus info               Show the active configuration.
"""

from __future__ import annotations

import asyncio

import typer

app = typer.Typer(
    add_completion=False,
    help="Nidus — Light Extensible Agentic RAG built on LangGraph.",
    no_args_is_help=True,
)


@app.command()
def info() -> None:
    """Show the resolved configuration (secrets masked)."""

    from nidus import __version__
    from nidus.config import get_settings
    from nidus.vectorstore import vectorstore_mode

    settings = get_settings()
    typer.echo(f"Nidus v{__version__}")
    typer.echo(f"  llm_model     : {settings.llm_model}")
    typer.echo(f"  llm_base_url  : {settings.llm_base_url}")
    typer.echo(f"  embed_model   : {settings.embed_model} (dim={settings.embed_dim})")
    typer.echo(f"  vectorstore   : {vectorstore_mode()}")
    typer.echo(f"  collection    : {settings.qdrant_collection}")
    typer.echo(f"  max_retries   : {settings.max_retries}")


@app.command("init-db")
def init_db() -> None:
    """Create the configured Qdrant collection if it does not exist."""

    from nidus.config import get_settings
    from nidus.vectorstore import ensure_collection

    ensure_collection()
    typer.secho(
        f"Collection '{get_settings().qdrant_collection}' is ready.",
        fg=typer.colors.GREEN,
    )


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to index."),
) -> None:
    """Load, split and index documents from PATH."""

    from nidus.ingest import ingest_path

    try:
        docs, chunks = ingest_path(path)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(
        f"Ingested {docs} document(s) -> {chunks} chunk(s).", fg=typer.colors.GREEN
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask."),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream tokens."),
    show_sources: bool = typer.Option(
        False, "--sources", help="Print retrieved sources."
    ),
    show_trace: bool = typer.Option(
        False, "--trace", help="Print the graph decision trace."
    ),
) -> None:
    """Ask a question against the indexed corpus."""

    if stream:
        _ask_streaming(question)
        return

    from nidus.service import answer_question

    resp = answer_question(question)
    typer.echo(resp.answer)

    if show_trace and resp.trace:
        typer.secho("\n-- trace --", fg=typer.colors.BLUE)
        for step in resp.trace:
            typer.echo(f"  {step}")
    if show_sources and resp.sources:
        typer.secho("\n-- sources --", fg=typer.colors.BLUE)
        for i, src in enumerate(resp.sources, 1):
            origin = src.metadata.get("source", "?")
            preview = src.content[:200].replace("\n", " ")
            typer.echo(f"  [{i}] {origin}: {preview}...")


def _ask_streaming(question: str) -> None:
    from nidus.service import astream_answer

    async def run() -> None:
        async for token in astream_answer(question):
            typer.echo(token, nl=False)
        typer.echo("")

    asyncio.run(run())


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host (default from config)."),
    port: int | None = typer.Option(None, help="Bind port (default from config)."),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev)."),
) -> None:
    """Run the FastAPI server."""

    import uvicorn

    from nidus.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "nidus.api.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
