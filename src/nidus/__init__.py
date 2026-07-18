"""Nidus — Light Extensible Agentic RAG built on LangGraph."""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__", "get_settings", "Settings"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    # Lazy imports keep ``import nidus`` cheap and avoid importing heavy
    # optional deps (langchain, qdrant) at package import time.
    if name in {"get_settings", "Settings"}:
        from nidus.config import Settings, get_settings

        return {"get_settings": get_settings, "Settings": Settings}[name]
    raise AttributeError(f"module 'nidus' has no attribute {name!r}")
