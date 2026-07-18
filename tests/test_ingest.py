"""Tests for document loading, splitting and ingestion into Qdrant."""

from __future__ import annotations

import importlib

import pytest
from langchain_core.documents import Document

from nidus import ingest


def test_load_text_file(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nHello world.", encoding="utf-8")
    docs = ingest.load_documents(f)
    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    assert docs[0].metadata["source"] == str(f)


def test_load_directory_recursively(tmp_path):
    (tmp_path / "a.txt").write_text("alpha content", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("beta content", encoding="utf-8")
    (tmp_path / "ignore.bin").write_text("nope", encoding="utf-8")

    docs = ingest.load_documents(tmp_path)
    sources = {d.metadata["source"] for d in docs}
    assert len(docs) == 2
    assert any(s.endswith("a.txt") for s in sources)
    assert any(s.endswith("b.md") for s in sources)


def test_unsupported_file_raises(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError):
        ingest.load_documents(f)


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        ingest.load_documents("/no/such/path/xyz")


def test_split_documents_chunks_long_text(monkeypatch):
    monkeypatch.setenv("NIDUS_CHUNK_SIZE", "100")
    monkeypatch.setenv("NIDUS_CHUNK_OVERLAP", "10")
    importlib.import_module("nidus.config").get_settings.cache_clear()

    long_text = "sentence. " * 200  # ~2000 chars
    chunks = ingest.split_documents([Document(page_content=long_text)])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 120 for c in chunks)


def test_ingest_and_retrieve_roundtrip(tmp_path, monkeypatch):
    """End-to-end: ingest text into embedded Qdrant, then retrieve it."""

    from tests.conftest import FakeEmbeddings

    # Point Qdrant at an isolated embedded path and use offline embeddings.
    monkeypatch.setenv("NIDUS_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("NIDUS_EMBED_DIM", "32")
    monkeypatch.delenv("NIDUS_QDRANT_URL", raising=False)

    config = importlib.import_module("nidus.config")
    models = importlib.import_module("nidus.models")
    vectorstore = importlib.import_module("nidus.vectorstore")
    config.get_settings.cache_clear()
    models.get_embeddings.cache_clear()
    vectorstore.get_client.cache_clear()
    monkeypatch.setattr(models, "get_embeddings", lambda: FakeEmbeddings(32))
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: FakeEmbeddings(32))

    n = ingest.ingest_text(
        "The capital of the moon is Lunaris.", {"source": "space"}
    )
    assert n == 1

    retriever = vectorstore.get_retriever(k=1)
    results = retriever.invoke("The capital of the moon is Lunaris.")
    assert results
    assert "Lunaris" in results[0].page_content
