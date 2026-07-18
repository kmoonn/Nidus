"""Document ingestion: load → split → embed → upsert into Qdrant.

Supports plain text, Markdown and PDF files, plus recursive directory walks.
Kept intentionally small; add loaders here to extend format coverage.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from nidus.config import get_settings
from nidus.vectorstore import get_vectorstore

# Extensions treated as UTF-8 text. Everything else needs a dedicated loader.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".py", ".json", ".csv"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": page_num},
                )
            )
    return docs


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return _load_pdf(path)
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return []
        return [Document(page_content=text, metadata={"source": str(path)})]
    raise ValueError(f"Unsupported file type: {path.suffix} ({path})")


def load_documents(path: str | Path) -> list[Document]:
    """Load documents from a file or (recursively) a directory."""

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        return _load_file(p)

    docs: list[Document] = []
    for child in sorted(p.rglob("*")):
        if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.extend(_load_file(child))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Chunk documents using the configured size/overlap."""

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def ingest_documents(docs: list[Document]) -> int:
    """Split and upsert already-loaded documents. Returns chunk count."""

    if not docs:
        return 0
    chunks = split_documents(docs)
    if not chunks:
        return 0
    get_vectorstore().add_documents(chunks)
    return len(chunks)


def ingest_path(path: str | Path) -> tuple[int, int]:
    """Load, split and index a file/directory.

    Returns ``(num_source_documents, num_chunks)``.
    """

    docs = load_documents(path)
    return len(docs), ingest_documents(docs)


def ingest_text(text: str, metadata: dict | None = None) -> int:
    """Index a raw text blob. Returns chunk count."""

    doc = Document(page_content=text, metadata=metadata or {})
    return ingest_documents([doc])
