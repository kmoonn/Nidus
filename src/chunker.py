"""Text chunker for Nidus RAG — fixed-size chunking with overlap."""

from dataclasses import dataclass, field

from .loader import Document


@dataclass
class Chunk:
    """A chunk of text with preserved metadata from the source document."""

    text: str
    metadata: dict = field(default_factory=dict)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split documents into fixed-size chunks with overlap.

    Args:
        documents: List of Document objects to chunk.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of Chunk objects.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []

    for doc in documents:
        text = doc.text
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                metadata = {
                    **doc.metadata,
                    "chunk_index": chunk_index,
                }
                chunks.append(Chunk(text=chunk_text, metadata=metadata))
                chunk_index += 1

            # Move forward by (chunk_size - overlap)
            start += chunk_size - chunk_overlap

    return chunks
