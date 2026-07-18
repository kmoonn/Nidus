"""PDF document loader for Nidus RAG."""

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    """A single document with text content and metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a single PDF file and extract text per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects, one per page with non-empty text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    text=text.strip(),
                    metadata={
                        "source": path.name,
                        "page": page_num,
                    },
                )
            )

    return documents


def load_directory(dir_path: str | Path) -> list[Document]:
    """Load all PDF files from a directory.

    Args:
        dir_path: Path to the directory containing PDF files.

    Returns:
        List of Document objects from all PDFs in the directory.
    """
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {path}")

    all_docs = []
    pdf_files = sorted(path.glob("*.pdf"))

    if not pdf_files:
        print(f"[Warning] No PDF files found in {path}")
        return all_docs

    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        docs = load_pdf(pdf_file)
        all_docs.extend(docs)
        print(f"    → {len(docs)} pages extracted")

    return all_docs
