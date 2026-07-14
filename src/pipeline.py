"""RAG pipeline for Nidus — orchestrates retrieve → generate."""

from dataclasses import dataclass, field

from .chunker import Chunk
from .generator import Generator
from .retriever import Retriever


@dataclass
class Answer:
    """Result of a RAG query."""

    query: str
    answer: str
    sources: list[dict] = field(default_factory=list)


class Pipeline:
    """Full RAG pipeline: retrieve relevant chunks, then generate an answer."""

    def __init__(self):
        self._retriever = Retriever()
        self._generator = Generator()

    def ask(self, query: str) -> Answer:
        """Ask a question and get an answer based on indexed documents.

        Args:
            query: User question.

        Returns:
            Answer object with the generated answer and source references.
        """
        # Step 1: Retrieve relevant chunks
        chunks = self._retriever.retrieve(query)

        if not chunks:
            return Answer(
                query=query,
                answer="抱歉，没有找到相关的文档内容。请先运行索引构建命令。",
                sources=[],
            )

        # Step 2: Generate answer with context
        answer_text = self._generator.generate(query, chunks)

        # Step 3: Collect source info
        sources = []
        for chunk in chunks:
            source = {
                "source": chunk.metadata.get("source", "未知"),
                "page": chunk.metadata.get("page", "?"),
                "distance": chunk.metadata.get("distance"),
            }
            if source not in sources:
                sources.append(source)

        return Answer(
            query=query,
            answer=answer_text,
            sources=sources,
        )
