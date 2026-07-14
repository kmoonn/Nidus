"""RAG pipeline for Nidus — orchestrates retrieve → generate."""

from dataclasses import dataclass, field

from .chunker import Chunk
from .config import get_config
from .generator import Generator
from .retriever import Retriever

# If the best (smallest) distance exceeds this threshold,
# the query is considered unrelated to the indexed documents.
DEFAULT_RELEVANCE_THRESHOLD = 0.50


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
        config = get_config()
        self._relevance_threshold = config.retriever.get(
            "relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD
        )

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

        # Step 2: Check relevance — if even the best result is too distant,
        # the query is unrelated to our documents (e.g. "你好", "今天天气怎么样").
        # Let the LLM answer freely without forcing document context.
        best_distance = chunks[0].metadata.get("distance", 1.0)
        if best_distance > self._relevance_threshold:
            answer_text = self._generator.generate(query, contexts=[])
            return Answer(query=query, answer=answer_text, sources=[])

        # Step 3: Generate answer with context
        answer_text = self._generator.generate(query, chunks)

        # Step 4: Collect source info
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
