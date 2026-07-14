"""LLM generator for Nidus RAG — uses SiliconFlow API for answer generation."""

from openai import OpenAI

from .chunker import Chunk
from .config import get_config

RAG_SYSTEM_PROMPT = """\
你是一个专业的文档问答助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请诚实地说"根据已有资料无法回答"。
请用中文回答，并引用信息来源（文件名和页码）。
"""

CHAT_SYSTEM_PROMPT = """\
你是一个友好的AI助手。用户的问题与已索引的文档无关，请直接用中文回答。
"""

RAG_USER_TEMPLATE = """\
参考资料：
{contexts}

用户问题：{query}

回答："""


def _format_contexts(chunks: list[Chunk], max_chars: int = 6000) -> str:
    """Format retrieved chunks into a context string for the LLM.

    Args:
        chunks: List of retrieved Chunk objects.
        max_chars: Maximum total characters for the context.

    Returns:
        Formatted context string.
    """
    parts = []
    total_len = 0

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "未知")
        page = chunk.metadata.get("page", "?")
        entry = f"[{i}] 来源: {source} 第{page}页\n{chunk.text}\n"

        if total_len + len(entry) > max_chars:
            break

        parts.append(entry)
        total_len += len(entry)

    return "\n".join(parts)


class Generator:
    """Generate answers using SiliconFlow LLM API (OpenAI-compatible)."""

    def __init__(self):
        config = get_config()
        llm_cfg = config.llm
        self._model = llm_cfg["model"]
        self._client = OpenAI(
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg["api_key"],
        )

    def generate(self, query: str, contexts: list[Chunk]) -> str:
        """Generate an answer for the query given the retrieved contexts.

        Args:
            query: User question.
            contexts: List of retrieved Chunk objects as context.
                     Empty list means the query is unrelated to documents — free chat mode.

        Returns:
            Generated answer string.
        """
        if not contexts:
            # Free chat mode — no document context, just answer directly
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content

        # RAG mode — answer with document context
        formatted = _format_contexts(contexts)
        user_message = RAG_USER_TEMPLATE.format(contexts=formatted, query=query)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        return response.choices[0].message.content
