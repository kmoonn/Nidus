"""Generate node — final answer generation.

空 context → free-chat 模式（问题与文档无关，直接回答）。
非空 context → RAG 模式（基于文档引用回答）。

prompt 模板照搬 dev/advanced 的 generator。
"""

from ..chunker import Chunk
from ..llm import get_chat_llm

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
    """Format retrieved chunks into a numbered context string for the LLM."""
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


def _collect_sources(chunks: list[Chunk]) -> list[dict]:
    """Build deduplicated source list (by source + page)."""
    sources: list[dict] = []
    seen: set[tuple] = set()
    for chunk in chunks:
        src = {
            "source": chunk.metadata.get("source", "未知"),
            "page": chunk.metadata.get("page", "?"),
            "distance": chunk.metadata.get("distance"),
            "rerank_score": chunk.metadata.get("rerank_score"),
        }
        key = (src["source"], src["page"])
        if key not in seen:
            seen.add(key)
            sources.append(src)
    return sources


def generate_node(state: dict) -> dict:
    """Generate the final answer + collect sources from context chunks."""
    query = state["query"]
    chunks: list[Chunk] = state.get("context") or []

    try:
        llm = get_chat_llm(temperature=0.7 if not chunks else 0.3)
        if not chunks:
            # Free-chat mode (query unrelated to documents)
            answer = llm.invoke(
                [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]
            ).content
        else:
            formatted = _format_contexts(chunks)
            answer = llm.invoke(
                [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": RAG_USER_TEMPLATE.format(
                            contexts=formatted, query=query
                        ),
                    },
                ]
            ).content
        answer = answer if isinstance(answer, str) else str(answer)
    except Exception:
        answer = "抱歉，生成回答时发生错误，请稍后重试。"

    sources = _collect_sources(chunks)
    return {"answer": answer, "sources": sources}
