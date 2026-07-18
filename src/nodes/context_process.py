"""Context process node — LLM-based compression of retrieved chunks.

检索到的 chunk 可能很长，无关部分浪费上下文、可能带偏生成。压缩用 LLM 把每个 chunk
裁剪成只保留与问题相关的核心信息；与问题无关的 chunk 输出"无关"被丢弃。

prompt 模板照搬 dev/advanced 的 context_compressor。仅 complex 路径启用（默认关闭）。
"""

from ..chunker import Chunk
from ..llm import get_chat_llm

_COMPRESS_SYSTEM_PROMPT = """\
你是一个文档压缩助手。给定用户问题和一段文档片段，请从文档中提取与问题直接相关
的核心信息，压缩成一段简短文字（不超过原长）。只输出压缩后的内容，不要解释、
不要回答问题本身。如果片段与问题无关，输出"无关"。
"""

_USER_TEMPLATE = "问题：{query}\n\n文档片段：\n{text}"


def _compress_one(query: str, text: str) -> str | None:
    """Compress one chunk; return None on LLM error (keep original)."""
    try:
        llm = get_chat_llm(temperature=0.1)
        result = llm.invoke(
            [
                {"role": "system", "content": _COMPRESS_SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(query=query, text=text)},
            ]
        ).content
        return result if isinstance(result, str) else None
    except Exception:
        return None


def context_process_node(state: dict) -> dict:
    """Compress each chunk to its query-relevant core; drop unrelated ones."""
    from ..config import get_config

    mod = get_config().modular
    if not mod.get("context_compression", {}).get("enabled", False):
        # Compression off — context = ranked chunks (or retrieved for simple path).
        chunks = state.get("ranked_chunks") or state.get("retrieved_chunks") or []
        return {"context": list(chunks)}

    query = state["query"]
    chunks: list[Chunk] = state.get("ranked_chunks") or state.get("retrieved_chunks") or []
    compressed: list[Chunk] = []
    for chunk in chunks:
        result = _compress_one(query, chunk.text)
        if result is None:
            # LLM failed — keep original, never silently drop.
            compressed.append(chunk)
            continue
        if result.strip() == "无关":
            continue
        compressed.append(
            Chunk(text=result.strip(), metadata={**chunk.metadata, "compressed": True})
        )
    return {"context": compressed}
