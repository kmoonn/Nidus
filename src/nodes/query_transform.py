"""Query transform node — route-dependent pre-retrieval transform.

Modular 的策略不再由 config 静态开关决定，而由运行时路由动态选择：
- simple：不做变换，直接用原查询（快速路径）。
- complex：查询分解（把复合问题拆成多个子问题分头检索）；可选 HyDE。

prompt 模板照搬 dev/advanced 的 query_transform（经实测验证有效），但用
langchain ChatOpenAI 重写调用。
"""

from ..llm import get_chat_llm
from .router import ROUTE_COMPLEX, ROUTE_SIMPLE

DECOMPOSITION_SYSTEM_PROMPT = """\
你是一个查询分解助手。如果用户的问题是复合问题（如"对比 A 和 B"、"A 和 B 的关系"，
或需要多个独立信息点才能回答），请将其拆分为多个独立、可单独检索的子问题，
每行输出一个子问题，不要编号、不要解释。如果问题已经是简单事实问题，原样返回单行即可。
"""

HYDE_SYSTEM_PROMPT = """\
你是一个文档助手。请针对用户问题，给出一段简短（不超过150字）的假设性回答，
假设相关文档存在。这段回答将用于向量检索，请包含可能出现在真实文档中的
关键词和表述。只输出假设回答本身，不要解释。
"""


def _parse_lines(raw: str) -> list[str]:
    """Split a multi-line LLM response into cleaned, non-empty lines."""
    if not raw:
        return []
    lines: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        for prefix in ("-", "、"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
        if len(cleaned) > 1 and cleaned[0].isdigit():
            j = 0
            while j < len(cleaned) and cleaned[j].isdigit():
                j += 1
            if j < len(cleaned) and cleaned[j] in ".)":
                cleaned = cleaned[j + 1 :]
        cleaned = cleaned.strip().strip('"').strip("'").strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _decompose(query: str) -> list[str]:
    """Split a compound question into sub-questions via LLM (fallback to [query])."""
    try:
        llm = get_chat_llm(temperature=0.2)
        raw = llm.invoke(
            [
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
        ).content
        sub_questions = _parse_lines(raw if isinstance(raw, str) else "")
        if not sub_questions:
            return [query]
        result: list[str] = []
        for sq in sub_questions:
            if sq and sq not in result:
                result.append(sq)
        return result or [query]
    except Exception:
        return [query]


def _hyde(query: str) -> str:
    """Generate a hypothetical answer for the query (fallback to query)."""
    try:
        llm = get_chat_llm(temperature=0.2)
        answer = llm.invoke(
            [
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
        ).content
        cleaned = (answer or "").strip() if isinstance(answer, str) else ""
        return cleaned or query
    except Exception:
        return query


def query_transform_node(state: dict) -> dict:
    """Produce one or more search queries based on the route.

    - simple → [query]（无变换，快速路径）
    - complex → decompose → 可选 HyDE per sub-query
    - time_sensitive → [query]（不依赖文档检索，WebSearch 节点直接用）
    """
    from ..config import get_config

    query = state["query"]
    route = state.get("route", ROUTE_SIMPLE)

    if route != ROUTE_COMPLEX:
        return {"queries": [query]}

    mod = get_config().modular
    queries = _decompose(query)
    if mod.get("hyde", {}).get("enabled", False):
        queries = [_hyde(sub) for sub in queries]

    return {"queries": queries or [query]}
