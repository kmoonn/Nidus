"""Router node — classifies the query into a route at runtime.

Modular RAG 的核心：运行时动态路由。Router 用 LLM（结构化输出）把问题分为
- simple：简单事实型 → 快速路径（直达检索 + 生成）
- complex：复杂分析型 → 深度路径（分解 + 多路检索 + rerank + 压缩 + 生成）
- time_sensitive：时效型 → Web 搜索路径（仅当 web_search.enabled 时启用）

web_search 关闭时，Router 退回 simple/complex 二分类，避免把时效型问题路由到
不存在的 web 路径。
"""

from pydantic import BaseModel, Field

from ..llm import get_chat_llm

# 与 graph.py 的 Route enum 保持字符串一致
ROUTE_SIMPLE = "simple"
ROUTE_COMPLEX = "complex"
ROUTE_TIME_SENSITIVE = "time_sensitive"


class RouteDecision(BaseModel):
    """LLM 结构化输出：问题分类 + 分类理由。"""

    route: str = Field(
        ...,
        description="问题类型：simple / complex / time_sensitive",
    )
    reason: str = Field(..., description="简短的分类理由")


SYSTEM_PROMPT = """\
你是一个问题路由助手。请根据用户问题判断其类型：

- simple：简单事实型问题，单一信息点，直接检索文档即可回答（如"碳达峰的目标年份？"）。
- complex：复杂分析型问题，需要对比/综合/多步推理，或需要多方面信息（如"对比三份报告能源转型的异同"）。
- time_sensitive：时效型问题，依赖最新/实时信息，文档中很可能没有（如"今天有什么新闻"、"最新的股价"）。

只输出一个 route 标签和简短理由。当问题与已索引的能源/碳/金融文档相关且为单一事实查询时，优先 simple。
当问题需要综合多份文档或多角度分析时，归为 complex。仅当问题明确依赖实时信息时才归为 time_sensitive。
"""

SYSTEM_PROMPT_NO_WEB = """\
你是一个问题路由助手。请根据用户问题判断其类型：

- simple：简单事实型问题，单一信息点，直接检索文档即可回答（如"碳达峰的目标年份？"）。
- complex：复杂分析型问题，需要对比/综合/多步推理，或需要多方面信息（如"对比三份报告能源转型的异同"）。

只输出 simple 或 complex 两个标签之一，并给出简短理由。当问题为单一事实查询时优先 simple，
需要综合多方面信息时归为 complex。
"""


def router_node(state: dict) -> dict:
    """Classify the query; write `route` + `reason` into state.

    Falls back to `simple` on any LLM error (never block the pipeline).
    """
    from ..config import get_config

    query = state["query"]
    web_enabled = get_config().web_search.get("enabled", False)

    system = SYSTEM_PROMPT if web_enabled else SYSTEM_PROMPT_NO_WEB
    route = ROUTE_SIMPLE
    reason = "fallback (LLM unavailable)"

    try:
        llm = get_chat_llm(temperature=0.0)
        decision: RouteDecision = llm.with_structured_output(RouteDecision).invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": query}]
        )
        route = decision.route.strip().lower()
        # Normalize / guard against unexpected labels.
        if web_enabled and route not in (
            ROUTE_SIMPLE,
            ROUTE_COMPLEX,
            ROUTE_TIME_SENSITIVE,
        ):
            route = ROUTE_COMPLEX
        elif not web_enabled and route not in (ROUTE_SIMPLE, ROUTE_COMPLEX):
            # time_sensitive 等无效标签退回 complex（走检索兜底，比丢失好）
            route = ROUTE_COMPLEX
        reason = decision.reason
    except Exception:
        # Never block the pipeline on the router LLM call.
        pass

    return {"route": route, "reason": reason}
