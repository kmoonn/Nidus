"""Web search node — time-sensitive path via DuckDuckGo.

仅当 config.web_search.enabled 为真时由 Router 路由到此节点。延迟导入
duckduckgo-search，避免硬依赖拖累未开启 web 路径的环境。

检索结果包装为 Chunk（metadata.source="web"），写入 state["context"] 供
Generate 节点引用。
"""

from ..chunker import Chunk


def web_search_node(state: dict) -> dict:
    """Search the web for the query; write results into context."""
    from ..config import get_config

    query = state["query"]
    config = get_config()
    web_cfg = config.web_search
    max_results = web_cfg.get("max_results", 5)

    chunks: list[Chunk] = []
    try:
        # Lazy import — duckduckgo-search is only needed when web path is active.
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "") or r.get("snippet", "")
            href = r.get("href") or r.get("link", "")
            text = f"{title}\n{body}".strip()
            if text:
                chunks.append(
                    Chunk(
                        text=text,
                        metadata={
                            "source": "web",
                            "page": href or title,
                            "web": True,
                        },
                    )
                )
    except Exception:
        # Web search failure — fall through with empty context (free-chat).
        pass

    return {"context": chunks}
