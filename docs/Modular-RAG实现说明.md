# Modular RAG 实现说明

> Nidus 从 Advanced RAG (`dev/advanced`) 演进到 Modular RAG (`dev/modular`) 的实现记录。
> 核心变化：从"固定线性流水线 + 配置静态开关"改为"LangGraph 动态路由 + 可组合模块网络"。
> 每个模块都标注：解决什么问题 → 怎么实现 → 如何验证。

---

## 总体架构

Advanced RAG 是一条固定线性流水线，所有问题走同一条 `query → 改写/分解/扩展/HyDE → 检索 → dedup/rerank/压缩 → 生成` 链路，策略由 `config.advanced.*.enabled` 在**启动时静态决定**。痛点：复杂分析题与简单事实题共用一套重逻辑（成本高），扩展需改 pipeline，无法按问题类型动态适配。

Modular RAG 用 **LangGraph** 把流水线变成可组合的模块网络，Router 在**运行时**按问题类型把不同问题路由到不同模块链：

```
                 ┌──────────┐
        query ──▶│  Router  │  ChatOpenAI.with_structured_output → {route, reason}
                 └────┬─────┘
            add_conditional_edges(route)
            ┌─────────┼──────────────────┐
            ▼         ▼                  ▼
         simple    complex            time_sensitive*
         快速路径   深度路径            web 搜索路径
            │         │                  │
            ▼         ▼                  ▼
       (无 transform  QueryTransform      WebSearch node
        直达 retrieve) (decompose)        (DuckDuckGo，门控)
            │         │                  │
            ▼         ▼                  │
        Retrieve    Retrieve (multi)    │
        (向量或混合)  + RRF              │
            │         │                  │
            ▼         ▼                  │
       相关度门控   Rank(dedup+rerank)    │
            │      + ContextCompress      │
            └─────────┼──────────────────┘
                      ▼
                  Generate
```
*`time_sensitive` 仅当 `config.web_search.enabled` 为真时由 Router 输出；否则 Router 退回 simple/complex 二分类。

### GraphState（`src/graph.py`）

LangGraph 节点间通过 `GraphState`（TypedDict）传递数据：`query` / `route` / `reason` / `queries` / `retrieved_chunks` / `ranked_chunks` / `context` / `answer` / `sources`。每个节点是 `state -> state` 的纯函数风格 callable，失败静默回退，绝不阻塞主流程（沿用 Advanced 的鲁棒性约定）。

### 与 Advanced 的核心区别

| 维度 | Advanced（已实现） | Modular（本文档） |
|---|---|---|
| 流程 | 固定线性流水线 | 可组合的模块网络（图） |
| 策略 | 配置开关**静态决定**（启动时） | **运行时动态选择**（按路由） |
| 扩展 | 加新功能改 pipeline | 加新节点即插即用 |
| 适配 | 一套策略应对所有问题 | 不同问题走不同路径 |
| 框架 | 纯 Python 自实现 | LangGraph 编排 |

---

## 一、Router 节点（`src/nodes/router.py`）

**解决问题**：所有问题共用同一套重逻辑——简单事实题走分解+rerank+压缩是浪费，复杂分析题只做快速检索又召回不全。需要运行时判断问题类型并路由到合适路径。

**实现**：`ChatOpenAI.with_structured_output(RouteDecision)` 让 LLM 返回结构化 `{route, reason}`，分类为 `simple` / `complex` / `time_sensitive`。`web_search.enabled=false` 时换用二分类 prompt，且把 `time_sensitive` 归一化为 `complex`（避免路由到不存在的 web 路径，走检索兜底）。LLM 失败时静默回退 `simple`。

**接入**：图入口 `set_entry_point("router")`，经 `add_conditional_edges("router", _route_after_router, {...})` 分流。

**验证**：`tests/test_router.py` —— mock `ChatOpenAI` 断言 simple/complex 分类、web 关闭时 time_sensitive 降级为 complex、web 开启时保留、LLM 异常回退 simple。

---

## 二、QueryTransform 节点（`src/nodes/query_transform.py`）

**解决问题**：complex 路径下用户问题常是复合问题（"对比 A 和 B"），单一查询检索不全，需拆成多个子问题分头检索。

**实现**：**策略按路由动态选**——`simple` 路径不做变换（`[query]`，快速路径），`complex` 路径走 LLM 查询分解（prompt 照搬 dev/advanced 的 `DECOMPOSITION_SYSTEM_PROMPT`），可选 HyDE。用 langchain `ChatOpenAI` 重写调用（替换原 `openai` SDK）。LLM 失败回退 `[query]`。

**接入**：complex 路由分支 → `query_transform` 节点 → `retrieve`。simple 路径直接 → `retrieve`。

**验证**：`tests/test_graph.py` 断言 complex 路径经过 `query_transform` 节点；简单问题跳过。

---

## 三、Retrieve 节点（`src/nodes/retrieve.py`）

**解决问题**：纯向量检索精确关键词匹配弱；多子查询需融合。

**实现**：复用基础设施层 `Store`（ChromaDB）/ `Embedder` / `bm25.py`（`BM25Index` + `reciprocal_rank_fusion`，纯算法原样保留）。每条查询做向量（+可选 BM25）召回，多查询再二次 RRF 融合。检索相关度（distance）写入 `chunk.metadata["distance"]`，供下游相关度门控使用。

**接入**：simple/complex 路径均汇聚到 `retrieve` 节点 → 相关度门控条件边。

**验证**：`tests/test_bm25.py`（复用）覆盖 BM25 排序与 RRF 融合。

---

## 四、相关度门控（`src/graph.py::_route_after_retrieve`）

**解决问题**：用户问"你好"或文档无相关内容时，不应强行检索拼接无关上下文。

**实现**：条件边 `_route_after_retrieve`：Retrieve 后若 `retrieved_chunks` 空、或最佳结果 `distance > config.retriever.relevance_threshold(0.50)` → 直达 `generate`（free-chat 模式，空 context）。相关时按路由分流——`complex` → `rank`（深度后处理），`simple` → `context_passthrough`（直接组装，不做 rerank/压缩）。

**验证**：`tests/test_graph.py` —— 空/不相关结果走 generate、相关 simple 走 passthrough、相关 complex 走 rank。

---

## 五、Rank 节点（`src/nodes/rank.py`）

**解决问题**：向量检索 Top-K 里混入噪声、顺序不准。

**实现**：先 `Deduplicator`（字符 3-gram Jaccard，`src/dedup.py` 原样保留）去重减少候选量，再用 SiliconFlow `/rerank`（Cross-Encoder，`BAAI/bge-reranker-v2-m3`）重排，保留 `top_n`。rerank 非 OpenAI 兼容接口，沿用 dev/advanced 的 urllib 直连。rerank 失败回退原序，绝不阻塞。

**接入**：complex 路径 `retrieve` → `rank` → `context_process`。

**验证**：`tests/test_dedup.py`（复用）覆盖去重逻辑。

---

## 六、ContextProcess 节点（`src/nodes/context_process.py`）

**解决问题**：chunk 过长，无关部分干扰回答、挤占上下文窗口。

**实现**：用 LLM 把每个 chunk 压缩成只保留与问题相关的核心信息，无关的输出"无关"被丢弃，LLM 失败则保留原文（绝不丢信息）。prompt 照搬 dev/advanced。默认关闭，仅 complex 路径启用时可配置开启。`context_passthrough`（simple 路径）跳过压缩。

**验证**：`tests/test_graph.py` complex 路径遍历 `context_process` 节点。

---

## 七、Generate 节点（`src/nodes/generate.py`）

**解决问题**：基于检索上下文生成答案；无上下文时退化为自由对话。

**实现**：空 context → free-chat 模式（`CHAT_SYSTEM_PROMPT`，temp 0.7）；非空 → RAG 模式（`RAG_SYSTEM_PROMPT` + 编号引用上下文，temp 0.3，截断 6000 字）。prompt 模板照搬 dev/advanced。同时收集去重后的 sources（按 source+page）。LLM 失败返回友好错误而非崩溃。

**验证**：`tests/test_graph.py` 三条路径均汇聚到 `generate` 并产出 answer。

---

## 八、WebSearch 节点（`src/nodes/web_search.py`）

**解决问题**：时效型问题（最新新闻/股价）文档中可能没有，需实时检索。

**实现**：配置门控——仅当 `config.web_search.enabled=true` 时 Router 输出 `time_sensitive`。延迟导入 `duckduckgo-search`（未开启时不硬依赖），检索结果包装为 Chunk（`source="web"`）写入 context 供 Generate 引用。失败回退空 context（free-chat）。

**接入**：`time_sensitive` 路由 → `web_search` → `generate`。

**验证**：`tests/test_graph.py` —— web 路径遍历 `web_search` 节点后直达 generate。

---

## 九、LLM 工厂（`src/llm.py`）

**解决问题**：节点共享 LLM 客户端，避免重复构建；便于测试 mock。

**实现**：`get_chat_llm()` 返回 `ChatOpenAI` 单例（`base_url`/`api_key`/`model` 来自 `config.llm`，指向 SiliconFlow）；`get_embeddings()` 返回 `OpenAIEmbeddings` 单例。`reset_llm_cache()` 清缓存（re-index / 测试用）。

---

## 十、基础设施复用（从 dev/advanced 拷贝）

按计划"重写模块层，复用基础设施层"，以下文件从 `dev/advanced` 原样拷贝（非重写）：`loader.py`、`chunker.py`、`store.py`（ChromaDB）、`embedder.py`、`config.py`、`config.yaml`、`static/index.html`、`pyproject.toml`（追加 langgraph 依赖）。纯算法 `bm25.py`、`dedup.py` 原样保留为 `src/` 顶层模块（仅修正 import 路径）。

**不迁入**：`semantic_chunker.py`、`hierarchical_index.py`、`query_rewriting`/`query_expansion` 等多策略——Modular 阶段聚焦动态路由，这些列入"后续演进"。

---

## 测试

```bash
uv run pytest        # 42 passed
```

- **纯逻辑**：`test_bm25.py`、`test_dedup.py`、`test_config.py`（均复用）。
- **节点/编排**：`test_router.py`（mock LLM 分类）、`test_graph.py`（条件边分支选择、三路径遍历、相关度门控）。
- mock 策略：`conftest.py` 的 autouse `_reset_config` 每个测试重置配置单例；测试用临时 YAML 注入隔离配置。

---

## 配置参考

`config.yaml` 新增块：

```yaml
modular:
  router: {enabled: true}
  query_decomposition: {enabled: true}
  hyde: {enabled: false}
  hybrid_search: {enabled: true}
  dedup: {enabled: true, similarity_threshold: 0.85}
  reranking: {enabled: true, top_n: 5}
  context_compression: {enabled: false}

web_search:
  enabled: false          # 开启后 time_sensitive 路径可用
  max_results: 5
```

---

## 后续演进

- **更多 Retrieve 模块**：知识图谱检索、LLM-as-reranker、模型集成（Stage 2 文档列出但本阶段未实现）。
- **语义分块 / 分层索引**：从 dev/advanced 迁入（按需）。
- **Agentic RAG**：LangGraph 的图模型天然支持 ReAct 循环，下一阶段把 Router 升级为可调用工具的 Agent。
