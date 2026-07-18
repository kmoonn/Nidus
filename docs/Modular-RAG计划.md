# Modular RAG 实现计划

> 本文件供**新会话**执行。Nidus 已完成 Native（`dev/native`）与 Advanced（`dev/advanced`）阶段。
> 下一阶段 Modular RAG 引入 LangGraph 框架，从 `main` 创建全新分支实现。

---

## 背景与判断结论

项目定位：**复现 RAG 演进阶段**（非"从零实现"）。因此框架与否取决于阶段是否演进到需要框架的程度。

**判断结论**（已与用户确认）：
- Modular 阶段**未到必须上框架的程度**，但引入框架有明显收益。
- **引入 LangGraph**：其节点 + 条件路由图模型，天然契合 Modular 的 "Router → Module Selector → 各模块" 架构，且为后续 Agentic 的 ReAct 循环铺路。
- 检索层 LlamaIndex 与自实现重复度高，Modular **不引入**，仍用 ChromaDB。
- 新分支**基于 `main` 全新创建**，模块全部用 LangGraph **重写**，不复用 `dev/advanced` 的自实现模块代码（保留作为理论参考）。

## Advanced → Modular 的核心区别

| 维度 | Advanced（已实现） | Modular（本计划） |
|---|---|---|
| 流程 | 固定线性流水线 | 可组合的模块网络 |
| 策略 | 配置开关静态决定 | **运行时动态选择**（按问题类型路由） |
| 扩展 | 加新功能改 pipeline | 加新模块即插即用 |
| 适配 | 一套策略应对所有问题 | 不同问题走不同路径 |
| 框架 | 纯 Python 自实现 | LangGraph 编排 |

---

## 目标

用 LangGraph 从干净起点重新实现 Modular RAG，体现"运行时动态路由 + 模块可组合"，为 Agentic 阶段铺路。

## 核心架构（LangGraph 图）

```
                 ┌──────────┐
        query ──▶│  Router  │  LLM 分类问题类型
                 └────┬─────┘
            ┌─────────┼─────────┐
            ▼         ▼         ▼
        简单事实型  复杂分析型  时效型
            │         │         │
            ▼         ▼         ▼
       快速路径   深度路径    搜索路径
       (向量检索) (分解+多路  (web search
        +生成)   +rerank+生成)  +生成)
            │         │         │
            └─────────┴─────────┘
                      ▼
                   Generate
```

### 图元素
- **State**（TypedDict）：`query` / `queries` / `retrieved_chunks` / `ranked_chunks` / `context` / `answer` / `sources` / `route`（路由标签）。
- **Router 节点**：LLM 分类问题类型 → 输出 `route` 标签（`simple` / `complex` / `time_sensitive`）。
- **Module Selector**（条件边）：根据 `route` 选不同模块链。
- **模块节点**（每个是 LangGraph 节点）：
  - Query Transform：改写 / 扩展 / 分解 / HyDE（按路由动态选）
  - Retrieve：向量 / BM25 / 混合（按路由选）
  - Rank：rerank / MMR / lost-in-the-middle
  - Context Process：dedup / filter / compress / truncate
  - Generate：LLM
- **条件边**：`add_conditional_edges(Router, lambda state: state["route"], {"simple": ..., "complex": ..., "time_sensitive": ...})`

## 框架与依赖

新增依赖（写入 `pyproject.toml`）：
- `langgraph`（图编排/状态机/条件路由）
- `langchain-core`（langgraph 依赖，State/消息抽象）
- `langchain-openai`（LLM/Embedding 接 SiliconFlow，OpenAI 兼容）
- ChromaDB（沿用，检索后端）

保留 `config.yaml` 统一配置风格。

## 重写范围（基于 main 全新实现）

- **从 `main` 创建 `dev/modular` 分支**：`git checkout main && git checkout -b dev/modular`。
- **不复用** `dev/advanced` 的 `src/advanced/` 模块代码；用 LangGraph 节点重写各模块逻辑。
- 保留 `docs/RAG演进阶段101.md` 作为理论参考；`docs/Advanced-RAG实现说明.md` 作为对照。
- README 定位句（"从零实现...不依赖 LangChain/LlamaIndex"）需更新为反映"复现 RAG 演进阶段 + Modular 引入 LangGraph"。

## 目录结构建议

```
src/
├── graph.py          # LangGraph 编排：State + 节点 + 条件边
├── nodes/            # 各模块节点
│   ├── router.py        # 问题分类路由
│   ├── query_transform.py
│   ├── retrieve.py
│   ├── rank.py
│   ├── context_process.py
│   └── generate.py
├── config.py         # 沿用配置加载
├── api.py            # FastAPI（沿用入口形态）
└── main.py           # CLI
tests/
└── test_graph.py 等
```

## 交付物

- LangGraph 编排的 RAG pipeline（动态路由）。
- CLI + FastAPI（沿用现有入口形态：`index` / `ask` / `interactive`）。
- 测试：Router 分类、条件边选择、各模块节点。
- 文档：`docs/Modular-RAG实现说明.md`。

## 验证

- **单元测试**：Router 分类（mock LLM）、条件路由分支选择、各模块节点。
- **端到端**：
  - 简单事实问题（"碳达峰的目标年份？"）→ 走快速路径。
  - 复杂分析问题（"对比三份报告能源转型的异同"）→ 走深度路径（分解+多路+rerank）。
- **对比**：相同问题在 Advanced（静态开关）vs Modular（动态路由）下的路径差异。

## 新会话执行指引

1. 新会话从 `main` 创建 `dev/modular` 分支。
2. 参考本文件 + `docs/RAG演进阶段101.md` Stage 2 章节。
3. 不复用 Advanced 自实现模块——全部用 LangGraph 重写。
4. 先搭 State + Router + 条件边骨架（能跑通动态路由），再逐个填充模块节点。
5. 首批依赖：`uv add langgraph langchain-openai`。
6. SiliconFlow 接入用 `langchain-openai` 的 `ChatOpenAI` / `OpenAIEmbeddings`，`base_url` 指向 `https://api.siliconflow.cn/v1`。
