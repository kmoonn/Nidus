# Nidus
Light Extensible Agentic RAG — 复现 RAG 演进阶段

## 简介

**Nidus 是一个复现 RAG 演进阶段的学习项目**，按 Native → Advanced → Modular → Agentic 四阶段递进实现：

- **Native RAG**（`dev/native`）：从零实现的朴素 RAG（loader → chunker → embedder → store → retriever → generator）。
- **Advanced RAG**（`dev/advanced`）：在朴素 RAG 上加 Pre/Post-Retrieval 优化（改写/分解/扩展/HyDE、混合检索+RRF、dedup、rerank、压缩），固定线性流水线 + 配置静态开关。
- **Modular RAG**（`dev/modular`，本分支）：引入 **LangGraph** 编排，运行时**动态路由** + 可组合模块网络。Router 按问题类型把不同问题路由到不同模块链（简单→快速路径、复杂→深度路径、时效型→web 搜索路径）。

底层基础设施（ChromaDB 向量存储、PDF 加载、SiliconFlow LLM/Embedding）从零或经轻量封装实现；Modular 阶段的编排层用 LangGraph。

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置 API Key（SiliconFlow，OpenAI 兼容）
export SILICONFLOW_API_KEY=your_key

# 3. 建索引（PDF 语料在 docs/files/）
uv run python -m src.main index

# 4. 提问（CLI）
uv run python -m src.main ask "碳达峰的目标年份？"
uv run python -m src.main interactive

# 5. 或启动 API + Web UI
uv run uvicorn src.api:app
```

## 文档

- [RAG 演进阶段 101](docs/RAG演进阶段101.md) — 各阶段理论。
- [Modular RAG 计划](docs/Modular-RAG计划.md) — 本阶段设计决策。
- [Modular RAG 实现说明](docs/Modular-RAG实现说明.md) — 每个模块"问题→实现→验证"。
