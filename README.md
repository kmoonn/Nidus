# Nidus

Light Extensible Agentic RAG — 轻量级可扩展 Agentic RAG 框架

## 简介

Nidus 是一个从零实现的 Native RAG 系统，不依赖 LangChain/LlamaIndex 等重型框架，帮助理解 RAG 核心原理。

**核心流程**：文档加载 → 文本分块 → 向量嵌入 → 相似度检索 → LLM 生成

## 技术栈

| 组件 | 方案 |
|------|------|
| LLM | SiliconFlow API (DeepSeek-V3) |
| Embedding | SiliconFlow API (BAAI/bge-large-zh-v1.5) |
| Vector Store | ChromaDB |
| PDF 解析 | pypdf |
| REST API | FastAPI |
| 包管理 | uv |

## 快速开始

### 1. 环境准备

```bash
# 安装 uv (如果还没有)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync
```

### 2. 配置 API Key

```bash
export SILICONFLOW_API_KEY="your-api-key-here"
```

API Key 可在 [SiliconFlow](https://cloud.siliconflow.cn/) 注册获取。

### 3. 构建索引

```bash
uv run python -m src.main index
```

扫描 `docs/` 目录下的 PDF 文件，进行解析、分块、嵌入并存入 ChromaDB。

### 4. 问答

```bash
# 单次提问
uv run python -m src.main ask "碳达峰的目标年份是什么？"

# 交互模式
uv run python -m src.main interactive
```

### 5. REST API 服务

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

启动后访问 http://127.0.0.1:8000/docs 查看 Swagger UI 在线调试。

**API 接口：**

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回索引状态 |
| POST | `/index` | 构建或重建文档索引 |
| POST | `/ask` | 问答接口 |

**示例请求：**

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 构建索引
curl -X POST http://127.0.0.1:8000/index

# 提问
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "碳中和的目标年份是什么？"}'
```

## 项目结构

```
Nidus/
├── config.yaml          # 统一配置文件
├── pyproject.toml       # 项目依赖
├── docs/                # 源文档 (PDF)
├── src/
│   ├── config.py        # 配置加载
│   ├── loader.py        # PDF 文档加载
│   ├── chunker.py       # 文本分块 (固定大小 + 重叠)
│   ├── embedder.py      # 文本嵌入 (SiliconFlow API)
│   ├── store.py         # 向量存储 (ChromaDB)
│   ├── retriever.py     # 相似度检索
│   ├── generator.py     # LLM 答案生成
│   ├── pipeline.py      # RAG 流水线编排
│   ├── api.py           # FastAPI REST API
│   └── main.py          # CLI 入口
└── data/                # 运行时数据 (gitignore)
    └── chroma_db/       # ChromaDB 持久化
```

## 相似度检索

向量检索使用 **余弦相似度 (Cosine Similarity)**，在 ChromaDB 中通过 HNSW 索引实现（`hnsw:space: cosine`）。

**计算公式**：

```
cos(A, B) = (A · B) / (|A| × |B|)
```

衡量两个向量的方向相似性，忽略向量长度，只看夹角：

| cos 值 | 含义 |
|--------|------|
| 1 | 方向完全相同（最相似） |
| 0 | 正交（无关） |
| -1 | 方向相反（最不相似） |

API 返回的 `distance` 字段 = `1 - cos(A, B)`，所以 **distance 越小越相似**（0 = 完全相同）。

> 文本 Embedding 模型（如 BGE）输出的向量通常已归一化，此时余弦相似度与 L2 距离、内积等价。Cosine 是最通用、最安全的选择。

## 配置说明

编辑 `config.yaml` 可调整所有参数：

- `llm` — LLM 模型和 API 配置
- `embedding` — 嵌入模型配置
- `chunker` — 分块大小和重叠
- `retriever` — 检索 top-k 数量
- `store` — ChromaDB 存储路径和集合名

## License

MIT
