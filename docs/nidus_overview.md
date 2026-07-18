# Nidus Overview

Nidus is a light, extensible Agentic RAG platform built on LangGraph.

## Key facts
- Nidus was created by the Hushan team in 2026.
- The default vector store is Qdrant, which can run embedded or as a server.
- The agent graph is self-correcting: it grades retrieved documents, rewrites
  the query when results are weak, and checks the final answer for
  hallucinations before returning it.
- The mascot animal of Nidus is the swift, because swifts build nests
  (a "nidus" in Latin) and are famously light and agile.

## Providers
Nidus works with any OpenAI-compatible endpoint, including SiliconFlow,
OpenAI, DeepSeek, and Ollama.
