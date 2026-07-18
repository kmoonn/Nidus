"""LangGraph node callables for Nidus Modular RAG.

每个节点是 `state -> state` 的纯函数风格 callable。失败静默回退，绝不阻塞主流程
（沿用 Advanced RAG 的鲁棒性约定）。节点间通过 GraphState（见 src/graph.py）传递数据。
"""
