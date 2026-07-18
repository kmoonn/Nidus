"""Tests for config loading — Advanced block + reranker exposure."""

import os
import yaml

import src.config as cfg_mod
from src.config import Config


def _load(config_dict):
    import tempfile

    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(config_dict, f)
    f.close()
    cfg_mod._config = Config(f.name)
    return cfg_mod._config


def _minimal():
    return {
        "llm": {"model": "m", "base_url": "u", "api_key": "${SILICONFLOW_API_KEY}"},
        "embedding": {"model": "m", "base_url": "u", "api_key": "${SILICONFLOW_API_KEY}"},
        "reranker": {"model": "r", "base_url": "u", "api_key": "${SILICONFLOW_API_KEY}"},
        "chunker": {"chunk_size": 500, "chunk_overlap": 50},
        "retriever": {"top_k": 5, "relevance_threshold": 0.5},
        "store": {"type": "chromadb", "persist_directory": "x", "collection_name": "c"},
        "advanced": {
            "hybrid_search": {"enabled": True},
            "reranking": {"enabled": True, "top_n": 5},
        },
    }


def test_env_var_resolution():
    os.environ["SILICONFLOW_API_KEY"] = "key123"
    cfg = _load(_minimal())
    assert cfg.llm["api_key"] == "key123"
    assert cfg.reranker["api_key"] == "key123"


def test_advanced_block_exposed():
    cfg = _load(_minimal())
    adv = cfg.advanced
    assert adv["hybrid_search"]["enabled"] is True
    assert adv["reranking"]["top_n"] == 5


def test_advanced_block_defaults_empty_when_absent():
    """Config without an advanced block should not crash."""
    d = _minimal()
    del d["advanced"]
    del d["reranker"]
    cfg = _load(d)
    assert cfg.advanced == {}
    assert cfg.reranker == {}
