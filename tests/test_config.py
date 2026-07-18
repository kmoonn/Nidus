"""Tests for configuration loading and env overrides."""

from __future__ import annotations

import importlib


def _fresh_settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = importlib.import_module("nidus.config")
    config.get_settings.cache_clear()
    return config.get_settings()


def test_defaults(monkeypatch):
    # Verify the in-code defaults, independent of any local ``.env`` file or
    # ambient NIDUS_ vars (``_env_file=None`` disables .env loading).
    for var in list(__import__("os").environ):
        if var.startswith("NIDUS_"):
            monkeypatch.delenv(var, raising=False)
    config = importlib.import_module("nidus.config")
    settings = config.Settings(_env_file=None)
    assert settings.qdrant_collection == "nidus"
    assert settings.qdrant_url is None
    assert settings.embed_dim == 1536
    assert settings.max_retries == 2


def test_env_override(monkeypatch):
    settings = _fresh_settings(
        monkeypatch,
        NIDUS_LLM_MODEL="doubao-pro-32k",
        NIDUS_QDRANT_COLLECTION="mycoll",
        NIDUS_MAX_RETRIES="5",
    )
    assert settings.llm_model == "doubao-pro-32k"
    assert settings.qdrant_collection == "mycoll"
    assert settings.max_retries == 5


def test_embed_credentials_fallback_to_llm(monkeypatch):
    settings = _fresh_settings(
        monkeypatch,
        NIDUS_LLM_BASE_URL="https://example.com/v1",
        NIDUS_LLM_API_KEY="key-abc",
    )
    monkeypatch.delenv("NIDUS_EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("NIDUS_EMBED_API_KEY", raising=False)
    assert settings.resolved_embed_base_url == "https://example.com/v1"
    assert settings.resolved_embed_api_key == "key-abc"


def test_embed_credentials_explicit(monkeypatch):
    settings = _fresh_settings(
        monkeypatch,
        NIDUS_LLM_BASE_URL="https://chat.example/v1",
        NIDUS_EMBED_BASE_URL="https://embed.example/v1",
        NIDUS_EMBED_API_KEY="embed-key",
    )
    assert settings.resolved_embed_base_url == "https://embed.example/v1"
    assert settings.resolved_embed_api_key == "embed-key"
