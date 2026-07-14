"""Configuration loader for Nidus RAG."""

import os
from pathlib import Path
from typing import Any

import yaml


def _resolve_env_vars(value: Any) -> Any:
    """Resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved = os.environ.get(env_var)
            if resolved is None:
                raise ValueError(
                    f"Environment variable {env_var} is not set. "
                    f"Please set it before running Nidus."
                )
            return resolved
        return value
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


class Config:
    """Global configuration object loaded from config.yaml."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self._data = _resolve_env_vars(raw)

    @property
    def llm(self) -> dict:
        return self._data["llm"]

    @property
    def embedding(self) -> dict:
        return self._data["embedding"]

    @property
    def chunker(self) -> dict:
        return self._data["chunker"]

    @property
    def retriever(self) -> dict:
        return self._data["retriever"]

    @property
    def store(self) -> dict:
        return self._data["store"]

    @property
    def docs_dir(self) -> Path:
        return Path(__file__).parent.parent / "docs" / "files"


# Global config singleton — loaded once on first access
_config: Config | None = None


def get_config(config_path: str | None = None) -> Config:
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config
