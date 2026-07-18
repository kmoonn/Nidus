"""Shared pytest fixtures for Nidus tests.

Modules under test read config via get_config(), which needs SILICONFLOW_API_KEY
to resolve ${...} placeholders. For pure-logic unit tests we point get_config at a
fixture config file with a dummy key so the config singleton loads cleanly.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure API key env exists so config ${SILICONFLOW_API_KEY} resolves in tests.
os.environ.setdefault("SILICONFLOW_API_KEY", "test-key-not-real")

from src.config import get_config  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset the config singleton between tests so isolated configs take effect."""
    import src.config as cfg_mod

    cfg_mod._config = None
    yield
    cfg_mod._config = None
