# tests/conftest.py
import os
import sys
import pytest

# Ensure project root is importable (so `from llm_adapters import ...` works)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture(autouse=True)
def _env_keys(monkeypatch):
    # Dummy keys so clients init without hitting env errors
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")