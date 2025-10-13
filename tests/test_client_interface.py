# tests/test_client_interface.py
import pytest
from llm_adapters import make_client, OpenAIClient, GrokClient, LLMClient

@pytest.fixture
def sample_messages():
    return [{"role": "user", "content": "Say hi"}]

def test_factory_returns_correct_classes():
    """Ensure make_client returns the right concrete client classes."""
    oai = make_client("openai", "gpt-4o-mini")
    grok = make_client("grok", "grok-4-fast")

    assert isinstance(oai, OpenAIClient)
    assert isinstance(grok, GrokClient)
    assert isinstance(oai, LLMClient)
    assert isinstance(grok, LLMClient)

def test_send_method_signature(monkeypatch, sample_messages):
    """Check that .send exists, accepts messages/temperature/max_tokens, and returns a (text, usage) tuple."""
    oai = make_client("openai", "gpt-4o-mini")
    grok = make_client("grok", "grok-4-fast")

    # fake _post so no HTTP is made
    def fake_post(self, url, headers, payload):
        return {
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }

    monkeypatch.setattr(oai.__class__, "_post", fake_post)
    monkeypatch.setattr(grok.__class__, "_post", fake_post)

    for client in (oai, grok):
        text, usage = client.send(messages=sample_messages, temperature=0.7, max_tokens=16)
        assert isinstance(text, str)
        assert isinstance(usage, dict)
        assert "cost_estimate_usd" in usage
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage

def test_factory_raises_on_unknown_vendor():
    """Invalid vendor should raise ValueError."""
    with pytest.raises(ValueError):
        make_client("unknown_vendor", "model-x")