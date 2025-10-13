# tests/test_clients_payloads.py
import pytest
from llm_adapters import OpenAIClient, GrokClient

@pytest.fixture
def sample_messages():
    return [{"role": "user", "content": "Say hello"}]

def test_openai_includes_penalties(monkeypatch, sample_messages):
    sent_payloads = {}

    def fake_post(self, url, headers, payload):
        sent_payloads["payload"] = payload
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}

    monkeypatch.setattr(OpenAIClient, "_post", fake_post)
    client = OpenAIClient("gpt-4o-mini", api_key="sk-test")
    client.send(messages=sample_messages, temperature=0.7, presence_penalty=0.3, frequency_penalty=0.2)
    payload = sent_payloads["payload"]

    # OpenAI supports penalties, so they should appear
    assert "presence_penalty" in payload
    assert "frequency_penalty" in payload
    assert payload["presence_penalty"] == 0.3
    assert payload["frequency_penalty"] == 0.2


def test_grok_excludes_penalties(monkeypatch, sample_messages):
    sent_payloads = {}

    def fake_post(self, url, headers, payload):
        sent_payloads["payload"] = payload
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}

    monkeypatch.setattr(GrokClient, "_post", fake_post)
    client = GrokClient("grok-4-fast", api_key="xai-test")
    client.send(messages=sample_messages, temperature=0.7, presence_penalty=0.3, frequency_penalty=0.2)
    payload = sent_payloads["payload"]

    # Grok should *not* include unsupported penalties
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload
    assert "model" in payload
    assert "messages" in payload