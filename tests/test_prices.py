# tests/test_cost_estimation.py
import math

from llm_adapters import make_client

def test_cost_estimation_path(monkeypatch):
    client = make_client("openai", "gpt-4o-mini")

    # If your client stores a price record on init, replace it with a fake:
    if hasattr(client, "cost_model"):
        client.cost_model = {"input_per_1k": 1.23, "output_per_1k": 4.56}

    # Monkeypatch the HTTP call to return fixed usage tokens
    def _fake_post(self, url, headers, payload):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1234, "completion_tokens": 5678},
        }
    client.__class__._post = _fake_post

    text, usage = client.send(
        messages=[{"role": "user", "content": "test"}],
        temperature=0.2,
        max_tokens=16,
    )
    assert text == "ok"
    assert usage["prompt_tokens"] == 1234
    assert usage["completion_tokens"] == 5678

    if "cost_estimate_usd" in usage:
        expected = (1.234 * 1.23) + (5.678 * 4.56)
        assert math.isclose(usage["cost_estimate_usd"], expected, rel_tol=1e-6)