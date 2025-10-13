# llm_adapters.py
# Unified adapters for OpenAI + Grok (x.ai) with YAML/JSON pricing, validation, retry logic,
# and Grok auto-fallback when penalties are not supported.

from __future__ import annotations

import os
import json
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception:
    raise SystemExit("The 'requests' package is required. Install with:\n  python3 -m pip install requests")

# YAML is optional; we enable it if available and prefer .yaml if present
try:
    import yaml  # PyYAML
except Exception:
    yaml = None  # We'll gracefully handle absence


# =====================================================================
# Public Interface
# =====================================================================

class LLMClient:
    """
    Base interface for LLM clients.
    Must return: (text, usage_dict) where usage_dict contains:
      - prompt_tokens (int, optional)
      - completion_tokens (int, optional)
      - cost_estimate_usd (float, optional)
      - price_source (str, optional)
    """

    def send(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 512,
        **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError


# =====================================================================
# Pricing + Token Estimation
# =====================================================================

@dataclass
class Pricebook:
    """
    Holds a flattened model->price map:
      models = {
        "vendor:model": {"input_per_1k": float, "output_per_1k": float},
        "model":        {"input_per_1k": float, "output_per_1k": float},
      }
    """
    models: Dict[str, Dict[str, float]]

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Pricebook":
        """
        Load prices from YAML or JSON.

        Search order if path is None:
          1) config/prices.yaml  (preferred)
          2) config/prices.json

        Supported schemas:
        A) Vendor-structured YAML:
           prices:
             vendor:
               input_per_1k: <float>   # vendor default
               output_per_1k: <float>
               models:
                 model_name:
                   input_per_1k: <float>   # overrides vendor default
                   output_per_1k: <float>

        B) Flat JSON:
           {
             "models": {
               "vendor:model": {"input_per_1k":..., "output_per_1k":...},
               "model": {"input_per_1k":..., "output_per_1k":...}
             }
           }
        """
        root_dir = os.path.dirname(__file__)
        default_yaml = os.path.join(root_dir, "config", "prices.yaml")
        default_json = os.path.join(root_dir, "config", "prices.json")

        p = path or (default_yaml if os.path.exists(default_yaml) else default_json)
        if not p or not os.path.exists(p):
            print("PRICEBOOK: No price file found; proceeding with empty pricebook.")
            return cls(models={})

        # Parse file
        try:
            if p.endswith((".yaml", ".yml")):
                if yaml is None:
                    raise RuntimeError("PyYAML not installed. Install with: python3 -m pip install pyyaml")
                with open(p, "r") as f:
                    data = yaml.safe_load(f) or {}
            else:
                with open(p, "r") as f:
                    data = json.load(f) or {}
        except Exception as e:
            print(f"PRICEBOOK LOAD ERROR for {p}: {e}")
            return cls(models={})

        flat: Dict[str, Dict[str, float]] = {}

        # Vendor-structured schema
        if isinstance(data, dict) and "prices" in data:
            prices = data.get("prices") or {}
            if not isinstance(prices, dict):
                print("PRICEBOOK WARNING: 'prices' must be a mapping; got:", type(prices).__name__)
                return cls(models={})

            for vendor, vend_conf in prices.items():
                if not isinstance(vend_conf, dict):
                    print(f"PRICEBOOK WARNING: vendor '{vendor}' config must be a mapping; got:", type(vend_conf).__name__)
                    continue

                v_in = vend_conf.get("input_per_1k")
                v_out = vend_conf.get("output_per_1k")
                if (v_in is None) or (v_out is None):
                    print(f"PRICEBOOK WARNING: vendor '{vendor}' missing defaults 'input_per_1k'/'output_per_1k'. "
                          f"Model entries must include both values explicitly.")
                models = vend_conf.get("models") or {}
                if not isinstance(models, dict):
                    print(f"PRICEBOOK WARNING: vendor '{vendor}'.models must be a mapping; got:", type(models).__name__)
                    continue

                for model, mm in models.items():
                    if not isinstance(mm, dict):
                        print(f"PRICEBOOK WARNING: model '{model}' under vendor '{vendor}' must be a mapping; got:", type(mm).__name__)
                        continue
                    m_in = mm.get("input_per_1k", v_in)
                    m_out = mm.get("output_per_1k", v_out)
                    # Validate fields
                    if (m_in is None) or (m_out is None):
                        print(f"PRICEBOOK WARNING: '{vendor}:{model}' missing 'input_per_1k' or 'output_per_1k'. Skipping.")
                        continue

                    # Store both vendor:model and model for flexible lookup
                    flat[f"{vendor}:{model}"] = {"input_per_1k": float(m_in), "output_per_1k": float(m_out)}
                    # plain model key (first hit wins for cross-vendor duplicates)
                    if model not in flat:
                        flat[model] = {"input_per_1k": float(m_in), "output_per_1k": float(m_out)}

            return cls(models=flat)

        # Flat JSON schema
        models = data.get("models") if isinstance(data, dict) else None
        if isinstance(models, dict):
            cleaned = {}
            for k, v in models.items():
                try:
                    cleaned[k] = {
                        "input_per_1k": float(v["input_per_1k"]),
                        "output_per_1k": float(v["output_per_1k"]),
                    }
                except Exception:
                    print(f"PRICEBOOK WARNING: model '{k}' missing or invalid numeric fields; skipping.")
            return cls(models=cleaned)

        print("PRICEBOOK WARNING: Unrecognized pricebook schema; proceeding empty.")
        return cls(models={})

    def find(self, vendor: str, model: str) -> Tuple[Optional[float], Optional[float], str]:
        """
        Lookup pricing; search order:
          1) "vendor:model"
          2) "model"
        Returns (input_per_1k, output_per_1k, source_tag)
        """
        key_vm = f"{vendor}:{model}"
        if key_vm in self.models:
            m = self.models[key_vm]
            return (m.get("input_per_1k"), m.get("output_per_1k"), "pricebook_vendor_model")
        if model in self.models:
            m = self.models[model]
            return (m.get("input_per_1k"), m.get("output_per_1k"), "pricebook_model")
        return (None, None, "none")


def rough_chat_token_count(messages: List[Dict[str, str]]) -> int:
    """
    Very rough token estimator for budget protection.
    ~4 chars/token heuristic + small per-message overhead.
    """
    if not messages:
        return 0
    total_chars = 0
    for m in messages:
        total_chars += len(m.get("content", "") or "")
        total_chars += 8  # role/format overhead
    return max(1, math.ceil(total_chars / 4))


# =====================================================================
# OpenAI Client
# =====================================================================

OPENAI_MODELS: Dict[str, str] = {
    # Current (2025)
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "o3": "o3",
    "o3-mini": "o3-mini",
    # Legacy aliases (kept for config compatibility)
    "gpt-4-turbo": "gpt-4o",
    "gpt-4": "gpt-4o",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}

class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        pricebook: Optional[Pricebook] = None,
    ):
        self.vendor = "openai"
        self.model = OPENAI_MODELS.get(model, model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.pricebook = pricebook or Pricebook.load()

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> Tuple[Optional[float], str]:
        ip, op, src = self.pricebook.find(self.vendor, self.model)
        if ip is None or op is None:
            return (None, "none")
        cost = (prompt_tokens/1000.0)*(ip or 0.0) + (completion_tokens/1000.0)*(op or 0.0)
        return (round(cost, 6), src)

    def send(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 512,
        **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.api_key:
            return "(stubbed OpenAI) Missing OPENAI_API_KEY.", {"cost_estimate_usd": 0.0}

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Optional params (only if present)
        for k in ("frequency_penalty", "presence_penalty", "top_p", "stop", "logit_bias", "n"):
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]

        # Two attempts on 429 or transient 5xx
        for attempt in (1, 2):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt == 1:
                    time.sleep(2.0)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage") or {}
                p_tok = int(usage.get("prompt_tokens") or 0)
                c_tok = int(usage.get("completion_tokens") or 0)
                cost, src = self._estimate_cost(p_tok, c_tok)
                return text, {
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "cost_estimate_usd": 0.0 if cost is None else cost,
                    "price_source": src,
                }
            except requests.HTTPError as e:
                body = ""
                try:
                    body = resp.text[:600]
                except Exception:
                    pass
                print(f"OPENAI HTTP ERROR: {resp.status_code if resp else None} :: {e} :: {body}")
                return f"(stubbed OpenAI) HTTP {getattr(resp,'status_code',None)}", {"cost_estimate_usd": 0.0}
            except Exception as e:
                print("OPENAI CLIENT ERROR:", repr(e))
                return f"(stubbed OpenAI) Error: {e}", {"cost_estimate_usd": 0.0}


# =====================================================================
# Grok (x.ai) Client
# =====================================================================

GROK_MODELS: Dict[str, str] = {
    # Adjust these to what your x.ai account supports
    "grok-4-fast": "grok-4-fast",
    "grok-4": "grok-4",
    "grok-2": "grok-2",
    "grok-beta": "grok-beta",
    "grok-1": "grok-1",
}

class GrokClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        pricebook: Optional[Pricebook] = None,
    ):
        self.vendor = "grok"
        self.model = GROK_MODELS.get(model, model)
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        # x.ai is OpenAI-compatible
        self.base_url = base_url or os.getenv("XAI_BASE_URL") or "https://api.x.ai/v1"
        self.pricebook = pricebook or Pricebook.load()

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> Tuple[Optional[float], str]:
        ip, op, src = self.pricebook.find(self.vendor, self.model)
        if ip is None or op is None:
            return (None, "none")
        cost = (prompt_tokens/1000.0)*(ip or 0.0) + (completion_tokens/1000.0)*(op or 0.0)
        return (round(cost, 6), src)

    def send(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 512,
        **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.api_key:
            return "(stubbed Grok) Missing XAI_API_KEY.", {"cost_estimate_usd": 0.0}

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        def _build_payload(allow_penalties: bool) -> Dict[str, Any]:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            # Only include penalties if allowed (some Grok models reject them)
            if allow_penalties:
                for k in ("frequency_penalty", "presence_penalty"):
                    if k in kwargs and kwargs[k] is not None:
                        payload[k] = kwargs[k]
            # Other optional params are fine
            for k in ("top_p", "stop", "logit_bias", "n"):
                if k in kwargs and kwargs[k] is not None:
                    payload[k] = kwargs[k]
            return payload

        attempts = [
            _build_payload(allow_penalties=True),   # first try with penalties
            _build_payload(allow_penalties=False),  # fallback without penalties
        ]

        for i, payload in enumerate(attempts, start=1):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
                # 429: simple backoff once per attempt
                if resp.status_code == 429:
                    time.sleep(2.0)
                    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

                # If first attempt got 400 and body mentions unsupported penalty, retry without penalties
                if resp.status_code == 400 and i == 1:
                    body = (resp.text or "").lower()
                    if ("presencepenalty" in body) or ("frequencypenalty" in body) or ("does not support parameter" in body):
                        # go to next iteration (no penalties)
                        continue

                resp.raise_for_status()
                data = resp.json()
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage") or {}
                p_tok = int(usage.get("prompt_tokens") or 0)
                c_tok = int(usage.get("completion_tokens") or 0)
                cost, src = self._estimate_cost(p_tok, c_tok)
                return text, {
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "cost_estimate_usd": 0.0 if cost is None else cost,
                    "price_source": src,
                }
            except requests.HTTPError as e:
                body = ""
                try:
                    body = resp.text[:600]
                except Exception:
                    pass
                print(f"GROK HTTP ERROR: {resp.status_code if resp else None} :: {e} :: {body}")
                # If this was the first attempt and it's clearly a penalty issue, try second attempt
                if (resp is not None) and (resp.status_code == 400) and i == 1:
                    lb = (body or "").lower()
                    if ("presencepenalty" in lb) or ("frequencypenalty" in lb) or ("does not support parameter" in lb):
                        continue
                return f"(stubbed Grok) HTTP {getattr(resp,'status_code',None)}", {"cost_estimate_usd": 0.0}
            except Exception as e:
                print("GROK CLIENT ERROR:", repr(e))
                return f"(stubbed Grok) Error: {e}", {"cost_estimate_usd": 0.0}


# =====================================================================
# Factory
# =====================================================================

def make_client(vendor: str, model: str) -> LLMClient:
    """
    Factory that returns an LLMClient given a vendor + model.
    """
    v = (vendor or "").strip().lower()
    m = (model or "").strip()
    pricebook = Pricebook.load()

    if v in ("openai", "oai"):
        return OpenAIClient(m, pricebook=pricebook)
    if v in ("grok", "xai", "x.ai", "x_ai"):
        return GrokClient(m, pricebook=pricebook)

    raise ValueError(f"Unknown vendor '{vendor}'. Supported: openai, grok")


# =====================================================================
# Convenience Exports
# =====================================================================

__all__ = [
    "LLMClient",
    "OpenAIClient",
    "GrokClient",
    "make_client",
    "rough_chat_token_count",
    "Pricebook",
]