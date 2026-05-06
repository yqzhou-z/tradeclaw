from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from langsmith import utils as langsmith_utils
    from langsmith.wrappers import wrap_openai
except Exception:  # pragma: no cover
    langsmith_utils = None
    wrap_openai = None


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_API_KEY_ENV_VARS = ("TRADING_LLM_API_KEY", "DEEPSEEK_API_KEY")


class OpenAIJsonClient:
    """OpenAI SDK based JSON client for DeepSeek's OpenAI-compatible API."""

    def __init__(
        self,
        enabled: bool = True,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        timeout_sec: int = 30,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
        api_key: str | None = None,
    ):
        if load_dotenv is not None:
            load_dotenv()

        self.model = model
        self.base_url = (base_url or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)
        self.api_key = (api_key or self._api_key_from_env()).strip()
        if enabled and not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY or TRADING_LLM_API_KEY is required when TRADING_LLM_ENABLED=true.")

        self.enabled = bool(enabled and self.api_key)
        self.client = None
        self.last_error: str | None = None
        if self.enabled:
            self.client = self._build_client()

    def complete_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled or self.client is None:
            raise RuntimeError("LLM JSON client is disabled or not initialized.")

        try:
            kwargs = self._base_request_kwargs(model or self.model)
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                response_format={"type": "json_object"},
                **kwargs,
            )
            text = (response.choices[0].message.content or "").strip()
            parsed = self._parse_json_text(text)
            self.last_error = None
            return parsed
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"DeepSeek JSON completion failed: {exc}") from exc

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        if not text:
            raise ValueError("DeepSeek returned an empty response.")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek response must be a JSON object.")
        return parsed

    def _base_request_kwargs(self, model: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout_sec,
        }
        if self._is_deepseek_model(model):
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if self.thinking_enabled else "disabled"},
            }
            if self.thinking_enabled:
                kwargs["reasoning_effort"] = self.reasoning_effort
        if not self._is_reasoning_model(model):
            kwargs["temperature"] = self.temperature
        return kwargs

    def _is_reasoning_model(self, model: str) -> bool:
        name = (model or "").strip().lower()
        return name.startswith("o")

    def _is_deepseek_model(self, model: str) -> bool:
        name = (model or "").strip().lower()
        return name.startswith("deepseek-")

    def _normalize_reasoning_effort(self, value: str) -> str:
        name = (value or "").strip().lower()
        return "max" if name in {"max", "xhigh"} else "high"

    def _api_key_from_env(self) -> str:
        for env_name in DEFAULT_API_KEY_ENV_VARS:
            value = os.getenv(env_name, "").strip()
            if value:
                return value
        return ""

    def _build_client(self) -> Any:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        if wrap_openai is None or langsmith_utils is None:
            return client

        cache_clear = getattr(getattr(langsmith_utils, "get_env_var", None), "cache_clear", None)
        if callable(cache_clear):
            cache_clear()

        try:
            if langsmith_utils.tracing_is_enabled():
                return wrap_openai(
                    client,
                    chat_name="TradingAgentDeepSeekChat",
                    completions_name="TradingAgentDeepSeek",
                )
        except Exception:
            return client

        return client
