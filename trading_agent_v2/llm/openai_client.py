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


class OpenAIJsonClient:
    def __init__(
        self,
        enabled: bool = True,
        model: str = "gpt-5.4",
        temperature: float = 0.2,
        max_tokens: int = 1200,
        timeout_sec: int = 30,
        api_key: str | None = None,
    ):
        if load_dotenv is not None:
            load_dotenv()

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if enabled and not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when TRADING_LLM_ENABLED=true.")

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
            raise RuntimeError("OpenAI JSON client is disabled or not initialized.")

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
            raise RuntimeError(f"OpenAI JSON completion failed: {exc}") from exc

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        if not text:
            raise ValueError("OpenAI returned an empty response.")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI response must be a JSON object.")
        return parsed

    def _base_request_kwargs(self, model: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": self.max_tokens,
            "timeout": self.timeout_sec,
        }
        if not self._is_reasoning_model(model):
            kwargs["temperature"] = self.temperature
        return kwargs

    def _is_reasoning_model(self, model: str) -> bool:
        name = (model or "").strip().lower()
        return name.startswith("o")

    def _build_client(self) -> Any:
        client = OpenAI(api_key=self.api_key)
        if wrap_openai is None or langsmith_utils is None:
            return client

        cache_clear = getattr(getattr(langsmith_utils, "get_env_var", None), "cache_clear", None)
        if callable(cache_clear):
            cache_clear()

        try:
            if langsmith_utils.tracing_is_enabled():
                return wrap_openai(
                    client,
                    chat_name="TradingAgentOpenAIChat",
                    completions_name="TradingAgentOpenAI",
                )
        except Exception:
            return client

        return client
