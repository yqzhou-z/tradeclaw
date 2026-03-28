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
        self.enabled = bool(enabled and self.api_key)
        self.client = None
        self.last_error: str | None = None
        if self.enabled:
            try:
                self.client = self._build_client()
            except Exception:
                self.client = None
                self.enabled = False

    def complete_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        model: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled or self.client is None:
            return None

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
            if parsed is not None:
                self.last_error = None
                return parsed
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

        # fallback for models that do not support response_format
        try:
            kwargs = self._base_request_kwargs(model or self.model)
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"{system_prompt} Return only JSON object."},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                **kwargs,
            )
            text = (response.choices[0].message.content or "").strip()
            parsed = self._parse_json_text(text)
            if parsed is not None:
                self.last_error = None
                return parsed
            self.last_error = "JSON parsing failed from fallback response."
            return None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def _parse_json_text(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
                return None
            except json.JSONDecodeError:
                return None

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
