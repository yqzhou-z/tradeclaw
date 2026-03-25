from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


class OpenAIJsonClient:
    def __init__(
        self,
        enabled: bool = True,
        model: str = "gpt-4o-mini",
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
        if self.enabled:
            try:
                self.client = OpenAI(api_key=self.api_key)
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
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                timeout=self.timeout_sec,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return None
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return None
            return parsed
        except Exception:
            return None
