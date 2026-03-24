from __future__ import annotations

import json
from typing import Any


NEWS_ANALYST_SYSTEM_PROMPT = """You are a news and sentiment analyst for a crypto paper-trading system.
Focus on headline tone, social momentum, and reversal risk.
Prefer robust and conservative judgment when signals conflict."""


def build_news_prompt(
    symbol: str,
    news_items: list[dict[str, Any]],
    social_data: dict[str, Any] | None = None,
    onchain_data: dict[str, Any] | None = None,
) -> str:
    payload = {
        "symbol": symbol,
        "news_items": news_items or [],
        "social_data": social_data or {},
        "onchain_data": onchain_data or {},
    }
    return (
        "Task:\n"
        "1) Judge short-term bias from news+social: bullish / bearish / neutral.\n"
        "2) Give confidence between 0 and 1.\n"
        "3) Highlight catalysts and possible reversal risks.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
