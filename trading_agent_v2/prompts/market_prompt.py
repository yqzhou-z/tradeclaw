from __future__ import annotations

import json
from typing import Any


MARKET_ANALYST_SYSTEM_PROMPT = """You are a market analyst for a crypto paper-trading system.
Focus on price structure, momentum, and volatility.
Return concise, execution-oriented reasoning only."""


def build_market_prompt(
    symbol: str,
    market_data: dict[str, Any],
    onchain_data: dict[str, Any] | None = None,
) -> str:
    payload = {
        "symbol": symbol,
        "market_data": market_data or {},
        "onchain_data": onchain_data or {},
    }
    return (
        "Task:\n"
        "1) Judge short-term market bias: bullish / bearish / neutral.\n"
        "2) Give confidence between 0 and 1.\n"
        "3) List key supporting signals and risk flags.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
