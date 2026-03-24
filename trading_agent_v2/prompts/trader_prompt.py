from __future__ import annotations

import json
from typing import Any


TRADER_SYSTEM_PROMPT = """You are the final trader agent in a paper-trading workflow.
Fuse analyst views and risk constraints into one executable decision.
Do not force trades when confidence or edge is weak."""


def build_trader_prompt(
    symbol: str,
    analyst_views: list[dict[str, Any]],
    proposal: dict[str, Any] | None = None,
    risk_report: dict[str, Any] | None = None,
) -> str:
    payload = {
        "symbol": symbol,
        "analyst_views": analyst_views or [],
        "proposal": proposal or {},
        "risk_report": risk_report or {},
    }
    return (
        "Task:\n"
        "1) Decide action: buy / sell / hold.\n"
        "2) If buy/sell, provide size_pct plus optional SL/TP.\n"
        "3) State concise rationale and key conditions.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
