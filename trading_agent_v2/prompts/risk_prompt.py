from __future__ import annotations

import json
from typing import Any


RISK_MANAGER_SYSTEM_PROMPT = """You are a risk manager for an automated paper-trading stack.
Your first priority is capital preservation and stable risk exposure.
Reject proposals that are not executable or exceed exposure limits."""


def build_risk_prompt(
    proposal: dict[str, Any],
    portfolio: dict[str, Any],
    recent_episodes: list[dict[str, Any]] | None = None,
    strategy_memory: dict[str, Any] | None = None,
) -> str:
    payload = {
        "proposal": proposal or {},
        "portfolio": portfolio or {},
        "recent_episodes": recent_episodes or [],
        "strategy_memory": strategy_memory or {},
    }
    return (
        "Task:\n"
        "1) Decide approve/reject.\n"
        "2) Produce a risk score in [0, 1].\n"
        "3) If approved, propose conservative size/SL/TP adjustments.\n"
        "4) Explain warnings and hard rejection reasons.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
