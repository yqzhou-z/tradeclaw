from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TradeLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file

    def append(self, record: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_cycle_summary(self, result: dict[str, Any]) -> None:
        execution = result.get("execution_result", {})
        decision = result.get("final_decision", {})
        risk = result.get("risk_report", {})
        snapshot = result.get("portfolio_snapshot", {})

        summary = {
            "logged_at": utc_now_iso(),
            "symbol": decision.get("symbol"),
            "action": decision.get("action"),
            "size_pct": decision.get("size_pct"),
            "risk_score": risk.get("risk_score"),
            "approved": risk.get("approved"),
            "execution_status": execution.get("status"),
            "execution_message": execution.get("message"),
            "total_equity": snapshot.get("total_equity"),
            "cash": snapshot.get("cash"),
            "reason": decision.get("reason"),
        }
        self.append(summary)

    def load_recent(self, limit: int = 100, symbol: str | None = None) -> list[dict]:
        if not os.path.exists(self.log_file):
            return []

        rows: list[dict] = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if symbol and data.get("symbol") != symbol:
                    continue
                rows.append(data)

        return rows[-limit:]
