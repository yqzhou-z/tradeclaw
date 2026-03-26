from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent_v2.config import build_default_config
from trading_agent_v2.portfolio.portfolio_manager import PortfolioManager


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _backup_file(path: Path, backup_dir: Path) -> None:
    if not path.exists():
        return
    _ensure_parent(backup_dir / path.name)
    shutil.copy2(path, backup_dir / path.name)


def _reset_portfolio_baseline(portfolio_file: Path) -> dict[str, Any]:
    manager = PortfolioManager(str(portfolio_file))
    portfolio = manager.load_portfolio()
    now = utc_now_iso()

    total_market_value = 0.0
    reset_positions: dict[str, dict[str, Any]] = {}
    for symbol, pos in dict(portfolio.positions or {}).items():
        quantity = _safe_float(pos.get("quantity"))
        if abs(quantity) <= 1e-12:
            continue

        market_price = _safe_float(pos.get("market_price"))
        avg_entry = _safe_float(pos.get("avg_entry_price"))
        market_value = _safe_float(pos.get("market_value"))

        if market_price <= 0.0 and quantity > 0.0 and market_value > 0.0:
            market_price = market_value / quantity
        if market_price <= 0.0:
            market_price = avg_entry

        reset_market_value = quantity * market_price
        total_market_value += reset_market_value

        reset_positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "avg_entry_price": market_price,
            "market_price": market_price,
            "market_value": reset_market_value,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "updated_at": now,
        }

    portfolio.positions = reset_positions
    portfolio.realized_pnl = 0.0
    portfolio.total_equity = _safe_float(portfolio.cash) + total_market_value
    portfolio.updated_at = now
    manager.save_portfolio(portfolio)

    return manager.get_portfolio_snapshot(portfolio)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _truncate_file(path: Path) -> None:
    _ensure_parent(path)
    path.write_text("", encoding="utf-8")


def main() -> None:
    config = build_default_config()
    now = utc_now_iso()

    backup_root = Path(config.data_dir) / "reset_backups"
    backup_dir = backup_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    tracked_files = [
        Path(config.portfolio_file),
        Path(config.episodic_memory_file),
        Path(config.reflection_file),
        Path(config.strategy_memory_file),
        Path(config.pattern_memory_file),
        Path(config.run_log_file),
    ]
    for path in tracked_files:
        _backup_file(path, backup_dir)

    snapshot = _reset_portfolio_baseline(Path(config.portfolio_file))

    _truncate_file(Path(config.episodic_memory_file))
    _truncate_file(Path(config.reflection_file))
    _truncate_file(Path(config.run_log_file))

    _write_json(
        Path(config.strategy_memory_file),
        {
            "updated_at": now,
            "active_insights": [],
            "risk_adjustments": {},
            "performance_summary": {},
            "metadata": {
                "reset_at": now,
                "reset_reason": "manual_baseline_reset",
                "backup_dir": str(backup_dir),
            },
        },
    )
    _write_json(
        Path(config.pattern_memory_file),
        {
            "updated_at": now,
            "patterns": {},
            "metadata": {
                "reset_at": now,
                "reset_reason": "manual_baseline_reset",
                "backup_dir": str(backup_dir),
            },
        },
    )

    print("=" * 88)
    print("TRADING STATE RESET COMPLETE")
    print("=" * 88)
    print(f"Backup Dir        : {backup_dir}")
    print(f"Execution Mode    : {config.execution.mode}")
    print(f"Cash Baseline     : {snapshot.get('cash', 0.0):,.4f}")
    print(f"Equity Baseline   : {snapshot.get('total_equity', 0.0):,.4f}")
    print(f"Position Count    : {snapshot.get('position_count', 0)}")
    print("PnL Baseline      : realized=0.0000, unrealized=0.0000, total=0.0000")


if __name__ == "__main__":
    main()
