from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EvaluationMetrics:
    cycle_count: int = 0
    filled_trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    fill_rate: float = 0.0
    win_rate: float = 0.0
    avg_risk_score: float = 0.0
    avg_size_pct: float = 0.0
    realized_pnl_change: float = 0.0
    equity_change: float = 0.0
    max_drawdown_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(cycle_results: list[dict[str, Any]]) -> EvaluationMetrics:
    if not cycle_results:
        return EvaluationMetrics()

    cycle_count = len(cycle_results)
    filled = 0
    buy_count = 0
    sell_count = 0
    hold_count = 0
    risk_scores: list[float] = []
    size_values: list[float] = []
    equity_series: list[float] = []
    realized_series: list[float] = []
    trade_pnl_deltas: list[float] = []

    for idx, row in enumerate(cycle_results):
        final_decision = row.get("final_decision") or {}
        execution = row.get("execution_result") or {}
        risk_report = row.get("risk_report") or {}
        snapshot = row.get("portfolio_snapshot") or {}

        action = str(final_decision.get("action", "hold")).lower()
        if action == "buy":
            buy_count += 1
        elif action == "sell":
            sell_count += 1
        else:
            hold_count += 1

        if str(execution.get("status", "")).lower() == "filled":
            filled += 1

        risk_scores.append(_safe_float(risk_report.get("risk_score"), 0.0))
        size_values.append(_safe_float(final_decision.get("size_pct"), 0.0))
        equity_series.append(_safe_float(snapshot.get("total_equity"), 0.0))
        realized_series.append(_safe_float(snapshot.get("realized_pnl"), 0.0))

        if idx > 0:
            trade_pnl_deltas.append(realized_series[idx] - realized_series[idx - 1])

    non_zero_deltas = [d for d in trade_pnl_deltas if d != 0]
    win_count = sum(1 for d in non_zero_deltas if d > 0)
    win_rate = (win_count / len(non_zero_deltas)) if non_zero_deltas else 0.0

    max_drawdown = _compute_max_drawdown(equity_series)
    equity_change = equity_series[-1] - equity_series[0] if len(equity_series) >= 2 else 0.0
    realized_change = realized_series[-1] - realized_series[0] if len(realized_series) >= 2 else 0.0

    return EvaluationMetrics(
        cycle_count=cycle_count,
        filled_trade_count=filled,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_count=hold_count,
        fill_rate=round(filled / cycle_count, 4),
        win_rate=round(win_rate, 4),
        avg_risk_score=round(sum(risk_scores) / len(risk_scores), 4) if risk_scores else 0.0,
        avg_size_pct=round(sum(size_values) / len(size_values), 4) if size_values else 0.0,
        realized_pnl_change=round(realized_change, 6),
        equity_change=round(equity_change, 6),
        max_drawdown_pct=round(max_drawdown, 6),
    )


def summarize_metrics(metrics: EvaluationMetrics) -> str:
    return (
        f"cycles={metrics.cycle_count}, "
        f"filled={metrics.filled_trade_count}, "
        f"fill_rate={metrics.fill_rate:.2%}, "
        f"win_rate={metrics.win_rate:.2%}, "
        f"equity_change={metrics.equity_change:.4f}, "
        f"max_drawdown={metrics.max_drawdown_pct:.2%}"
    )


def _compute_max_drawdown(equity_series: list[float]) -> float:
    if not equity_series:
        return 0.0

    peak = equity_series[0]
    max_dd = 0.0
    for value in equity_series:
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
    return max_dd


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
