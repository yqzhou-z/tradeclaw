from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from trading_agent_v2.config import AppConfig, build_default_config
from trading_agent_v2.evaluation.metrics import EvaluationMetrics, compute_metrics
from trading_agent_v2.main import run_cycle


@dataclass
class BacktestResult:
    config_name: str
    symbols: list[str]
    cycles: int
    metrics: EvaluationMetrics
    cycle_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


class BacktestEngine:
    """
    Minimal backtest wrapper around run_cycle.
    It reuses the production pipeline with isolated data directories.
    """

    def __init__(self, cycle_runner: Callable[..., dict[str, Any]] | None = None):
        self.cycle_runner = cycle_runner or run_cycle

    def run(
        self,
        cycles: int = 20,
        symbols: list[str] | None = None,
        app_config: AppConfig | None = None,
        data_dir: str | Path | None = None,
        config_name: str = "baseline",
    ) -> BacktestResult:
        if cycles <= 0:
            raise ValueError("cycles must be > 0")

        base_config = app_config or build_default_config()
        config = deepcopy(base_config)

        if data_dir is not None:
            config.data_dir = Path(data_dir)
        config.data_dir.mkdir(parents=True, exist_ok=True)

        target_symbols = symbols or config.symbols
        results: list[dict[str, Any]] = []

        for _ in range(cycles):
            for symbol in target_symbols:
                result = self.cycle_runner(symbol=symbol, app_config=config)
                results.append(result)

        metrics = compute_metrics(results)
        return BacktestResult(
            config_name=config_name,
            symbols=target_symbols,
            cycles=cycles,
            metrics=metrics,
            cycle_results=results,
        )
