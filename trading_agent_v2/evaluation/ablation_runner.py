from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_agent_v2.config import AppConfig, build_default_config
from trading_agent_v2.evaluation.backtest_engine import BacktestEngine, BacktestResult


@dataclass
class AblationSummary:
    variant_name: str
    equity_change: float
    win_rate: float
    fill_rate: float
    max_drawdown_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "equity_change": self.equity_change,
            "win_rate": self.win_rate,
            "fill_rate": self.fill_rate,
            "max_drawdown_pct": self.max_drawdown_pct,
        }


class AblationRunner:
    """
    Runs controlled config variants against the same cycle runner.
    """

    def __init__(self, backtest_engine: BacktestEngine | None = None):
        self.backtest_engine = backtest_engine or BacktestEngine()

    def run(
        self,
        cycles: int = 20,
        symbols: list[str] | None = None,
        base_config: AppConfig | None = None,
        output_root: str | Path | None = None,
    ) -> dict[str, Any]:
        config = base_config or build_default_config()
        output_dir = Path(output_root or (config.data_dir / "ablation"))
        output_dir.mkdir(parents=True, exist_ok=True)

        variants = self._build_variants(config)
        results: dict[str, BacktestResult] = {}
        summaries: list[AblationSummary] = []

        for name, variant_config in variants.items():
            variant_data_dir = output_dir / name
            result = self.backtest_engine.run(
                cycles=cycles,
                symbols=symbols,
                app_config=variant_config,
                data_dir=variant_data_dir,
                config_name=name,
            )
            results[name] = result
            summaries.append(
                AblationSummary(
                    variant_name=name,
                    equity_change=result.metrics.equity_change,
                    win_rate=result.metrics.win_rate,
                    fill_rate=result.metrics.fill_rate,
                    max_drawdown_pct=result.metrics.max_drawdown_pct,
                )
            )

        summaries.sort(key=lambda x: x.equity_change, reverse=True)

        return {
            "ranking": [item.to_dict() for item in summaries],
            "variants": {name: result.to_dict() for name, result in results.items()},
        }

    def _build_variants(self, config: AppConfig) -> dict[str, AppConfig]:
        baseline = deepcopy(config)

        no_news_weight = deepcopy(config)
        no_news_weight.trader.analyst_weights["news_analyst"] = 0.0

        no_market_weight = deepcopy(config)
        no_market_weight.trader.analyst_weights["market_analyst"] = 0.0

        tight_risk = deepcopy(config)
        tight_risk.risk.max_total_invested_pct = min(config.risk.max_total_invested_pct, 0.60)
        tight_risk.risk.max_single_position_pct = min(config.risk.max_single_position_pct, 0.20)

        loose_risk = deepcopy(config)
        loose_risk.risk.max_total_invested_pct = max(config.risk.max_total_invested_pct, 0.95)
        loose_risk.risk.max_single_position_pct = max(config.risk.max_single_position_pct, 0.50)

        return {
            "baseline": baseline,
            "no_news_weight": no_news_weight,
            "no_market_weight": no_market_weight,
            "tight_risk": tight_risk,
            "loose_risk": loose_risk,
        }
