from __future__ import annotations

from typing import Any

from trading_agent_v2.config import AppConfig
from trading_agent_v2.execution.okx_executor import OkxExecutor
from trading_agent_v2.portfolio.portfolio_manager import PortfolioManager


def load_live_portfolio_snapshot(config: AppConfig) -> tuple[dict[str, Any], str | None, str]:
    manager = PortfolioManager(str(config.portfolio_file))
    manager.ensure_portfolio_exists(initial_cash=config.initial_cash)
    portfolio = manager.load_portfolio()

    sync_error: str | None = None
    source = "local_snapshot"
    execution_mode = str(config.execution.mode or "").lower().strip()

    if execution_mode == "okx":
        try:
            executor = OkxExecutor(
                api_key=config.execution.okx_api_key,
                secret=config.execution.okx_secret,
                passphrase=config.execution.okx_passphrase,
                use_sandbox=config.execution.okx_use_sandbox,
                timeout_ms=config.execution.okx_timeout_ms,
                enable_rate_limit=config.execution.okx_enable_rate_limit,
            )
            portfolio = executor.sync_portfolio_state(
                portfolio=portfolio,
                symbols=list((portfolio.positions or {}).keys()) + list(config.symbols or []),
            )
            manager.save_portfolio(portfolio)
            source = "okx_live"
        except Exception as exc:
            sync_error = str(exc)

    snapshot = manager.get_portfolio_snapshot(portfolio)
    return snapshot, sync_error, source
