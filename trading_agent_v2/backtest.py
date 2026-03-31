from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from trading_agent_v2.config import build_default_config
from trading_agent_v2.evaluation.backtest_engine import BacktestEngine
from trading_agent_v2.evaluation.metrics import summarize_metrics
from trading_agent_v2.portfolio.live_snapshot import load_live_portfolio_snapshot
from trading_agent_v2.workflows.langgraph_workflow import run_cycle_with_langgraph


def _parse_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip().upper() for item in str(value).split(",")]
    symbols = [item for item in items if item and "/" in item]
    return symbols or None


def _resolve_backtest_symbols(
    explicit_symbols: list[str] | None,
) -> tuple[object, list[str], str, str | None, str]:
    config = build_default_config()
    if explicit_symbols:
        config.symbols = explicit_symbols
        return config, explicit_symbols, "cli", None, "manual override"

    snapshot, sync_error, portfolio_source = load_live_portfolio_snapshot(config)
    position_symbols = [
        str(symbol).upper()
        for symbol in list((snapshot.get("positions") or {}).keys())
        if isinstance(symbol, str) and "/" in symbol
    ]
    if position_symbols:
        config.symbols = position_symbols
        symbol_source = "current account holdings" if not sync_error else "latest synced holdings"
        return config, position_symbols, portfolio_source, sync_error, symbol_source

    fallback_symbols = [str(symbol).upper() for symbol in (config.symbols or []) if "/" in str(symbol)]
    return config, fallback_symbols, portfolio_source, sync_error, "config fallback"


def _print_backtest_summary(
    result: object,
    *,
    symbol_source: str,
    portfolio_source: str,
    sync_error: str | None,
) -> None:
    metrics = getattr(result, "metrics", None)
    summary_text = summarize_metrics(metrics) if metrics is not None else "metrics unavailable"
    print("\n" + "=" * 72)
    print("TRADECLAW BACKTEST")
    print("=" * 72)
    print(f"Symbols: {', '.join(getattr(result, 'symbols', []) or [])}")
    print(f"Symbol Source: {symbol_source}")
    print(f"Portfolio Source: {portfolio_source}")
    if sync_error:
        print(f"Portfolio Sync: stale snapshot ({sync_error})")
    else:
        print("Portfolio Sync: live")
    print(
        f"Window: {getattr(result, 'start_timestamp', 'n/a')} -> "
        f"{getattr(result, 'end_timestamp', 'n/a')}"
    )
    print(
        f"Timeframe: {getattr(result, 'timeframe', 'n/a')} "
        f"| Warmup: {getattr(result, 'warmup_candles', 'n/a')} "
        f"| Steps: {getattr(result, 'cycles', 'n/a')}"
    )
    print(f"Metrics: {summary_text}")
    print("\nBacktest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TRADECLAW historical backtests.")
    parser.add_argument("--symbols", help="Optional comma-separated symbols, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--cycles", type=int, default=200, help="Max historical steps to simulate.")
    parser.add_argument("--timeframe", default="1h", help="Historical candle timeframe, e.g. 1h or 1d.")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup candles before the first simulated step.")
    parser.add_argument("--candle-limit", type=int, default=800, help="How many candles to fetch per symbol.")
    parser.add_argument("--start", help="Historical start time in ISO format.")
    parser.add_argument("--end", help="Historical end time in ISO format.")
    parser.add_argument("--data-dir", help="Optional isolated data directory for backtest state.")
    parser.add_argument("--news-lookback-hours", type=int, default=72, help="Historical news lookback window.")
    parser.add_argument("--onchain-lookback-hours", type=int, default=48, help="Historical onchain lookback window.")
    args = parser.parse_args()

    explicit_symbols = _parse_symbols(args.symbols)
    config, symbols, portfolio_source, sync_error, symbol_source = _resolve_backtest_symbols(explicit_symbols)
    engine = BacktestEngine(cycle_runner=run_cycle_with_langgraph)
    result = engine.run(
        cycles=max(1, args.cycles),
        symbols=symbols,
        app_config=config,
        data_dir=args.data_dir,
        config_name="backtest_cli",
        timeframe=str(args.timeframe or "1h").strip() or "1h",
        warmup_candles=max(20, args.warmup),
        candle_limit=max(100, args.candle_limit),
        news_lookback_hours=max(1, args.news_lookback_hours),
        onchain_lookback_hours=max(24, args.onchain_lookback_hours),
        start=args.start,
        end=args.end,
    )
    _print_backtest_summary(
        result,
        symbol_source=symbol_source,
        portfolio_source=portfolio_source,
        sync_error=sync_error,
    )


if __name__ == "__main__":
    main()
