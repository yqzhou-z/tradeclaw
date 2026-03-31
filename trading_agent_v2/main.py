from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from trading_agent_v2.config import AppConfig, bootstrap_langsmith_env, build_default_config
from trading_agent_v2.evaluation.backtest_engine import BacktestEngine
from trading_agent_v2.evaluation.metrics import summarize_metrics

bootstrap_langsmith_env()

from trading_agent_v2.workflows.langgraph_workflow import (
    resolve_batch_symbols,
    run_batch_with_langgraph,
    run_cycle_with_langgraph,
)


def run_cycle(symbol: str = "BTC/USDT", app_config: AppConfig | None = None) -> dict:
    return run_cycle_with_langgraph(symbol=symbol, app_config=app_config)


def run_batch(symbols: list[str] | None = None, app_config: AppConfig | None = None) -> list[dict]:
    return run_batch_with_langgraph(symbols=symbols, app_config=app_config)


def _parse_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip().upper() for item in str(value).split(",")]
    symbols = [item for item in items if item and "/" in item]
    return symbols or None


def _status_label(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "on" if value else "off"


def _format_pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_number(value: object, decimals: int = 6) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def _shorten(text: object, max_len: int = 110) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."


def _summarize_result(result: dict) -> list[str]:
    symbol = str(result.get("symbol", "UNKNOWN"))
    final_decision = result.get("final_decision") or {}
    execution_result = result.get("execution_result") or {}
    validation = result.get("validation") or {}

    action = str(final_decision.get("action", execution_result.get("action", "hold")) or "hold").upper()
    size_pct = final_decision.get("size_pct")
    decision_summary = f"{symbol} | decision={action}"
    if action in {"BUY", "SELL"}:
        decision_summary += f" | size={_format_pct(size_pct)}"

    status = str(execution_result.get("status", "unknown") or "unknown").upper()
    exec_summary = f"status={status}"

    filled_qty = execution_result.get("filled_qty")
    filled_price = execution_result.get("filled_price")
    if filled_qty not in (None, "") and filled_price not in (None, ""):
        exec_summary += (
            f" | filled={_format_number(filled_qty, decimals=8)}"
            f" @ {_format_number(filled_price, decimals=6)}"
        )

    errors = validation.get("errors") or []
    if errors:
        exec_summary += f" | validation={_shorten(errors[0], max_len=90)}"
    else:
        message = str(execution_result.get("message", "") or final_decision.get("reason", "")).strip()
        if message:
            exec_summary += f" | {_shorten(message, max_len=90)}"

    return [decision_summary, exec_summary]


def _print_backtest_summary(result: object) -> None:
    metrics = getattr(result, "metrics", None)
    summary_text = summarize_metrics(metrics) if metrics is not None else "metrics unavailable"
    print("\n" + "=" * 72)
    print("TRADECLAW BACKTEST")
    print("=" * 72)
    print(f"Symbols: {', '.join(getattr(result, 'symbols', []) or [])}")
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
    parser = argparse.ArgumentParser(description="Run TRADECLAW live mode or historical backtest.")
    parser.add_argument("--symbols", help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--backtest", action="store_true", help="Run historical candle backtest instead of live batch.")
    parser.add_argument("--backtest-cycles", type=int, default=200, help="Max historical steps to simulate.")
    parser.add_argument("--backtest-timeframe", default="1h", help="Historical candle timeframe, e.g. 1h or 1d.")
    parser.add_argument("--backtest-warmup", type=int, default=50, help="Warmup candles before the first simulated step.")
    parser.add_argument("--backtest-candle-limit", type=int, default=800, help="How many candles to fetch per symbol.")
    parser.add_argument("--backtest-start", help="Historical start time in ISO format.")
    parser.add_argument("--backtest-end", help="Historical end time in ISO format.")
    parser.add_argument("--backtest-data-dir", help="Optional isolated data directory for backtest state.")
    args = parser.parse_args()

    config = build_default_config()
    explicit_symbols = _parse_symbols(args.symbols)
    if explicit_symbols:
        config.symbols = explicit_symbols

    if args.backtest:
        engine = BacktestEngine(cycle_runner=run_cycle_with_langgraph)
        result = engine.run(
            cycles=max(1, args.backtest_cycles),
            symbols=explicit_symbols or config.symbols,
            app_config=config,
            data_dir=args.backtest_data_dir,
            config_name="main_cli",
            timeframe=str(args.backtest_timeframe or "1h").strip() or "1h",
            warmup_candles=max(20, args.backtest_warmup),
            candle_limit=max(100, args.backtest_candle_limit),
            start=args.backtest_start,
            end=args.backtest_end,
        )
        _print_backtest_summary(result)
        return

    target_symbols, selection = resolve_batch_symbols(symbols=explicit_symbols, app_config=config)
    results = run_batch(symbols=target_symbols, app_config=config)
    first_result = results[0] if results else {}
    execution_meta = first_result.get("execution") or {}
    llm_meta = first_result.get("llm") or {}
    observability = ((first_result.get("observability") or {}).get("langsmith") or {})

    print("\n" + "=" * 72)
    print("TRADEZ RUN SUMMARY")
    print("=" * 72)
    print(
        "Mode: "
        f"{str(execution_meta.get('mode', config.execution.mode)).upper()} "
        f"| OKX sandbox={_status_label(execution_meta.get('okx_sandbox'))} "
        f"| LLM={llm_meta.get('model', config.llm.model)} "
        f"| LangSmith={_status_label(observability.get('ready'))}"
    )
    print(f"Symbols: {', '.join(target_symbols)}")
    if selection is not None:
        print(f"Discovery: scanned={selection.candidate_count} | llm_selector={_status_label(selection.llm_used)}")

    print("\nActions:")
    if not results:
        print("- No symbols were processed.")
    for result in results:
        summary_lines = _summarize_result(result)
        for line in summary_lines:
            print(f"- {line}")

    print("\nRun complete.")


if __name__ == "__main__":
    main()
