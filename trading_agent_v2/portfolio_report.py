from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from trading_agent_v2.config import build_default_config
from trading_agent_v2.portfolio.live_snapshot import load_live_portfolio_snapshot
from trading_agent_v2.portfolio.trade_logger import TradeLogger


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_money(value: float) -> str:
    return f"{value:,.4f}"


def _format_qty(value: float) -> str:
    return f"{value:,.8f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _truncate(text: str, limit: int = 90) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _format_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")

    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local_dt = dt.astimezone()
            offset = local_dt.utcoffset() or timezone.utc.utcoffset(None)
            total_minutes = int(offset.total_seconds() // 60)
            sign = "+" if total_minutes >= 0 else "-"
            hours, minutes = divmod(abs(total_minutes), 60)
            return f"{local_dt.strftime('%Y-%m-%d %H:%M:%S')} (UTC{sign}{hours:02d}:{minutes:02d})"
        except ValueError:
            continue

    return text


def _infer_pnl_unit(symbols: list[str]) -> str:
    quotes = set()
    for symbol in symbols:
        if not isinstance(symbol, str) or "/" not in symbol:
            continue
        quote = symbol.split("/", 1)[1].split(":", 1)[0].strip().upper()
        if quote:
            quotes.add(quote)
    if len(quotes) == 1:
        return next(iter(quotes))
    if len(quotes) > 1:
        return "mixed quote currencies"
    return "account currency"


def _load_recent_trades(log_file: str, limit: int, symbol: str | None) -> list[dict[str, Any]]:
    logger = TradeLogger(log_file)
    rows = logger.load_recent(limit=max(1, limit), symbol=symbol)
    return list(reversed(rows))


def _print_summary(
    snapshot: dict[str, Any],
    positions: dict[str, Any],
    *,
    initial_cash: float,
    execution_mode: str,
    pnl_unit: str,
    portfolio_source: str,
    sync_error: str | None,
) -> None:
    cash = _safe_float(snapshot.get("cash"))
    total_equity = _safe_float(snapshot.get("total_equity"))
    realized_pnl = _safe_float(snapshot.get("realized_pnl"))
    total_market_value = sum(_safe_float(pos.get("market_value")) for pos in positions.values())
    total_unrealized_pnl = sum(_safe_float(pos.get("unrealized_pnl")) for pos in positions.values())
    total_pnl = realized_pnl + total_unrealized_pnl
    estimated_starting_equity = total_equity - total_pnl
    return_rate = (total_pnl / estimated_starting_equity) if estimated_starting_equity > 0 else None
    pnl_vs_initial = (total_pnl / initial_cash) if initial_cash > 0 else None

    print("=" * 88)
    print("PORTFOLIO SUMMARY")
    print("=" * 88)
    print(f"Updated At        : {_format_timestamp(snapshot.get('updated_at', ''))}")
    print(f"Execution Mode    : {execution_mode}")
    print(f"Portfolio Source  : {portfolio_source}")
    if sync_error:
        print(f"Sync Status       : stale local snapshot ({_truncate(sync_error, limit=100)})")
    else:
        print("Sync Status       : live")
    print(f"PnL Unit          : {pnl_unit} (same unit as portfolio cash / quote currency)")
    print(f"Cash              : {_format_money(cash)}")
    print(f"Positions Value   : {_format_money(total_market_value)}")
    print(f"Total Equity      : {_format_money(total_equity)}")
    print(f"Realized PnL      : {_format_money(realized_pnl)}")
    print(f"Unrealized PnL    : {_format_money(total_unrealized_pnl)}")
    print(f"Total PnL         : {_format_money(total_pnl)}")
    if return_rate is not None:
        print(f"Return Rate       : {_format_pct(return_rate)}")
    else:
        print("Return Rate       : N/A")
    if execution_mode == "paper" and pnl_vs_initial is not None:
        print(f"PnL vs Initial    : {_format_pct(pnl_vs_initial)} (initial_cash={_format_money(initial_cash)})")
    else:
        print("PnL vs Initial    : N/A (only meaningful in paper mode)")
    if total_equity > 0:
        print(f"Invested Ratio    : {_format_pct(total_market_value / total_equity)}")
    print(f"Position Count    : {len(positions)}")


def _print_positions(positions: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("CURRENT POSITIONS")
    print("=" * 88)

    if not positions:
        print("No open positions.")
        return

    header = (
        f"{'Symbol':<14} {'Quantity':>14} {'Avg Entry':>14} {'Market Px':>14} "
        f"{'Mkt Value':>14} {'Unreal PnL':>14} {'Ret %':>9} {'Realized':>14}"
    )
    print(header)
    print("-" * len(header))

    for symbol, pos in positions.items():
        quantity = _safe_float(pos.get("quantity"))
        avg_entry = _safe_float(pos.get("avg_entry_price"))
        market_price = _safe_float(pos.get("market_price"))
        market_value = _safe_float(pos.get("market_value"))
        unrealized_pnl = _safe_float(pos.get("unrealized_pnl"))
        realized_pnl = _safe_float(pos.get("realized_pnl"))
        cost_basis = quantity * avg_entry
        unrealized_return = (unrealized_pnl / cost_basis) if cost_basis > 0 else 0.0

        print(
            f"{symbol:<14} "
            f"{_format_qty(quantity):>14} "
            f"{_format_money(avg_entry):>14} "
            f"{_format_money(market_price):>14} "
            f"{_format_money(market_value):>14} "
            f"{_format_money(unrealized_pnl):>14} "
            f"{_format_pct(unrealized_return):>9} "
            f"{_format_money(realized_pnl):>14}"
        )


def _print_recent_trades(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 88)
    print("RECENT TRADE SUMMARY")
    print("=" * 88)

    if not rows:
        print("No trade log entries found.")
        return

    for idx, row in enumerate(rows, start=1):
        print(
            f"[{idx}] {_format_timestamp(row.get('logged_at', ''))} | "
            f"{row.get('symbol', '')} | {row.get('action', '')}"
        )
        print(
            f"    status={row.get('execution_status', '')} "
            f"size_pct={_safe_float(row.get('size_pct')):.4f} "
            f"risk_score={_safe_float(row.get('risk_score')):.4f} "
            f"equity={_format_money(_safe_float(row.get('total_equity')))} "
            f"cash={_format_money(_safe_float(row.get('cash')))}"
        )
        print(f"    reason={_truncate(row.get('reason', ''))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show current portfolio holdings, PnL, and recent trade summaries.",
    )
    parser.add_argument(
        "--symbol",
        help="Filter positions and recent trade summaries by symbol, e.g. BTC/USDT.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many recent trade summaries to show. Default: 10.",
    )
    parser.add_argument(
        "--positions-only",
        action="store_true",
        help="Only print portfolio summary and current positions.",
    )
    args = parser.parse_args()

    config = build_default_config()
    snapshot, sync_error, portfolio_source = load_live_portfolio_snapshot(config)
    all_positions = dict(snapshot.get("positions", {}) or {})
    symbol_scope = list(all_positions.keys()) or list(config.symbols or [])
    if args.symbol:
        symbol_scope = [args.symbol]
    positions = (
        {args.symbol: all_positions[args.symbol]}
        if args.symbol and args.symbol in all_positions
        else ({} if args.symbol else all_positions)
    )

    _print_summary(
        snapshot=snapshot,
        positions=positions if args.symbol else all_positions,
        initial_cash=float(config.initial_cash),
        execution_mode=str(config.execution.mode),
        pnl_unit=_infer_pnl_unit(symbol_scope),
        portfolio_source=portfolio_source,
        sync_error=sync_error,
    )
    _print_positions(positions=positions)

    if not args.positions_only:
        recent_trades = _load_recent_trades(
            log_file=str(config.run_log_file),
            limit=args.limit,
            symbol=args.symbol,
        )
        _print_recent_trades(recent_trades)


if __name__ == "__main__":
    main()
