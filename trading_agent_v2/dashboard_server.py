from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from trading_agent_v2.config import build_default_config
from trading_agent_v2.portfolio.live_snapshot import load_live_portfolio_snapshot


DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
MAX_HISTORY_POINTS = 120
HISTORY_INTERVALS = {
    "raw": "Raw",
    "hour": "Hourly",
    "day": "Daily",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        return "MIXED"
    return "ACCOUNT"


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None

    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")

    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _bucket_history(rows: list[dict[str, Any]], interval: str) -> list[dict[str, Any]]:
    normalized = interval if interval in HISTORY_INTERVALS else "raw"
    if normalized == "raw":
        return rows

    buckets: dict[datetime, dict[str, Any]] = {}

    for row in rows:
        dt = _parse_timestamp(row.get("timestamp"))
        if dt is None:
            continue

        if normalized == "hour":
            bucket_dt = dt.replace(minute=0, second=0, microsecond=0)
        else:
            bucket_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        # Keep the last equity seen inside each bucket.
        buckets[bucket_dt] = {
            "timestamp": bucket_dt.isoformat(),
            "equity": _safe_float(row.get("equity")),
        }

    if not buckets:
        return []

    ordered_buckets = sorted(buckets.keys())
    bucket_step = timedelta(hours=1) if normalized == "hour" else timedelta(days=1)
    cursor = ordered_buckets[0]
    end = ordered_buckets[-1]
    expanded: list[dict[str, Any]] = []
    last_equity: float | None = None

    while cursor <= end:
        bucket_row = buckets.get(cursor)
        if bucket_row is not None:
            last_equity = _safe_float(bucket_row.get("equity"))
        if last_equity is not None:
            expanded.append(
                {
                    "timestamp": cursor.isoformat(),
                    "equity": last_equity,
                }
            )
        cursor += bucket_step

    return expanded


def _load_equity_history(
    log_file: Path,
    snapshot: dict[str, Any],
    interval: str = "raw",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                timestamp = str(item.get("logged_at", "")).strip()
                equity = item.get("total_equity")
                if not timestamp or equity is None:
                    continue

                rows.append(
                    {
                        "timestamp": timestamp,
                        "equity": _safe_float(equity),
                    }
                )

    snapshot_ts = str(snapshot.get("updated_at", "")).strip() or utc_now_iso()
    snapshot_equity = _safe_float(snapshot.get("total_equity"))
    rows.append(
        {
            "timestamp": snapshot_ts,
            "equity": snapshot_equity,
        }
    )

    rows.sort(
        key=lambda item: _parse_timestamp(item.get("timestamp"))
        or datetime.min.replace(tzinfo=timezone.utc)
    )

    if interval == "raw":
        return rows

    rows = _bucket_history(rows, interval)
    if not rows:
        return [
            {
                "timestamp": snapshot_ts,
                "equity": snapshot_equity,
            }
        ]

    return rows[-MAX_HISTORY_POINTS:]


def build_dashboard_payload(interval: str = "raw") -> dict[str, Any]:
    normalized_interval = interval if interval in HISTORY_INTERVALS else "raw"
    config = build_default_config()
    snapshot, sync_error, portfolio_source = load_live_portfolio_snapshot(config)

    positions_map = dict(snapshot.get("positions", {}) or {})
    symbols = list(positions_map.keys()) or list(config.symbols or [])

    positions: list[dict[str, Any]] = []
    total_unrealized_pnl = 0.0

    for symbol, pos in sorted(
        positions_map.items(),
        key=lambda item: _safe_float((item[1] or {}).get("market_value")),
        reverse=True,
    ):
        quantity = _safe_float(pos.get("quantity"))
        avg_entry_price = _safe_float(pos.get("avg_entry_price"))
        market_price = _safe_float(pos.get("market_price"))
        market_value = _safe_float(pos.get("market_value"))
        unrealized_pnl = _safe_float(pos.get("unrealized_pnl"))
        realized_pnl = _safe_float(pos.get("realized_pnl"))
        cost_basis = quantity * avg_entry_price
        return_pct = (unrealized_pnl / cost_basis) if cost_basis > 0 else 0.0

        total_unrealized_pnl += unrealized_pnl
        positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "avg_entry_price": avg_entry_price,
                "market_price": market_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "return_pct": return_pct,
                "updated_at": pos.get("updated_at", ""),
            }
        )

    cash = _safe_float(snapshot.get("cash"))
    total_equity = _safe_float(snapshot.get("total_equity"))
    realized_pnl = _safe_float(snapshot.get("realized_pnl"))
    total_pnl = realized_pnl + total_unrealized_pnl

    execution_mode = str(config.execution.mode).lower().strip()
    if execution_mode == "paper":
        base_equity = float(config.initial_cash)
        return_rate = (total_pnl / base_equity) if base_equity > 0 else 0.0
    else:
        return_rate = 0.0

    history = _load_equity_history(
        Path(config.run_log_file),
        snapshot,
        interval=normalized_interval,
    )

    return {
        "title": "TRADECLAW",
        "generated_at": utc_now_iso(),
        "execution_mode": str(config.execution.mode),
        "pnl_unit": _infer_pnl_unit(symbols),
        "history_interval": normalized_interval,
        "history_interval_label": HISTORY_INTERVALS[normalized_interval],
        "history_interval_options": [
            {"value": key, "label": value} for key, value in HISTORY_INTERVALS.items()
        ],
        "summary": {
            "cash": cash,
            "total_equity": total_equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": total_unrealized_pnl,
            "total_pnl": total_pnl,
            "return_rate": return_rate,
            "position_count": len(positions),
            "updated_at": snapshot.get("updated_at", ""),
        },
        "history_last_timestamp": history[-1]["timestamp"] if history else "",
        "portfolio_source": portfolio_source,
        "portfolio_sync_error": sync_error,
        "positions": positions,
        "history": history,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/dashboard":
            query = parse_qs(parsed.query)
            interval = str((query.get("interval") or ["raw"])[0]).strip().lower()
            payload = build_dashboard_payload(interval=interval)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TRADECLAW dashboard.")
    parser.add_argument("--host", default=os.getenv("TRADING_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TRADING_DASHBOARD_PORT", "8765")))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"TRADECLAW dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
