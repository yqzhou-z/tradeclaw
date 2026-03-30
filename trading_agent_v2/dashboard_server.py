from __future__ import annotations

import argparse
import json
import os
import threading
import time
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
HISTORY_INTERVALS = {
    "hour": "Hourly",
    "day": "Daily",
}
HISTORY_RANGES_BY_INTERVAL = {
    "hour": {
        "24h": "24H",
        "7d": "7D",
        "30d": "30D",
        "all": "All",
    },
    "day": {
        "7d": "7D",
        "30d": "30D",
        "all": "All",
    },
}
SNAPSHOT_CACHE_TTL_SECONDS = 15.0
_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: Any = None
_SNAPSHOT_CACHE_LOCK = threading.Lock()
_SNAPSHOT_CACHE: dict[str, Any] = {
    "key": None,
    "expires_at": 0.0,
    "snapshot": None,
    "sync_error": None,
    "portfolio_source": "local_snapshot",
}
_RUN_LOG_CACHE_LOCK = threading.Lock()
_RUN_LOG_CACHE: dict[str, Any] = {
    "key": None,
    "rows": [],
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
    normalized = interval if interval in HISTORY_INTERVALS else "hour"

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


def _normalize_history_interval(interval: str) -> str:
    candidate = str(interval or "").strip().lower()
    return candidate if candidate in HISTORY_INTERVALS else "hour"


def _normalize_history_range(interval: str, range_key: str) -> str:
    normalized_interval = _normalize_history_interval(interval)
    available = HISTORY_RANGES_BY_INTERVAL[normalized_interval]
    candidate = str(range_key or "").strip().lower()
    return candidate if candidate in available else next(iter(available.keys()))


def _range_window_start(interval: str, range_key: str, end: datetime) -> datetime | None:
    normalized_interval = _normalize_history_interval(interval)
    normalized_range = _normalize_history_range(normalized_interval, range_key)
    if normalized_range == "all":
        return None

    if normalized_interval == "hour":
        steps = {
            "24h": 24,
            "7d": 24 * 7,
            "30d": 24 * 30,
        }.get(normalized_range, 24 * 7)
        return end - timedelta(hours=max(steps - 1, 0))

    steps = {
        "7d": 7,
        "30d": 30,
    }.get(normalized_range, 7)
    return end - timedelta(days=max(steps - 1, 0))


def _get_dashboard_config() -> Any:
    global _CONFIG_CACHE
    with _CONFIG_LOCK:
        if _CONFIG_CACHE is None:
            _CONFIG_CACHE = build_default_config()
        return _CONFIG_CACHE


def _portfolio_cache_key(config: Any) -> tuple[Any, ...]:
    execution_mode = str(config.execution.mode or "").lower().strip()
    return (
        str(config.portfolio_file),
        execution_mode,
        bool(getattr(config.execution, "okx_use_sandbox", False)),
        tuple(str(symbol).upper() for symbol in (config.symbols or [])),
    )


def _load_cached_portfolio_snapshot(
    config: Any,
    force_refresh: bool = False,
) -> tuple[dict[str, Any], str | None, str]:
    cache_key = _portfolio_cache_key(config)
    now = time.monotonic()

    with _SNAPSHOT_CACHE_LOCK:
        is_fresh = (
            not force_refresh
            and _SNAPSHOT_CACHE.get("key") == cache_key
            and now < float(_SNAPSHOT_CACHE.get("expires_at", 0.0))
            and isinstance(_SNAPSHOT_CACHE.get("snapshot"), dict)
        )
        if is_fresh:
            return (
                dict(_SNAPSHOT_CACHE.get("snapshot") or {}),
                _SNAPSHOT_CACHE.get("sync_error"),
                str(_SNAPSHOT_CACHE.get("portfolio_source") or "local_snapshot"),
            )

    snapshot, sync_error, portfolio_source = load_live_portfolio_snapshot(config)

    with _SNAPSHOT_CACHE_LOCK:
        _SNAPSHOT_CACHE.update(
            {
                "key": cache_key,
                "expires_at": time.monotonic() + SNAPSHOT_CACHE_TTL_SECONDS,
                "snapshot": dict(snapshot or {}),
                "sync_error": sync_error,
                "portfolio_source": portfolio_source,
            }
        )

    return snapshot, sync_error, portfolio_source


def _load_run_log_rows(log_file: Path) -> list[dict[str, Any]]:
    try:
        stat = log_file.stat()
        cache_key = (str(log_file), stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        cache_key = (str(log_file), None, None)

    with _RUN_LOG_CACHE_LOCK:
        if _RUN_LOG_CACHE.get("key") == cache_key:
            return list(_RUN_LOG_CACHE.get("rows") or [])

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

    with _RUN_LOG_CACHE_LOCK:
        _RUN_LOG_CACHE.update(
            {
                "key": cache_key,
                "rows": list(rows),
            }
        )

    return rows


def _load_equity_history(
    log_file: Path,
    snapshot: dict[str, Any],
    interval: str = "hour",
    range_key: str = "7d",
) -> list[dict[str, Any]]:
    rows = _load_run_log_rows(log_file)

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

    normalized_interval = _normalize_history_interval(interval)
    normalized_range = _normalize_history_range(normalized_interval, range_key)
    rows = _bucket_history(rows, normalized_interval)
    if not rows:
        return [
            {
                "timestamp": snapshot_ts,
                "equity": snapshot_equity,
            }
        ]

    if normalized_range == "all":
        return rows

    end_dt = _parse_timestamp(rows[-1].get("timestamp")) or _parse_timestamp(snapshot_ts)
    if end_dt is None:
        return rows

    start_dt = _range_window_start(normalized_interval, normalized_range, end_dt)
    if start_dt is None:
        return rows

    return [
        row for row in rows
        if (_parse_timestamp(row.get("timestamp")) or end_dt) >= start_dt
    ]


def _compute_return_rate(total_equity: float, total_pnl: float) -> float | None:
    estimated_starting_equity = total_equity - total_pnl
    if estimated_starting_equity <= 0:
        return None
    return total_pnl / estimated_starting_equity


def build_dashboard_payload(
    interval: str = "hour",
    range_key: str = "7d",
    force_refresh: bool = False,
) -> dict[str, Any]:
    normalized_interval = _normalize_history_interval(interval)
    normalized_range = _normalize_history_range(normalized_interval, range_key)
    config = _get_dashboard_config()
    snapshot, sync_error, portfolio_source = _load_cached_portfolio_snapshot(
        config=config,
        force_refresh=force_refresh,
    )

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
    return_rate = _compute_return_rate(total_equity=total_equity, total_pnl=total_pnl)

    history = _load_equity_history(
        Path(config.run_log_file),
        snapshot,
        interval=normalized_interval,
        range_key=normalized_range,
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
        "history_range": normalized_range,
        "history_range_label": HISTORY_RANGES_BY_INTERVAL[normalized_interval][normalized_range],
        "history_range_options": [
            {"value": key, "label": value}
            for key, value in HISTORY_RANGES_BY_INTERVAL[normalized_interval].items()
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
            interval = str((query.get("interval") or ["hour"])[0]).strip().lower()
            range_key = str((query.get("range") or ["7d"])[0]).strip().lower()
            refresh = str((query.get("refresh") or ["0"])[0]).strip().lower() in {"1", "true", "yes"}
            payload = build_dashboard_payload(
                interval=interval,
                range_key=range_key,
                force_refresh=refresh,
            )
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
