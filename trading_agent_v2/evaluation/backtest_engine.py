from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from trading_agent_v2.config import AppConfig, build_default_config
from trading_agent_v2.evaluation.metrics import EvaluationMetrics, compute_metrics
from trading_agent_v2.schemas import RawContext
from trading_agent_v2.tools.market_tools import MarketTools
from trading_agent_v2.tools.news_tools import NewsTools
from trading_agent_v2.tools.onchain_tools import OnchainTools
from trading_agent_v2.tools.social_tools import SocialTools
from trading_agent_v2.workflows.langgraph_workflow import run_cycle_with_langgraph


def _parse_time_to_ms(value: str | datetime | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        candidates = [text]
        if text.endswith("Z"):
            candidates.append(text[:-1] + "+00:00")
        dt = None
        for candidate in candidates:
            try:
                dt = datetime.fromisoformat(candidate)
                break
            except ValueError:
                continue
        if dt is None:
            raise ValueError(f"Unsupported datetime format: {value}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _timestamp_ms_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).replace(microsecond=0).isoformat()


def _normalize_symbols(symbols: list[str] | None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for symbol in symbols or []:
        item = str(symbol or "").strip().upper()
        if not item or "/" not in item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


@dataclass
class HistoricalCandle:
    timestamp_ms: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_ohlcv(self) -> list[float]:
        return [self.timestamp_ms, self.open, self.high, self.low, self.close, self.volume]


@dataclass
class BacktestResult:
    config_name: str
    symbols: list[str]
    cycles: int
    timeframe: str
    warmup_candles: int
    start_timestamp: str
    end_timestamp: str
    metrics: EvaluationMetrics
    cycle_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


class BacktestEngine:
    """
    Historical candle simulator for the production trading workflow.
    Each cycle consumes only candle data available up to that simulated timestamp.
    """

    def __init__(self, cycle_runner: Callable[..., dict[str, Any]] | None = None):
        self.cycle_runner = cycle_runner or run_cycle_with_langgraph

    def run(
        self,
        cycles: int = 200,
        symbols: list[str] | None = None,
        app_config: AppConfig | None = None,
        data_dir: str | Path | None = None,
        config_name: str = "historical",
        timeframe: str = "1h",
        warmup_candles: int = 50,
        candle_limit: int = 800,
        news_lookback_hours: int = 72,
        onchain_lookback_hours: int = 48,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> BacktestResult:
        if cycles <= 0:
            raise ValueError("cycles must be > 0")

        base_config = app_config or build_default_config()
        config = deepcopy(base_config)
        config.execution.mode = "paper"
        config.discovery.enabled = False
        config.langsmith.enabled = False

        target_symbols = _normalize_symbols(symbols or config.symbols)
        if not target_symbols:
            raise ValueError("At least one symbol is required for backtesting.")
        config.symbols = target_symbols

        resolved_data_dir = Path(data_dir or (config.data_dir / "backtests" / config_name))
        config.data_dir = resolved_data_dir
        self._reset_backtest_state(config)

        market_tools = MarketTools(
            exchange_id="okx",
            fallback_exchange_id="binanceus",
            timeframe=timeframe,
            ohlcv_limit=max(120, warmup_candles + 20),
        )
        news_tools = NewsTools()
        onchain_tools = OnchainTools()
        social_tools = SocialTools()

        since_ms = _parse_time_to_ms(start)
        until_ms = _parse_time_to_ms(end)
        history = self._load_symbol_histories(
            market_tools=market_tools,
            symbols=target_symbols,
            timeframe=timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
            candle_limit=max(candle_limit, warmup_candles + cycles + 5),
        )
        timeline = self._build_timeline(history=history, warmup_candles=warmup_candles, max_steps=cycles)
        if not timeline:
            raise ValueError("No aligned historical candles available for backtest window.")

        multimodal_context = self._prepare_multimodal_context(
            symbols=target_symbols,
            timeline=timeline,
            news_tools=news_tools,
            onchain_tools=onchain_tools,
            news_lookback_hours=news_lookback_hours,
            onchain_lookback_hours=onchain_lookback_hours,
        )

        results: list[dict[str, Any]] = []
        for timestamp_ms in timeline:
            for symbol in target_symbols:
                idx = history[symbol]["index"][timestamp_ms]
                window = history[symbol]["candles"][: idx + 1]
                result = self._run_historical_cycle(
                    symbol=symbol,
                    candles=window,
                    market_tools=market_tools,
                    news_tools=news_tools,
                    onchain_tools=onchain_tools,
                    social_tools=social_tools,
                    app_config=config,
                    timeframe=timeframe,
                    news_lookback_hours=news_lookback_hours,
                    onchain_lookback_hours=onchain_lookback_hours,
                    preloaded_context=multimodal_context.get(symbol, {}),
                )
                results.append(result)

        metrics = compute_metrics(results)
        return BacktestResult(
            config_name=config_name,
            symbols=target_symbols,
            cycles=len(timeline),
            timeframe=timeframe,
            warmup_candles=warmup_candles,
            start_timestamp=_timestamp_ms_to_iso(timeline[0]),
            end_timestamp=_timestamp_ms_to_iso(timeline[-1]),
            metrics=metrics,
            cycle_results=results,
        )

    def _reset_backtest_state(self, config: AppConfig) -> None:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        for path in [
            config.portfolio_file,
            config.episodic_memory_file,
            config.reflection_file,
            config.strategy_memory_file,
            config.pattern_memory_file,
            config.run_log_file,
        ]:
            if path.exists() and path.is_file():
                path.unlink()

    def _load_symbol_histories(
        self,
        market_tools: MarketTools,
        symbols: list[str],
        timeframe: str,
        since_ms: int | None,
        until_ms: int | None,
        candle_limit: int,
    ) -> dict[str, dict[str, Any]]:
        history: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            rows = market_tools.fetch_historical_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=candle_limit,
            )
            candles = [
                HistoricalCandle(
                    timestamp_ms=int(row[0]),
                    timestamp=_timestamp_ms_to_iso(int(row[0])),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
                for row in rows
                if len(row) >= 6
            ]
            if not candles:
                raise ValueError(f"No historical candles available for {symbol}.")
            history[symbol] = {
                "candles": candles,
                "index": {candle.timestamp_ms: idx for idx, candle in enumerate(candles)},
            }
        return history

    def _build_timeline(
        self,
        history: dict[str, dict[str, Any]],
        warmup_candles: int,
        max_steps: int,
    ) -> list[int]:
        timeline_sets = [
            set(int(candle.timestamp_ms) for candle in payload["candles"])
            for payload in history.values()
        ]
        if not timeline_sets:
            return []

        common = set.intersection(*timeline_sets)
        ordered = sorted(common)
        eligible: list[int] = []
        for timestamp_ms in ordered:
            if all(int(payload["index"][timestamp_ms]) >= warmup_candles - 1 for payload in history.values()):
                eligible.append(timestamp_ms)
        return eligible[:max_steps]

    def _prepare_multimodal_context(
        self,
        symbols: list[str],
        timeline: list[int],
        news_tools: NewsTools,
        onchain_tools: OnchainTools,
        news_lookback_hours: int,
        onchain_lookback_hours: int,
    ) -> dict[str, dict[str, Any]]:
        if not timeline:
            return {}

        start_timestamp = _timestamp_ms_to_iso(timeline[0])
        end_timestamp = _timestamp_ms_to_iso(timeline[-1])
        news_start_timestamp = _timestamp_ms_to_iso(
            timeline[0] - max(1, news_lookback_hours) * 3600 * 1000
        )
        onchain_start_ms = timeline[0] - max(24, onchain_lookback_hours) * 3600 * 1000

        output: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            news_tools.preload_historical_news_window(
                symbol=symbol,
                start=news_start_timestamp,
                end=end_timestamp,
                lookback_hours=news_lookback_hours,
            )
            output[symbol] = {
                "onchain_series": onchain_tools.preload_historical_onchain_series(
                    symbol=symbol,
                    start_ms=onchain_start_ms,
                    end_ms=timeline[-1],
                ),
            }
        return output

    def _run_historical_cycle(
        self,
        symbol: str,
        candles: list[HistoricalCandle],
        market_tools: MarketTools,
        news_tools: NewsTools,
        onchain_tools: OnchainTools,
        social_tools: SocialTools,
        app_config: AppConfig,
        timeframe: str,
        news_lookback_hours: int,
        onchain_lookback_hours: int,
        preloaded_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_candle = candles[-1]
        ohlcv_window = [candle.to_ohlcv() for candle in candles]
        market_data = market_tools.build_historical_market_snapshot(
            symbol=symbol,
            ohlcv_window=ohlcv_window,
            source=f"historical:{timeframe}",
        )
        news_data = news_tools.get_historical_news(
            symbol=symbol,
            as_of=last_candle.timestamp,
            lookback_hours=news_lookback_hours,
            limit=10,
        )
        news_summary = news_tools.summarize_sentiment(news_data)
        onchain_data = onchain_tools.get_historical_onchain_snapshot(
            symbol=symbol,
            as_of=last_candle.timestamp,
            lookback_hours=onchain_lookback_hours,
            preloaded_series=dict((preloaded_context or {}).get("onchain_series") or {}),
        )
        social_data = social_tools.get_historical_social_snapshot(
            symbol=symbol,
            as_of=last_candle.timestamp,
            news_data=news_data,
        )
        social_data.update(news_summary)
        raw_context = RawContext(
            symbol=symbol,
            timestamp=last_candle.timestamp,
            market_data=market_data,
            news_data=news_data,
            onchain_data=onchain_data,
            social_data=social_data,
        )
        result = self.cycle_runner(
            symbol=symbol,
            app_config=app_config,
            raw_context_override=raw_context,
            market_prices_override={symbol: float(last_candle.close)},
            simulation_timestamp=last_candle.timestamp,
        )
        result["backtest"] = {
            "timeframe": timeframe,
            "candle_timestamp": last_candle.timestamp,
            "candle_open": last_candle.open,
            "candle_high": last_candle.high,
            "candle_low": last_candle.low,
            "candle_close": last_candle.close,
            "candle_volume": last_candle.volume,
            "window_size": len(candles),
        }
        return result
