from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TraderAgentConfig:
    buy_threshold: float = 0.20
    sell_threshold: float = -0.20
    min_confidence: float = 0.55
    default_buy_size_pct: float = 0.10
    default_sell_size_pct: float = 0.50
    default_stop_loss_pct: float = 0.03
    default_take_profit_pct: float = 0.06
    analyst_weights: dict[str, float] = field(
        default_factory=lambda: {
            "market_analyst": 1.2,
            "news_analyst": 0.8,
        }
    )


@dataclass
class RiskManagerConfig:
    max_total_invested_pct: float = 0.80
    max_single_position_pct: float = 0.30
    min_proposal_confidence: float = 0.55
    max_conflicting_factors: int = 3
    max_loss_streak: int = 3
    volatility_size_cut_ratio: float = 0.5
    high_risk_score_threshold: float = 0.75
    medium_risk_score_threshold: float = 0.45


@dataclass
class ValidationConfig:
    min_size_pct: float = 0.01
    max_size_pct: float = 0.30
    allow_short: bool = False


@dataclass
class ExecutionConfig:
    trading_fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    enforce_min_size: bool = False


@dataclass
class MemoryConfig:
    recent_episodes: int = 30
    strategy_lookback_episodes: int = 60
    reflection_lookback_episodes: int = 20


@dataclass
class LLMConfig:
    enabled: bool = True
    model: str = "o3"
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_sec: int = 30


@dataclass
class AppConfig:
    base_dir: Path
    data_dir: Path

    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    initial_cash: float = 10000.0

    trader: TraderAgentConfig = field(default_factory=TraderAgentConfig)
    risk: RiskManagerConfig = field(default_factory=RiskManagerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @property
    def portfolio_file(self) -> Path:
        return self.data_dir / "paper_portfolio.json"

    @property
    def episodic_memory_file(self) -> Path:
        return self.data_dir / "trade_history.jsonl"

    @property
    def reflection_file(self) -> Path:
        return self.data_dir / "reflections.jsonl"

    @property
    def strategy_memory_file(self) -> Path:
        return self.data_dir / "strategy_memory.json"

    @property
    def pattern_memory_file(self) -> Path:
        return self.data_dir / "pattern_memory.json"

    @property
    def run_log_file(self) -> Path:
        return self.data_dir / "run_log.jsonl"


def build_default_config(base_dir: Path | None = None) -> AppConfig:
    resolved_base_dir = base_dir or Path(__file__).resolve().parent
    data_dir = resolved_base_dir / "data"
    llm_enabled = _env_bool("TRADING_LLM_ENABLED", True)
    llm_model = os.getenv("TRADING_LLM_MODEL", "o3")
    llm_temperature = _env_float("TRADING_LLM_TEMPERATURE", 0.2)
    llm_max_tokens = _env_int("TRADING_LLM_MAX_TOKENS", 1200)
    llm_timeout_sec = _env_int("TRADING_LLM_TIMEOUT_SEC", 30)

    return AppConfig(
        base_dir=resolved_base_dir,
        data_dir=data_dir,
        llm=LLMConfig(
            enabled=llm_enabled,
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            timeout_sec=llm_timeout_sec,
        ),
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
