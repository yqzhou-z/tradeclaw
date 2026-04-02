from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


@dataclass
class TraderAgentConfig:
    buy_threshold: float = 0.06
    sell_threshold: float = -0.06
    min_confidence: float = 0.45
    default_buy_size_pct: float = 0.40
    default_sell_size_pct: float = 0.50
    min_directional_size_pct: float = 0.12
    default_stop_loss_pct: float = 0.03
    default_take_profit_pct: float = 0.06
    analyst_weights: dict[str, float] = field(
        default_factory=lambda: {
            "market_analyst": 1.2,
            "news_analyst": 0.8,
        }
    )


@dataclass
class PlannerConfig:
    action_threshold: float = 0.04
    min_trade_confidence: float = 0.30
    min_directional_size_pct: float = 0.15
    max_directional_size_pct: float = 1.00
    aggressive_size_multiplier: float = 1.80


@dataclass
class RiskManagerConfig:
    max_total_invested_pct: float = 0.90
    max_single_position_pct: float = 0.45
    min_proposal_confidence: float = 0.25
    max_conflicting_factors: int = 6
    max_loss_streak: int = 4
    volatility_size_cut_ratio: float = 0.70
    high_risk_score_threshold: float = 0.85
    medium_risk_score_threshold: float = 0.60
    elevated_risk_size_multiplier: float = 0.85
    conflict_size_cut_ratio: float = 0.90
    volatility_stop_loss_tighten_ratio: float = 0.92


@dataclass
class ValidationConfig:
    min_size_pct: float = 0.01
    max_size_pct: float = 1.00
    allow_short: bool = False


@dataclass
class ExecutionConfig:
    mode: str = "okx"  # paper | okx
    trading_fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    enforce_min_size: bool = False
    okx_api_key: str = ""
    okx_secret: str = ""
    okx_passphrase: str = ""
    okx_use_sandbox: bool = True
    okx_timeout_ms: int = 10000
    okx_enable_rate_limit: bool = True


@dataclass
class MemoryConfig:
    recent_episodes: int = 30
    strategy_lookback_episodes: int = 60
    reflection_lookback_episodes: int = 20


@dataclass
class LLMConfig:
    enabled: bool = True
    model: str = "gpt-5.4"
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_sec: int = 30


@dataclass
class MarketDiscoveryConfig:
    enabled: bool = True
    quote_assets: list[str] = field(default_factory=lambda: ["USDT"])
    scout_limit: int = 80
    llm_candidate_pool_size: int = 24
    shortlist_size: int = 6
    force_include_current_positions: bool = True


@dataclass
class LangSmithConfig:
    enabled: bool = False
    project: str = "trading-agent-v2"
    endpoint: str = "https://api.smith.langchain.com"
    api_key: str = ""
    tags: list[str] = field(default_factory=lambda: ["trading-agent-v2"])


@dataclass
class AppConfig:
    base_dir: Path
    data_dir: Path

    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    initial_cash: float = 10000.0

    planner: PlannerConfig = field(default_factory=PlannerConfig)
    trader: TraderAgentConfig = field(default_factory=TraderAgentConfig)
    risk: RiskManagerConfig = field(default_factory=RiskManagerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    discovery: MarketDiscoveryConfig = field(default_factory=MarketDiscoveryConfig)
    langsmith: LangSmithConfig = field(default_factory=LangSmithConfig)

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
    def daily_review_file(self) -> Path:
        return self.data_dir / "daily_reviews.jsonl"

    @property
    def run_log_file(self) -> Path:
        return self.data_dir / "run_log.jsonl"


def bootstrap_langsmith_env(load_env: bool = True) -> None:
    if load_env:
        _load_project_env()

    enabled = _env_bool(
        "TRADING_LANGSMITH_ENABLED",
        _env_bool("LANGSMITH_TRACING", False),
    )
    project = (
        os.getenv("TRADING_LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "trading-agent-v2")).strip()
        or "trading-agent-v2"
    )
    endpoint = (
        os.getenv(
            "TRADING_LANGSMITH_ENDPOINT",
            os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        ).strip()
        or "https://api.smith.langchain.com"
    )
    api_key = os.getenv("TRADING_LANGSMITH_API_KEY", os.getenv("LANGSMITH_API_KEY", "")).strip()

    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    # Keep both names in sync for compatibility with mixed LangChain/LangSmith stacks.
    os.environ["LANGCHAIN_TRACING"] = "true" if enabled else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key

    clear_langsmith_env_cache()


def build_default_config(base_dir: Path | None = None) -> AppConfig:
    resolved_base_dir = base_dir or Path(__file__).resolve().parent
    _load_project_env(resolved_base_dir)
    data_dir = resolved_base_dir / "data"
    llm_enabled = _env_bool("TRADING_LLM_ENABLED", True)
    llm_model = os.getenv("TRADING_LLM_MODEL", "gpt-5.4").strip() or "gpt-5.4"
    llm_temperature = _env_float("TRADING_LLM_TEMPERATURE", 0.2)
    llm_max_tokens = _env_int("TRADING_LLM_MAX_TOKENS", 1200)
    llm_timeout_sec = _env_int("TRADING_LLM_TIMEOUT_SEC", 30)
    configured_symbols = _env_list("TRADING_SYMBOLS", default=["BTC/USDT"])
    discovery_enabled = _env_bool("TRADING_SYMBOL_DISCOVERY_ENABLED", True)
    discovery_quote_assets = _env_list("TRADING_DISCOVERY_QUOTES", default=["USDT"])
    discovery_scout_limit = _env_int("TRADING_DISCOVERY_SCOUT_LIMIT", 80)
    discovery_llm_candidate_pool_size = _env_int("TRADING_DISCOVERY_LLM_CANDIDATE_POOL_SIZE", 24)
    discovery_shortlist_size = _env_int("TRADING_DISCOVERY_SHORTLIST_SIZE", 6)
    discovery_force_include_positions = _env_bool("TRADING_DISCOVERY_FORCE_INCLUDE_POSITIONS", True)
    planner_action_threshold = _env_float("TRADING_PLANNER_ACTION_THRESHOLD", 0.04)
    planner_min_trade_confidence = _env_float("TRADING_PLANNER_MIN_TRADE_CONFIDENCE", 0.30)
    planner_min_directional_size_pct = _env_float("TRADING_PLANNER_MIN_DIRECTIONAL_SIZE_PCT", 0.15)
    planner_max_directional_size_pct = _env_float("TRADING_PLANNER_MAX_DIRECTIONAL_SIZE_PCT", 1.00)
    planner_aggressive_size_multiplier = _env_float("TRADING_PLANNER_AGGRESSIVE_SIZE_MULTIPLIER", 1.80)
    execution_mode = os.getenv("TRADING_EXECUTION_MODE", "okx").strip().lower()
    okx_api_key = os.getenv("OKX_API_KEY", "").strip()
    okx_secret = os.getenv("OKX_SECRET_KEY", os.getenv("OKX_SECRET", "")).strip()
    okx_passphrase = os.getenv("OKX_PASSPHRASE", "").strip()
    okx_use_sandbox = _env_bool("TRADING_OKX_USE_SANDBOX", True)
    okx_timeout_ms = _env_int("TRADING_OKX_TIMEOUT_MS", 10000)
    okx_enable_rate_limit = _env_bool("TRADING_OKX_ENABLE_RATE_LIMIT", True)
    langsmith_enabled = _env_bool(
        "TRADING_LANGSMITH_ENABLED",
        _env_bool("LANGSMITH_TRACING", False),
    )
    langsmith_project = (
        os.getenv("TRADING_LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "trading-agent-v2")).strip()
        or "trading-agent-v2"
    )
    langsmith_endpoint = (
        os.getenv(
            "TRADING_LANGSMITH_ENDPOINT",
            os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        ).strip()
        or "https://api.smith.langchain.com"
    )
    langsmith_api_key = os.getenv("TRADING_LANGSMITH_API_KEY", os.getenv("LANGSMITH_API_KEY", "")).strip()
    langsmith_tags = _env_list("TRADING_LANGSMITH_TAGS", default=["trading-agent-v2"])

    return AppConfig(
        base_dir=resolved_base_dir,
        data_dir=data_dir,
        symbols=configured_symbols or ["BTC/USDT"],
        planner=PlannerConfig(
            action_threshold=max(0.0, min(1.0, planner_action_threshold)),
            min_trade_confidence=max(0.0, min(1.0, planner_min_trade_confidence)),
            min_directional_size_pct=max(0.0, min(1.0, planner_min_directional_size_pct)),
            max_directional_size_pct=max(0.0, min(1.0, planner_max_directional_size_pct)),
            aggressive_size_multiplier=max(1.0, planner_aggressive_size_multiplier),
        ),
        execution=ExecutionConfig(
            mode=execution_mode if execution_mode in {"paper", "okx"} else "paper",
            okx_api_key=okx_api_key,
            okx_secret=okx_secret,
            okx_passphrase=okx_passphrase,
            okx_use_sandbox=okx_use_sandbox,
            okx_timeout_ms=max(1000, okx_timeout_ms),
            okx_enable_rate_limit=okx_enable_rate_limit,
        ),
        llm=LLMConfig(
            enabled=llm_enabled,
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            timeout_sec=llm_timeout_sec,
        ),
        discovery=MarketDiscoveryConfig(
            enabled=discovery_enabled,
            quote_assets=discovery_quote_assets or ["USDT"],
            scout_limit=max(10, discovery_scout_limit),
            llm_candidate_pool_size=max(6, discovery_llm_candidate_pool_size),
            shortlist_size=max(1, discovery_shortlist_size),
            force_include_current_positions=discovery_force_include_positions,
        ),
        langsmith=LangSmithConfig(
            enabled=langsmith_enabled,
            project=langsmith_project,
            endpoint=langsmith_endpoint,
            api_key=langsmith_api_key,
            tags=langsmith_tags,
        ),
    )


def clear_langsmith_env_cache() -> None:
    try:
        from langsmith import utils as langsmith_utils
    except Exception:  # pragma: no cover
        return

    cache_clear = getattr(getattr(langsmith_utils, "get_env_var", None), "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


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


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return list(default or [])
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _load_project_env(base_dir: Path | None = None) -> None:
    if load_dotenv is None:
        return

    resolved_base_dir = base_dir or Path(__file__).resolve().parent
    candidates = [
        resolved_base_dir / ".env",
        resolved_base_dir.parent / ".env",
    ]

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            load_dotenv(path, override=False)
