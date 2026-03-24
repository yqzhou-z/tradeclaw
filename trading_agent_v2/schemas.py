from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# =========================================================
# Base mixin
# =========================================================

@dataclass
class SerializableDataclass:
    """Simple helper mixin for JSON-friendly conversion."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# Raw data collection layer
# =========================================================

@dataclass
class RawContext(SerializableDataclass):
    symbol: str
    timestamp: str

    market_data: Dict[str, Any] = field(default_factory=dict)
    news_data: List[Dict[str, Any]] = field(default_factory=list)
    onchain_data: Dict[str, Any] = field(default_factory=dict)
    social_data: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Analyst output
# =========================================================

@dataclass
class AnalystView(SerializableDataclass):
    analyst_name: str
    symbol: str
    timestamp: str

    bias: str  # bullish / bearish / neutral
    confidence: float  # 0.0 ~ 1.0
    summary: str

    supporting_signals: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Trader proposal
# =========================================================

@dataclass
class TradeProposal(SerializableDataclass):
    symbol: str
    timestamp: str

    action: str  # buy / sell / hold
    confidence: float
    thesis: str

    supporting_factors: List[str] = field(default_factory=list)
    conflicting_factors: List[str] = field(default_factory=list)

    suggested_size_pct: float = 0.0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Risk manager output
# =========================================================

@dataclass
class RiskReport(SerializableDataclass):
    symbol: str
    timestamp: str

    approved: bool
    risk_score: float
    summary: str

    warnings: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None

    adjusted_size_pct: Optional[float] = None
    adjusted_stop_loss_pct: Optional[float] = None
    adjusted_take_profit_pct: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Final decision passed to execution
# =========================================================

@dataclass
class FinalDecision(SerializableDataclass):
    symbol: str
    timestamp: str

    action: str  # buy / sell / hold
    reason: str

    size_pct: float = 0.0
    order_type: str = "market"

    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Order validation
# =========================================================

@dataclass
class ValidationResult(SerializableDataclass):
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =========================================================
# Execution output
# =========================================================

@dataclass
class ExecutionResult(SerializableDataclass):
    symbol: str
    timestamp: str

    status: str  # filled / rejected / skipped / failed
    action: str

    filled_price: Optional[float] = None
    filled_qty: Optional[float] = None
    notional_value: Optional[float] = None

    fees: float = 0.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Portfolio / account snapshot
# =========================================================

@dataclass
class Position(SerializableDataclass):
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    updated_at: str = ""


@dataclass
class PortfolioState(SerializableDataclass):
    cash: float = 10000.0
    total_equity: float = 10000.0
    realized_pnl: float = 0.0
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    updated_at: str = ""


# =========================================================
# Memory / reflection layer
# =========================================================

@dataclass
class EpisodeRecord(SerializableDataclass):
    symbol: str
    timestamp: str

    raw_context: Dict[str, Any]
    analyst_views: List[Dict[str, Any]]

    proposal: Dict[str, Any]
    risk_report: Dict[str, Any]
    final_decision: Dict[str, Any]
    execution_result: Optional[Dict[str, Any]] = None

    portfolio_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionNote(SerializableDataclass):
    symbol: str
    timestamp: str

    lesson: str
    mistake: Optional[str] = None
    improvement: Optional[str] = None

    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Strategy memory
# =========================================================

@dataclass
class StrategyMemory(SerializableDataclass):
    updated_at: str = ""
    active_insights: List[str] = field(default_factory=list)
    risk_adjustments: Dict[str, Any] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _to_serializable_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}
# =========================================================
# Helper builders
# =========================================================

def build_episode_record(
    raw_context: RawContext,
    analyst_views: List[AnalystView],
    proposal: TradeProposal | Dict[str, Any],
    risk_report: RiskReport | Dict[str, Any],
    final_decision: FinalDecision | Dict[str, Any],
    execution_result: Optional[ExecutionResult | Dict[str, Any]],
    portfolio_snapshot: Dict[str, Any],
) -> EpisodeRecord:
    return EpisodeRecord(
        symbol=raw_context.symbol,
        timestamp=raw_context.timestamp,
        raw_context=_to_serializable_dict(raw_context) or {},
        analyst_views=[_to_serializable_dict(view) or {} for view in analyst_views],
        proposal=_to_serializable_dict(proposal) or {},
        risk_report=_to_serializable_dict(risk_report) or {},
        final_decision=_to_serializable_dict(final_decision) or {},
        execution_result=_to_serializable_dict(execution_result),
        portfolio_snapshot=portfolio_snapshot,
    )


def build_empty_execution_result(
    symbol: str,
    timestamp: str,
    action: str,
    status: str = "skipped",
    message: str = "",
) -> ExecutionResult:
    return ExecutionResult(
        symbol=symbol,
        timestamp=timestamp,
        status=status,
        action=action,
        message=message,
    )