from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypedDict

from trading_agent_v2.agents.critic_agent import CriticAgent
from trading_agent_v2.agents.market_analyst import MarketAnalyst
from trading_agent_v2.agents.news_analyst import NewsAnalyst
from trading_agent_v2.agents.planner_agent import PlannerAgent
from trading_agent_v2.agents.risk_manager import RiskManager
from trading_agent_v2.agents.trader_agent import TraderAgent
from trading_agent_v2.config import AppConfig, build_default_config
from trading_agent_v2.execution.okx_executor import OkxExecutor
from trading_agent_v2.execution.order_validator import OrderValidator
from trading_agent_v2.execution.paper_executor import PaperExecutor
from trading_agent_v2.execution.position_sizer import PositionSizer
from trading_agent_v2.llm.openai_client import OpenAIJsonClient
from trading_agent_v2.memory.episodic_memory import EpisodicMemoryStore
from trading_agent_v2.memory.pattern_memory import PatternMemoryStore
from trading_agent_v2.memory.reflection_engine import ReflectionEngine
from trading_agent_v2.memory.retrieval import MemoryRetriever
from trading_agent_v2.memory.strategic_memory import StrategicMemoryStore
from trading_agent_v2.portfolio.portfolio_manager import PortfolioManager
from trading_agent_v2.portfolio.trade_logger import TradeLogger
from trading_agent_v2.schemas import (
    RawContext,
)
from trading_agent_v2.skills import TradingSkills
from trading_agent_v2.tools.feature_builder import FeatureBuilder
from trading_agent_v2.tools.market_tools import MarketTools
from trading_agent_v2.tools.news_tools import NewsTools
from trading_agent_v2.tools.onchain_tools import OnchainTools
from trading_agent_v2.tools.social_tools import SocialTools

try:
    from langgraph.graph import END, START, StateGraph
except Exception as exc:  # pragma: no cover
    END = START = StateGraph = None
    _LANGGRAPH_IMPORT_ERROR: Exception | None = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None


@dataclass
class RuntimeServices:
    market_tools: MarketTools
    news_tools: NewsTools
    onchain_tools: OnchainTools
    social_tools: SocialTools
    market_analyst: MarketAnalyst
    news_analyst: NewsAnalyst
    feature_builder: FeatureBuilder
    llm_client: OpenAIJsonClient
    planner_agent: PlannerAgent
    critic_agent: CriticAgent
    memory_retriever: MemoryRetriever
    portfolio_manager: PortfolioManager
    episodic_memory: EpisodicMemoryStore
    reflection_engine: ReflectionEngine
    strategic_memory_store: StrategicMemoryStore
    pattern_memory_store: PatternMemoryStore
    trade_logger: TradeLogger
    trader_agent: TraderAgent
    risk_manager: RiskManager
    validator: OrderValidator
    position_sizer: PositionSizer
    executor: Any
    skills: TradingSkills


class TradingGraphState(TypedDict, total=False):
    symbol: str
    config: AppConfig
    execution_mode: str
    services: RuntimeServices
    portfolio: Any
    recent_memory: list[dict[str, Any]]
    strategy_memory: dict[str, Any]
    raw_context: RawContext
    market_prices: dict[str, float]
    portfolio_sync_error: str | None
    features: dict[str, Any]
    analyst_views: list[Any]
    retrieval_context: dict[str, Any]
    similar_cases: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    reviewed_proposals: list[dict[str, Any]]
    best_proposal: dict[str, Any]
    risk_report: Any
    final_decision: Any
    validation: Any
    execution_result: Any
    portfolio_snapshot: dict[str, Any]
    reflection_note: Any
    updated_strategy_memory: Any
    updated_pattern_memory: dict[str, Any]
    result: dict[str, Any]


def _setup_runtime(state: TradingGraphState) -> dict[str, Any]:
    config = state.get("config") or build_default_config()
    symbol = str(state["symbol"])
    execution_mode = str(config.execution.mode or "paper").lower().strip()

    market_tools = MarketTools(
        exchange_id="okx" if execution_mode == "okx" else "binanceus",
        fallback_exchange_id="coinbase",
    )
    news_tools = NewsTools()
    onchain_tools = OnchainTools()
    social_tools = SocialTools()

    market_analyst = MarketAnalyst()
    news_analyst = NewsAnalyst()

    feature_builder = FeatureBuilder()
    llm_client = OpenAIJsonClient(
        enabled=config.llm.enabled,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout_sec=config.llm.timeout_sec,
    )
    planner_agent = PlannerAgent(
        action_threshold=config.planner.action_threshold,
        min_trade_confidence=config.planner.min_trade_confidence,
        min_directional_size_pct=config.planner.min_directional_size_pct,
        max_directional_size_pct=config.planner.max_directional_size_pct,
        aggressive_size_multiplier=config.planner.aggressive_size_multiplier,
        llm_client=llm_client,
        llm_primary=config.llm.enabled,
    )
    critic_agent = CriticAgent(
        llm_client=llm_client,
        llm_primary=config.llm.enabled,
    )
    memory_retriever = MemoryRetriever()

    portfolio_manager = PortfolioManager(str(config.portfolio_file))
    episodic_memory = EpisodicMemoryStore(str(config.episodic_memory_file))
    reflection_engine = ReflectionEngine(str(config.reflection_file))
    strategic_memory_store = StrategicMemoryStore(str(config.strategy_memory_file))
    pattern_memory_store = PatternMemoryStore(str(config.pattern_memory_file))
    trade_logger = TradeLogger(str(config.run_log_file))

    trader_agent = TraderAgent(
        buy_threshold=config.trader.buy_threshold,
        sell_threshold=config.trader.sell_threshold,
        min_confidence=config.trader.min_confidence,
        default_buy_size_pct=config.trader.default_buy_size_pct,
        default_sell_size_pct=config.trader.default_sell_size_pct,
        min_directional_size_pct=config.trader.min_directional_size_pct,
        default_stop_loss_pct=config.trader.default_stop_loss_pct,
        default_take_profit_pct=config.trader.default_take_profit_pct,
        analyst_weights=config.trader.analyst_weights,
        llm_client=llm_client,
        llm_primary=config.llm.enabled,
    )
    risk_manager = RiskManager(
        max_total_invested_pct=config.risk.max_total_invested_pct,
        max_single_position_pct=config.risk.max_single_position_pct,
        min_proposal_confidence=config.risk.min_proposal_confidence,
        max_conflicting_factors=config.risk.max_conflicting_factors,
        max_loss_streak=config.risk.max_loss_streak,
        volatility_size_cut_ratio=config.risk.volatility_size_cut_ratio,
        high_risk_score_threshold=config.risk.high_risk_score_threshold,
        medium_risk_score_threshold=config.risk.medium_risk_score_threshold,
        elevated_risk_size_multiplier=config.risk.elevated_risk_size_multiplier,
        conflict_size_cut_ratio=config.risk.conflict_size_cut_ratio,
        volatility_stop_loss_tighten_ratio=config.risk.volatility_stop_loss_tighten_ratio,
    )
    validator = OrderValidator(
        min_size_pct=config.validation.min_size_pct,
        max_size_pct=config.validation.max_size_pct,
        allow_short=config.validation.allow_short,
    )
    position_sizer = PositionSizer(
        min_size_pct=config.validation.min_size_pct,
        max_size_pct=config.validation.max_size_pct,
        enforce_min_size=config.execution.enforce_min_size,
    )
    if execution_mode == "okx":
        executor = OkxExecutor(
            api_key=config.execution.okx_api_key,
            secret=config.execution.okx_secret,
            passphrase=config.execution.okx_passphrase,
            use_sandbox=config.execution.okx_use_sandbox,
            timeout_ms=config.execution.okx_timeout_ms,
            enable_rate_limit=config.execution.okx_enable_rate_limit,
        )
    elif execution_mode == "paper":
        executor = PaperExecutor(
            trading_fee_rate=config.execution.trading_fee_rate,
            slippage_rate=config.execution.slippage_rate,
        )
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    skills = TradingSkills(
        market_tools=market_tools,
        news_tools=news_tools,
        onchain_tools=onchain_tools,
        social_tools=social_tools,
        feature_builder=feature_builder,
        market_analyst=market_analyst,
        news_analyst=news_analyst,
        planner_agent=planner_agent,
        critic_agent=critic_agent,
        memory_retriever=memory_retriever,
        risk_manager=risk_manager,
        trader_agent=trader_agent,
        position_sizer=position_sizer,
        validator=validator,
        executor=executor,
        portfolio_manager=portfolio_manager,
        episodic_memory=episodic_memory,
        reflection_engine=reflection_engine,
        strategic_memory_store=strategic_memory_store,
        pattern_memory_store=pattern_memory_store,
        trade_logger=trade_logger,
        llm_client=llm_client,
    )

    services = RuntimeServices(
        market_tools=market_tools,
        news_tools=news_tools,
        onchain_tools=onchain_tools,
        social_tools=social_tools,
        market_analyst=market_analyst,
        news_analyst=news_analyst,
        feature_builder=feature_builder,
        llm_client=llm_client,
        planner_agent=planner_agent,
        critic_agent=critic_agent,
        memory_retriever=memory_retriever,
        portfolio_manager=portfolio_manager,
        episodic_memory=episodic_memory,
        reflection_engine=reflection_engine,
        strategic_memory_store=strategic_memory_store,
        pattern_memory_store=pattern_memory_store,
        trade_logger=trade_logger,
        trader_agent=trader_agent,
        risk_manager=risk_manager,
        validator=validator,
        position_sizer=position_sizer,
        executor=executor,
        skills=skills,
    )

    portfolio_manager.ensure_portfolio_exists(initial_cash=config.initial_cash)
    portfolio = portfolio_manager.load_portfolio()

    recent_memory = episodic_memory.load_recent(
        limit=config.memory.recent_episodes,
        symbol=symbol,
    )
    strategy_memory_obj = strategic_memory_store.load()
    strategy_memory = strategy_memory_obj.to_dict()
    pattern_memory = pattern_memory_store.load()
    strategy_memory["pattern_insights"] = (pattern_memory.get("metadata") or {}).get("insights", [])
    strategy_memory["pattern_stats"] = pattern_memory.get("patterns", {})

    return {
        "config": config,
        "execution_mode": execution_mode,
        "services": services,
        "portfolio": portfolio,
        "recent_memory": recent_memory,
        "strategy_memory": strategy_memory,
        "portfolio_sync_error": None,
    }


def _collect_data(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    symbol = str(state["symbol"])

    raw_context = services.skills.collect_raw_context(symbol=symbol)
    market_prices = services.skills.build_market_prices(raw_context)
    return {
        "raw_context": raw_context,
        "market_prices": market_prices,
    }


def _route_portfolio_sync(state: TradingGraphState) -> str:
    if str(state.get("execution_mode", "")).lower() == "okx":
        return "sync_portfolio"
    return "mark_to_market"


def _sync_portfolio(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    config = state["config"]
    portfolio = state["portfolio"]
    symbol = str(state["symbol"])
    market_prices = state["market_prices"]

    try:
        portfolio = services.executor.sync_portfolio_state(
            portfolio=portfolio,
            symbols=config.symbols or [symbol],
            market_prices=market_prices,
        )
        services.portfolio_manager.save_portfolio(portfolio)
        return {"portfolio": portfolio, "portfolio_sync_error": None}
    except Exception as exc:
        return {"portfolio_sync_error": str(exc)}


def _mark_to_market(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    portfolio = services.portfolio_manager.mark_to_market(
        state["portfolio"],
        state["market_prices"],
    )
    return {"portfolio": portfolio}


def _build_features(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    raw_context = state["raw_context"]
    symbol = str(state["symbol"])

    features = services.skills.build_features(symbol=symbol, raw_context=raw_context)
    return {"features": features}


def _run_analysts(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    raw_context = state["raw_context"]
    analyst_views = services.skills.run_analysts(raw_context)
    return {"analyst_views": analyst_views}


def _plan_proposals(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    symbol = str(state["symbol"])
    retrieval_context = services.skills.build_retrieval_context(
        symbol=symbol,
        features=state["features"],
        recent_memory=state["recent_memory"],
    )
    similar_cases = retrieval_context.get("similar_cases", [])

    proposals = services.skills.generate_proposals(
        symbol=symbol,
        analyst_views=state["analyst_views"],
        features=state["features"],
        portfolio=state["portfolio"],
        strategy_memory=state["strategy_memory"],
        similar_cases=similar_cases,
    )

    return {
        "retrieval_context": retrieval_context,
        "similar_cases": similar_cases,
        "proposals": proposals,
    }


def _review_proposals(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    symbol = str(state["symbol"])
    proposals = state.get("proposals") or []

    reviewed_proposals, best_proposal = services.skills.review_proposals(
        symbol=symbol,
        proposals=proposals,
        features=state["features"],
        similar_cases=state.get("similar_cases", []),
        strategy_memory=state.get("strategy_memory", {}),
    )

    return {
        "reviewed_proposals": reviewed_proposals,
        "best_proposal": best_proposal,
    }


def _run_risk(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    risk_report = services.skills.evaluate_risk(
        proposal=state["best_proposal"],
        portfolio=state["portfolio"],
        recent_memory=state["recent_memory"],
        strategy_memory=state["strategy_memory"],
    )
    return {"risk_report": risk_report}


def _make_final_decision(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    final_decision = services.skills.make_final_decision(
        proposal=state["best_proposal"],
        risk_report=state["risk_report"],
    )
    return {"final_decision": final_decision}


def _validate_order(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    validation = services.skills.validate_decision(
        decision=state["final_decision"],
        portfolio=state["portfolio"],
        market_prices=state["market_prices"],
    )
    return {"validation": validation}


def _route_execution(state: TradingGraphState) -> str:
    return "execute_order" if state["validation"].valid else "reject_execution"


def _execute_order(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    execution_result = services.skills.execute_decision(
        decision=state["final_decision"],
        portfolio=state["portfolio"],
        market_prices=state["market_prices"],
    )
    return {"execution_result": execution_result}


def _reject_execution(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    symbol = str(state["symbol"])
    final_decision = state["final_decision"]
    validation = state["validation"]

    execution_result = services.skills.build_rejected_execution(
        symbol=symbol,
        decision=final_decision,
        validation=validation,
    )
    return {"execution_result": execution_result}


def _update_portfolio(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]

    portfolio, snapshot = services.skills.apply_execution_and_snapshot(
        portfolio=state["portfolio"],
        execution_result=state["execution_result"],
        market_prices=state["market_prices"],
    )
    return {
        "portfolio": portfolio,
        "portfolio_snapshot": snapshot,
    }


def _update_memories(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    config = state["config"]
    symbol = str(state["symbol"])

    reflection_note, updated_strategy_memory, updated_pattern_memory = services.skills.update_memory_skills(
        symbol=symbol,
        config=config,
        raw_context=state["raw_context"],
        analyst_views=state["analyst_views"],
        proposal=state["best_proposal"],
        risk_report=state["risk_report"],
        final_decision=state["final_decision"],
        execution_result=state["execution_result"],
        portfolio_snapshot=state["portfolio_snapshot"],
    )

    return {
        "reflection_note": reflection_note,
        "updated_strategy_memory": updated_strategy_memory,
        "updated_pattern_memory": updated_pattern_memory,
    }


def _assemble_result(state: TradingGraphState) -> dict[str, Any]:
    services = state["services"]
    config = state["config"]

    result = services.skills.assemble_cycle_result(
        symbol=str(state["symbol"]),
        config=config,
        execution_mode=str(state["execution_mode"]),
        portfolio_sync_error=state.get("portfolio_sync_error"),
        raw_context=state["raw_context"],
        analyst_views=state["analyst_views"],
        features=state["features"],
        retrieval_context=state.get("retrieval_context", {}),
        proposals=state.get("proposals", []),
        reviewed_proposals=state.get("reviewed_proposals", []),
        best_proposal=state["best_proposal"],
        risk_report=state["risk_report"],
        final_decision=state["final_decision"],
        validation=state["validation"],
        execution_result=state["execution_result"],
        portfolio_snapshot=state["portfolio_snapshot"],
        reflection_note=state["reflection_note"],
        updated_strategy_memory=state["updated_strategy_memory"],
        updated_pattern_memory=state["updated_pattern_memory"],
    )

    return {"result": result}


def _ensure_langgraph_available() -> None:
    if _LANGGRAPH_IMPORT_ERROR is not None or StateGraph is None:
        raise RuntimeError(
            "LangGraph is required for v2 workflow. Install dependencies with "
            "`pip install -r requirements.txt` first."
        ) from _LANGGRAPH_IMPORT_ERROR


def _build_graph():
    _ensure_langgraph_available()
    builder = StateGraph(TradingGraphState)

    builder.add_node("setup_runtime", _setup_runtime)
    builder.add_node("collect_data", _collect_data)
    builder.add_node("sync_portfolio", _sync_portfolio)
    builder.add_node("mark_to_market", _mark_to_market)
    builder.add_node("build_features", _build_features)
    builder.add_node("run_analysts", _run_analysts)
    builder.add_node("plan_proposals", _plan_proposals)
    builder.add_node("review_proposals", _review_proposals)
    builder.add_node("run_risk", _run_risk)
    builder.add_node("make_final_decision", _make_final_decision)
    builder.add_node("validate_order", _validate_order)
    builder.add_node("execute_order", _execute_order)
    builder.add_node("reject_execution", _reject_execution)
    builder.add_node("update_portfolio", _update_portfolio)
    builder.add_node("update_memories", _update_memories)
    builder.add_node("assemble_result", _assemble_result)

    builder.add_edge(START, "setup_runtime")
    builder.add_edge("setup_runtime", "collect_data")
    builder.add_conditional_edges(
        "collect_data",
        _route_portfolio_sync,
        {
            "sync_portfolio": "sync_portfolio",
            "mark_to_market": "mark_to_market",
        },
    )
    builder.add_edge("sync_portfolio", "mark_to_market")
    builder.add_edge("mark_to_market", "build_features")
    builder.add_edge("build_features", "run_analysts")
    builder.add_edge("run_analysts", "plan_proposals")
    builder.add_edge("plan_proposals", "review_proposals")
    builder.add_edge("review_proposals", "run_risk")
    builder.add_edge("run_risk", "make_final_decision")
    builder.add_edge("make_final_decision", "validate_order")
    builder.add_conditional_edges(
        "validate_order",
        _route_execution,
        {
            "execute_order": "execute_order",
            "reject_execution": "reject_execution",
        },
    )
    builder.add_edge("execute_order", "update_portfolio")
    builder.add_edge("reject_execution", "update_portfolio")
    builder.add_edge("update_portfolio", "update_memories")
    builder.add_edge("update_memories", "assemble_result")
    builder.add_edge("assemble_result", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_trading_graph():
    return _build_graph()


def run_cycle_with_langgraph(symbol: str = "BTC/USDT", app_config: AppConfig | None = None) -> dict[str, Any]:
    config = app_config or build_default_config()
    graph = get_trading_graph()
    final_state = graph.invoke({"symbol": symbol, "config": config})
    return final_state["result"]


def run_batch_with_langgraph(
    symbols: list[str] | None = None,
    app_config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    config = app_config or build_default_config()
    target_symbols = symbols or config.symbols
    results: list[dict[str, Any]] = []

    for symbol in target_symbols:
        results.append(run_cycle_with_langgraph(symbol=symbol, app_config=config))

    return results
