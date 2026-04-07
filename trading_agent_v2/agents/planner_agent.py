from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from trading_agent_v2.llm.openai_client import OpenAIJsonClient


@dataclass
class Evidence:
    source: str
    direction: int  # +1 bullish, -1 bearish, 0 neutral
    strength: float  # 0..1
    weight: float
    rationale: str

    @property
    def contribution(self) -> float:
        return float(self.direction) * float(self.strength) * float(self.weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "direction": self.direction,
            "strength": round(self.strength, 4),
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 6),
            "rationale": self.rationale,
        }


class PlannerAgent:
    """
    Explainable reasoning planner.
    Instead of hardcoded templates, it builds explicit evidence items and
    aggregates them into directional edge + confidence.
    """

    def __init__(
        self,
        action_threshold: float = 0.04,
        min_trade_confidence: float = 0.30,
        min_directional_size_pct: float = 0.15,
        max_directional_size_pct: float = 1.00,
        aggressive_size_multiplier: float = 1.80,
        llm_client: OpenAIJsonClient | None = None,
        llm_primary: bool = True,
    ):
        self.action_threshold = action_threshold
        self.min_trade_confidence = min_trade_confidence
        self.min_directional_size_pct = max(0.0, min(1.0, min_directional_size_pct))
        self.max_directional_size_pct = max(
            self.min_directional_size_pct,
            min(1.0, max_directional_size_pct),
        )
        self.aggressive_size_multiplier = max(1.0, aggressive_size_multiplier)
        self.llm_client = llm_client
        self.llm_primary = llm_primary

    def _extract_view_fields(self, view: Any) -> tuple[str, float, str]:
        if isinstance(view, dict):
            return (
                str(view.get("bias", "neutral")).lower(),
                self._safe_float(view.get("confidence"), 0.0),
                str(view.get("analyst_name", "analyst")),
            )

        return (
            str(getattr(view, "bias", "neutral")).lower(),
            self._safe_float(getattr(view, "confidence", 0.0), 0.0),
            str(getattr(view, "analyst_name", "analyst")),
        )

    def generate_proposals(
        self,
        symbol: str,
        analyst_views: List[Any],
        features: Dict[str, Any],
        portfolio: Dict[str, Any],
        strategy_memory: Dict[str, Any] | None = None,
        similar_cases: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        strategy_memory = strategy_memory or {}
        similar_cases = similar_cases or []

        if self.llm_primary:
            return self._generate_with_llm(
                symbol=symbol,
                analyst_views=analyst_views,
                features=features,
                portfolio=portfolio,
                strategy_memory=strategy_memory,
                similar_cases=similar_cases,
            )

        evidence = self._build_evidence(
            analyst_views=analyst_views,
            features=features,
            strategy_memory=strategy_memory,
            similar_cases=similar_cases,
        )
        reasoning = self._aggregate_reasoning(evidence=evidence, features=features)
        decided_action = self._decide_action(reasoning)
        base_size = self._derive_base_size(reasoning)

        proposals = self._build_style_proposals(
            symbol=symbol,
            decided_action=decided_action,
            reasoning=reasoning,
            base_size=base_size,
            portfolio=portfolio,
        )
        return proposals

    def _generate_with_llm(
        self,
        symbol: str,
        analyst_views: List[Any],
        features: Dict[str, Any],
        portfolio: Dict[str, Any],
        strategy_memory: Dict[str, Any],
        similar_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.llm_client is None:
            raise RuntimeError("PlannerAgent requires an LLM client when llm_primary=true.")

        analyst_payload = []
        for view in analyst_views:
            if isinstance(view, dict):
                analyst_payload.append(
                    {
                        "analyst_name": view.get("analyst_name"),
                        "bias": view.get("bias"),
                        "confidence": view.get("confidence"),
                        "summary": view.get("summary"),
                        "supporting_signals": view.get("supporting_signals", []),
                        "risk_flags": view.get("risk_flags", []),
                    }
                )
            else:
                analyst_payload.append(
                    {
                        "analyst_name": getattr(view, "analyst_name", "analyst"),
                        "bias": getattr(view, "bias", "neutral"),
                        "confidence": getattr(view, "confidence", 0.0),
                        "summary": getattr(view, "summary", ""),
                        "supporting_signals": getattr(view, "supporting_signals", []),
                        "risk_flags": getattr(view, "risk_flags", []),
                    }
                )

        payload = {
            "symbol": symbol,
            "analyst_views": analyst_payload,
            "features": features,
            "portfolio": {
                "cash": (portfolio or {}).get("cash"),
                "total_equity": (portfolio or {}).get("total_equity"),
                "positions": list((portfolio or {}).get("positions", {}).keys()),
            },
            "strategy_memory": {
                "active_insights": strategy_memory.get("active_insights", []),
                "risk_adjustments": strategy_memory.get("risk_adjustments", {}),
                "pattern_insights": strategy_memory.get("pattern_insights", []),
            },
            "similar_cases": similar_cases[:5],
        }

        system_prompt = (
            "You are an explainable crypto trading planner. "
            "Generate exactly 3 proposals in JSON for styles defensive/base/aggressive. "
            "Use action in {buy,sell,hold}. "
            "Prefer directional base/aggressive actions; use hold only when evidence is highly uncertain. "
            "For buy/sell ensure size_pct in [0,1.00]. "
            "Return only valid JSON with key 'proposals'. "
            "Each proposal must include: proposal_id, style, action, confidence, size_pct, "
            "thesis, supporting_factors, conflicting_factors, reasoning_trace."
        )
        response = self.llm_client.complete_json(system_prompt=system_prompt, payload=payload)

        proposals_raw = response.get("proposals", [])
        if not isinstance(proposals_raw, list) or not proposals_raw:
            raise ValueError("PlannerAgent LLM response must include non-empty 'proposals' list.")

        normalized = self._normalize_llm_proposals(symbol=symbol, proposals_raw=proposals_raw)
        return normalized

    def _normalize_llm_proposals(
        self,
        symbol: str,
        proposals_raw: List[Any],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        default_styles = ["defensive", "base", "aggressive"]

        for idx, raw in enumerate(proposals_raw[:3]):
            if not isinstance(raw, dict):
                raise ValueError("PlannerAgent LLM proposal must be an object.")
            action = str(raw.get("action", "")).lower().strip()
            if action not in {"buy", "sell", "hold"}:
                raise ValueError(f"PlannerAgent LLM proposal has invalid action: {action!r}.")

            style = str(raw.get("style", "")).lower().strip()
            if style not in {"defensive", "base", "aggressive"}:
                raise ValueError(f"PlannerAgent LLM proposal has invalid style: {style!r}.")

            if "size_pct" not in raw:
                raise ValueError("PlannerAgent LLM proposal is missing size_pct.")
            try:
                size_pct = float(raw.get("size_pct"))
            except (TypeError, ValueError) as exc:
                raise ValueError("PlannerAgent LLM proposal size_pct must be numeric.") from exc
            if action == "hold":
                size_pct = 0.0
            if action in {"buy", "sell"}:
                size_pct = max(self.min_directional_size_pct, size_pct)
            size_pct = max(0.0, min(self.max_directional_size_pct, size_pct))

            if "confidence" not in raw:
                raise ValueError("PlannerAgent LLM proposal is missing confidence.")
            try:
                confidence = float(raw.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise ValueError("PlannerAgent LLM proposal confidence must be numeric.") from exc
            confidence = max(0.0, min(1.0, confidence))
            thesis = str(raw.get("thesis", "") or "").strip()
            if not thesis:
                raise ValueError("PlannerAgent LLM proposal is missing thesis.")

            supporting_factors = self._normalize_text_list(raw.get("supporting_factors"))
            conflicting_factors = self._normalize_text_list(raw.get("conflicting_factors"))
            reasoning_trace = raw.get("reasoning_trace")
            if not isinstance(reasoning_trace, dict):
                raise ValueError("PlannerAgent LLM proposal reasoning_trace must be an object.")
            reasoning_trace.setdefault("planner_type", "openai_llm")
            reasoning_trace.setdefault("llm_response_index", idx)
            reasoning_trace.setdefault("raw_json", json.dumps(raw, ensure_ascii=False)[:1500])

            proposal_id = str(raw.get("proposal_id", style)).strip() or style
            if proposal_id not in {"defensive", "base", "aggressive"}:
                raise ValueError(f"PlannerAgent LLM proposal has invalid proposal_id: {proposal_id!r}.")

            normalized.append(
                {
                    "proposal_id": proposal_id,
                    "symbol": symbol,
                    "action": action,
                    "size_pct": size_pct,
                    "confidence": confidence,
                    "thesis": thesis,
                    "style": style,
                    "supporting_factors": supporting_factors,
                    "conflicting_factors": conflicting_factors,
                    "reasoning_trace": reasoning_trace,
                    "metadata": {
                        "planner_type": "openai_llm",
                    },
                }
            )

        if len(normalized) != 3:
            raise ValueError("PlannerAgent LLM must return exactly 3 proposals.")

        by_style = {item.get("style"): item for item in normalized}
        missing_styles = [style for style in default_styles if style not in by_style]
        if missing_styles:
            raise ValueError(f"PlannerAgent LLM missing proposal style(s): {', '.join(missing_styles)}.")
        return [by_style[style] for style in default_styles]

    def _normalize_text_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(value)]

    def _build_evidence(
        self,
        analyst_views: List[Any],
        features: Dict[str, Any],
        strategy_memory: Dict[str, Any],
        similar_cases: List[Dict[str, Any]],
    ) -> List[Evidence]:
        evidence: List[Evidence] = []
        evidence.extend(self._analyst_evidence(analyst_views))
        evidence.extend(self._market_evidence(features))
        evidence.extend(self._sentiment_evidence(features))
        evidence.extend(self._onchain_evidence(features))
        evidence.extend(self._memory_pattern_evidence(features, strategy_memory))
        evidence.extend(self._similar_case_evidence(similar_cases))
        return evidence

    def _analyst_evidence(self, analyst_views: List[Any]) -> List[Evidence]:
        output: List[Evidence] = []
        weight_map = {
            "market_analyst": 1.0,
            "news_analyst": 0.85,
        }
        for view in analyst_views:
            bias, confidence, analyst_name = self._extract_view_fields(view)
            direction = 0
            if bias == "bullish":
                direction = 1
            elif bias == "bearish":
                direction = -1

            output.append(
                Evidence(
                    source=f"analyst:{analyst_name}",
                    direction=direction,
                    strength=max(0.0, min(1.0, confidence)),
                    weight=weight_map.get(analyst_name, 0.8),
                    rationale=f"{analyst_name} bias={bias}, confidence={confidence:.2f}",
                )
            )
        return output

    def _market_evidence(self, features: Dict[str, Any]) -> List[Evidence]:
        output: List[Evidence] = []
        trend = str(features.get("trend", "neutral")).lower()
        rsi = self._safe_float(features.get("rsi"), 50.0)
        atr = self._safe_float(features.get("atr"), 0.0)

        if trend == "up":
            output.append(
                Evidence(
                    source="market:trend",
                    direction=1,
                    strength=0.55,
                    weight=0.45,
                    rationale="EMA structure indicates upward trend.",
                )
            )
        elif trend == "down":
            output.append(
                Evidence(
                    source="market:trend",
                    direction=-1,
                    strength=0.55,
                    weight=0.45,
                    rationale="EMA structure indicates downward trend.",
                )
            )

        if rsi <= 35:
            output.append(
                Evidence(
                    source="market:rsi_rebound",
                    direction=1,
                    strength=min(1.0, (35 - rsi) / 15.0),
                    weight=0.22,
                    rationale=f"RSI={rsi:.2f} is in oversold zone.",
                )
            )
        elif rsi >= 65:
            output.append(
                Evidence(
                    source="market:rsi_overheat",
                    direction=-1,
                    strength=min(1.0, (rsi - 65) / 15.0),
                    weight=0.22,
                    rationale=f"RSI={rsi:.2f} is in overbought zone.",
                )
            )

        if atr >= 0.035:
            output.append(
                Evidence(
                    source="market:high_volatility",
                    direction=0,
                    strength=min(1.0, (atr - 0.03) / 0.03),
                    weight=0.20,
                    rationale=f"ATR={atr:.4f} suggests elevated volatility uncertainty.",
                )
            )
        return output

    def _sentiment_evidence(self, features: Dict[str, Any]) -> List[Evidence]:
        output: List[Evidence] = []
        news_sentiment = self._safe_float(features.get("news_sentiment"), 0.0)
        social_sentiment = self._safe_float(features.get("social_sentiment"), 0.0)
        blended = news_sentiment * 0.6 + social_sentiment * 0.4

        direction = 0
        if blended > 0.08:
            direction = 1
        elif blended < -0.08:
            direction = -1

        output.append(
            Evidence(
                source="sentiment:news_social",
                direction=direction,
                strength=min(1.0, abs(blended)),
                weight=0.28,
                rationale=(
                    f"blended_sentiment={blended:.3f} "
                    f"(news={news_sentiment:.3f}, social={social_sentiment:.3f})."
                ),
            )
        )
        return output

    def _onchain_evidence(self, features: Dict[str, Any]) -> List[Evidence]:
        signal = str(features.get("onchain_signal", "neutral")).lower()
        score = self._safe_float(features.get("onchain_score"), 0.0)

        direction = 0
        if signal == "bullish":
            direction = 1
        elif signal == "bearish":
            direction = -1

        return [
            Evidence(
                source="onchain:signal",
                direction=direction,
                strength=min(1.0, abs(score)),
                weight=0.24,
                rationale=f"onchain_signal={signal}, onchain_score={score:.3f}",
            )
        ]

    def _memory_pattern_evidence(
        self,
        features: Dict[str, Any],
        strategy_memory: Dict[str, Any],
    ) -> List[Evidence]:
        pattern_stats = strategy_memory.get("pattern_stats", {}) or {}
        if not isinstance(pattern_stats, dict):
            return []

        trend = str(features.get("trend", "down")).lower()
        sentiment_bucket = self._sentiment_bucket(features.get("news_sentiment", 0.0))
        buy_key = f"trend={trend}|sentiment={sentiment_bucket}|action=buy"
        sell_key = f"trend={trend}|sentiment={sentiment_bucket}|action=sell"

        buy_stat = pattern_stats.get(buy_key, {}) if isinstance(pattern_stats.get(buy_key), dict) else {}
        sell_stat = pattern_stats.get(sell_key, {}) if isinstance(pattern_stats.get(sell_key), dict) else {}

        buy_edge = self._winrate_edge(buy_stat)
        sell_edge = self._winrate_edge(sell_stat)
        net_edge = buy_edge - sell_edge

        if abs(net_edge) < 0.05:
            return []

        direction = 1 if net_edge > 0 else -1
        strength = min(1.0, abs(net_edge) * 2.0)
        return [
            Evidence(
                source="memory:pattern_regime",
                direction=direction,
                strength=strength,
                weight=0.24,
                rationale=(
                    f"pattern edge in regime trend={trend}, sentiment={sentiment_bucket}; "
                    f"buy_edge={buy_edge:.3f}, sell_edge={sell_edge:.3f}"
                ),
            )
        ]

    def _similar_case_evidence(self, similar_cases: List[Dict[str, Any]]) -> List[Evidence]:
        if not similar_cases:
            return []

        buy_wins = buy_losses = sell_wins = sell_losses = 0
        for case in similar_cases:
            action = str(case.get("action", "")).lower()
            outcome = str(case.get("outcome", "")).lower()
            if action == "buy":
                if outcome == "win":
                    buy_wins += 1
                elif outcome == "loss":
                    buy_losses += 1
            elif action == "sell":
                if outcome == "win":
                    sell_wins += 1
                elif outcome == "loss":
                    sell_losses += 1

        buy_edge = self._outcome_edge(buy_wins, buy_losses)
        sell_edge = self._outcome_edge(sell_wins, sell_losses)
        net_edge = buy_edge - sell_edge
        if abs(net_edge) < 0.10:
            return []

        direction = 1 if net_edge > 0 else -1
        strength = min(1.0, abs(net_edge))
        return [
            Evidence(
                source="memory:similar_cases",
                direction=direction,
                strength=strength,
                weight=0.20,
                rationale=(
                    "similar-case edge "
                    f"(buy_wins={buy_wins}, buy_losses={buy_losses}, "
                    f"sell_wins={sell_wins}, sell_losses={sell_losses})"
                ),
            )
        ]

    def _aggregate_reasoning(self, evidence: List[Evidence], features: Dict[str, Any]) -> Dict[str, Any]:
        positive = sum(max(0.0, ev.contribution) for ev in evidence)
        negative = abs(sum(min(0.0, ev.contribution) for ev in evidence))
        net = positive - negative
        total_abs = positive + negative
        normalized_edge = (net / total_abs) if total_abs > 1e-9 else 0.0

        conflict_index = 1.0
        if total_abs > 1e-9:
            conflict_index = 1.0 - abs(positive - negative) / total_abs

        volatility = self._safe_float(features.get("volatility"), 0.0)
        volatility_penalty = max(0.0, min(0.12, (volatility - 0.028) * 3.0))
        uncertainty_penalty = conflict_index * 0.12 + volatility_penalty

        base_confidence = 0.48 + min(0.38, abs(normalized_edge) * 0.45)
        confidence = max(0.0, min(1.0, base_confidence - uncertainty_penalty))

        return {
            "positive_score": round(positive, 6),
            "negative_score": round(negative, 6),
            "net_score": round(net, 6),
            "normalized_edge": round(normalized_edge, 6),
            "conflict_index": round(conflict_index, 6),
            "volatility_penalty": round(volatility_penalty, 6),
            "uncertainty_penalty": round(uncertainty_penalty, 6),
            "base_confidence": round(base_confidence, 6),
            "confidence": round(confidence, 6),
            "evidence": [ev.to_dict() for ev in evidence],
        }

    def _decide_action(self, reasoning: Dict[str, Any]) -> str:
        edge = self._safe_float(reasoning.get("normalized_edge"), 0.0)
        confidence = self._safe_float(reasoning.get("confidence"), 0.0)
        if confidence < self.min_trade_confidence:
            return "hold"
        if edge >= self.action_threshold:
            return "buy"
        if edge <= -self.action_threshold:
            return "sell"
        return "hold"

    def _derive_base_size(self, reasoning: Dict[str, Any]) -> float:
        confidence = self._safe_float(reasoning.get("confidence"), 0.0)
        conviction = abs(self._safe_float(reasoning.get("normalized_edge"), 0.0))
        raw = 0.12 + conviction * 0.28 + max(0.0, confidence - 0.50) * 0.20
        upper = min(self.max_directional_size_pct, 0.80)
        return max(self.min_directional_size_pct, min(upper, raw))

    def _build_style_proposals(
        self,
        symbol: str,
        decided_action: str,
        reasoning: Dict[str, Any],
        base_size: float,
        portfolio: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        trace = {
            "planner_type": "explainable_reasoner_v1",
            **reasoning,
            "action_gate": {
                "action_threshold": self.action_threshold,
                "min_trade_confidence": self.min_trade_confidence,
                "chosen_action": decided_action,
            },
        }
        supporting, conflicting = self._extract_factor_lists(
            evidence=reasoning.get("evidence", []),
            action=decided_action,
        )
        explanation = self._build_explanation(decided_action, reasoning, supporting, conflicting)

        proposals: List[Dict[str, Any]] = [
            {
                "proposal_id": "defensive",
                "symbol": symbol,
                "action": "hold",
                "size_pct": 0.0,
                "confidence": max(0.50, self._safe_float(reasoning.get("confidence"), 0.50) - 0.03),
                "thesis": "Defensive policy: keep optionality while evidence remains uncertain.",
                "style": "defensive",
                "supporting_factors": supporting,
                "conflicting_factors": conflicting,
                "reasoning_trace": trace,
                "metadata": {
                    "portfolio_position_count": len((portfolio or {}).get("positions", {})),
                    "explanation": explanation,
                },
            }
        ]

        if decided_action == "hold":
            proposals.append(
                {
                    "proposal_id": "base",
                    "symbol": symbol,
                    "action": "hold",
                    "size_pct": 0.0,
                    "confidence": self._safe_float(reasoning.get("confidence"), 0.50),
                    "thesis": explanation,
                    "style": "base",
                    "supporting_factors": supporting,
                    "conflicting_factors": conflicting,
                    "reasoning_trace": trace,
                    "metadata": {"risk_mode": "wait_for_clearer_edge"},
                }
            )
            proposals.append(
                {
                    "proposal_id": "aggressive",
                    "symbol": symbol,
                    "action": "hold",
                    "size_pct": 0.0,
                    "confidence": max(0.45, self._safe_float(reasoning.get("confidence"), 0.50) - 0.04),
                    "thesis": "Even aggressive lane remains flat because directional edge is insufficient.",
                    "style": "aggressive",
                    "supporting_factors": supporting,
                    "conflicting_factors": conflicting,
                    "reasoning_trace": trace,
                    "metadata": {"risk_mode": "edge_not_clear"},
                }
            )
            return proposals

        base_conf = self._safe_float(reasoning.get("confidence"), 0.50)
        proposals.append(
            {
                "proposal_id": "base",
                "symbol": symbol,
                "action": decided_action,
                "size_pct": round(base_size, 4),
                "confidence": base_conf,
                "thesis": explanation,
                "style": "base",
                "supporting_factors": supporting,
                "conflicting_factors": conflicting,
                "reasoning_trace": trace,
                "metadata": {"risk_mode": "balanced"},
            }
        )

        aggressive_size = min(
            self.max_directional_size_pct,
            base_size * self.aggressive_size_multiplier,
        )
        aggressive_conf = max(0.0, min(1.0, base_conf - 0.03))
        proposals.append(
            {
                "proposal_id": "aggressive",
                "symbol": symbol,
                "action": decided_action,
                "size_pct": round(aggressive_size, 4),
                "confidence": aggressive_conf,
                "thesis": (
                    f"{explanation} Aggressive lane scales size from {base_size:.3f} "
                    f"to {aggressive_size:.3f}."
                ),
                "style": "aggressive",
                "supporting_factors": supporting,
                "conflicting_factors": conflicting,
                "reasoning_trace": trace,
                "metadata": {"risk_mode": "high_conviction"},
            }
        )
        return proposals

    def _extract_factor_lists(
        self,
        evidence: List[Dict[str, Any]],
        action: str,
    ) -> tuple[List[str], List[str]]:
        supporting: List[str] = []
        conflicting: List[str] = []
        desired_direction = 1 if action == "buy" else -1

        for ev in evidence:
            source = str(ev.get("source", "signal"))
            direction = int(self._safe_float(ev.get("direction"), 0))
            contribution = self._safe_float(ev.get("contribution"), 0.0)

            if action == "hold":
                if direction == 0:
                    supporting.append(f"{source}:uncertain")
                continue

            if direction == desired_direction and contribution > 0:
                supporting.append(f"{source}:supportive")
            elif direction == -desired_direction and contribution < 0:
                conflicting.append(f"{source}:opposing")
            elif direction == 0:
                conflicting.append(f"{source}:uncertainty")

        return self._dedupe_preserve_order(supporting), self._dedupe_preserve_order(conflicting)

    def _build_explanation(
        self,
        action: str,
        reasoning: Dict[str, Any],
        supporting: List[str],
        conflicting: List[str],
    ) -> str:
        edge = self._safe_float(reasoning.get("normalized_edge"), 0.0)
        confidence = self._safe_float(reasoning.get("confidence"), 0.0)
        conflict_index = self._safe_float(reasoning.get("conflict_index"), 0.0)
        support_count = len(supporting)
        conflict_count = len(conflicting)

        if action == "hold":
            return (
                "No execution edge after evidence aggregation: "
                f"edge={edge:.3f}, confidence={confidence:.2f}, conflict={conflict_index:.2f}. "
                f"support={support_count}, conflict={conflict_count}."
            )
        return (
            f"Explainable reasoning prefers {action.upper()}: "
            f"edge={edge:.3f}, confidence={confidence:.2f}, conflict={conflict_index:.2f}, "
            f"support={support_count}, conflict={conflict_count}."
        )

    def _winrate_edge(self, stats: Dict[str, Any]) -> float:
        if not isinstance(stats, dict):
            return 0.0
        count = int(stats.get("count", 0))
        if count < 3:
            return 0.0
        wins = self._safe_float(stats.get("wins"), 0.0)
        losses = self._safe_float(stats.get("losses"), 0.0)
        total = wins + losses
        if total <= 0:
            return 0.0
        return (wins / total) - 0.5

    def _outcome_edge(self, wins: int, losses: int) -> float:
        total = wins + losses
        if total < 2:
            return 0.0
        return (wins - losses) / total

    def _sentiment_bucket(self, value: Any) -> str:
        numeric = self._safe_float(value, 0.0)
        if numeric >= 0.2:
            return "positive"
        if numeric <= -0.2:
            return "negative"
        return "neutral"

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        output = []
        for item in items:
            if item not in seen:
                seen.add(item)
                output.append(item)
        return output
