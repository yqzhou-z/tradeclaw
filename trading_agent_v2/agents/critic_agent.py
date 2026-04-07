from __future__ import annotations

import json
from typing import Any, Dict, List

from trading_agent_v2.llm.openai_client import OpenAIJsonClient


class CriticAgent:
    """
    Explainable critic.
    Evaluates each proposal through auditable checks and exposes score impacts.
    """

    def __init__(
        self,
        llm_client: OpenAIJsonClient | None = None,
        llm_primary: bool = True,
    ):
        self.llm_client = llm_client
        self.llm_primary = llm_primary

    def review(
        self,
        proposals: List[Dict[str, Any]],
        features: Dict[str, Any],
        similar_cases: List[Dict[str, Any]] | None = None,
        strategy_memory: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        similar_cases = similar_cases or []
        strategy_memory = strategy_memory or {}

        if self.llm_primary:
            return self._review_with_llm(
                proposals=proposals,
                features=features,
                similar_cases=similar_cases,
                strategy_memory=strategy_memory,
            )

        reviewed = []
        pattern_stats = strategy_memory.get("pattern_stats", {}) or {}
        for proposal in proposals:
            reviewed.append(
                self._review_single(
                    proposal=proposal,
                    features=features,
                    similar_cases=similar_cases,
                    pattern_stats=pattern_stats,
                )
            )

        return self._sort_reviews(reviewed)

    def _review_with_llm(
        self,
        proposals: List[Dict[str, Any]],
        features: Dict[str, Any],
        similar_cases: List[Dict[str, Any]],
        strategy_memory: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self.llm_client is None:
            raise RuntimeError("CriticAgent requires an LLM client when llm_primary=true.")
        if not proposals:
            raise ValueError("CriticAgent received no proposals to review.")

        system_prompt = (
            "You are an explainable trading critic. "
            "Review one proposal and return JSON with keys: proposal_id, confidence, critic_score, "
            "critic_checks(list of {name,category,impact,message}), "
            "critic_strengths, critic_weaknesses, critic_reasoning. "
            "Use key impact exactly inside critic_checks (never imapct). "
            "Keep your output concise and strictly valid JSON."
        )

        llm_results: List[Dict[str, Any]] = []
        for proposal in proposals:
            payload = {
                "features": features,
                "proposal": proposal,
                "similar_cases": similar_cases[:5],
                "strategy_memory": {
                    "active_insights": strategy_memory.get("active_insights", []),
                    "risk_adjustments": strategy_memory.get("risk_adjustments", {}),
                    "pattern_insights": strategy_memory.get("pattern_insights", []),
                },
            }
            response = self.llm_client.complete_json(system_prompt=system_prompt, payload=payload)
            normalized = self._normalize_single_llm_review(proposal=proposal, raw=response)
            llm_results.append(normalized)

        if len(llm_results) != len(proposals):
            raise ValueError("CriticAgent LLM did not review every proposal.")
        return self._sort_reviews(llm_results)

    def _normalize_single_llm_review(
        self,
        proposal: Dict[str, Any],
        raw: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("CriticAgent LLM review must be a JSON object.")

        merged = dict(proposal)
        if "confidence" not in raw:
            raise ValueError("CriticAgent LLM review is missing confidence.")
        try:
            conf = float(raw.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("CriticAgent LLM review confidence must be numeric.") from exc
        conf = max(0.0, min(1.0, conf))
        if "critic_score" not in raw:
            raise ValueError("CriticAgent LLM review is missing critic_score.")
        try:
            llm_reported_score = float(raw.get("critic_score"))
        except (TypeError, ValueError) as exc:
            raise ValueError("CriticAgent LLM review critic_score must be numeric.") from exc
        llm_reported_score = max(0.0, min(1.0, llm_reported_score))
        checks = raw.get("critic_checks", [])
        if not isinstance(checks, list):
            raise ValueError("CriticAgent LLM review must include critic_checks as a list.")
        norm_checks = []
        for check in checks:
            if not isinstance(check, dict):
                raise ValueError("CriticAgent critic_checks items must be objects.")
            impact_raw = check.get("impact")
            if impact_raw is None:
                raise ValueError("CriticAgent critic_checks items must include 'impact'.")
            norm_checks.append(
                {
                    "name": str(check.get("name", "llm_check")),
                    "category": str(check.get("category", "llm")),
                    "impact": round(self._safe_float(impact_raw, 0.0), 6),
                    "message": str(check.get("message", "")),
                }
            )
        norm_checks.sort(
            key=lambda item: abs(self._safe_float(item.get("impact"), 0.0)),
            reverse=True,
        )

        total_delta = sum(self._safe_float(item.get("impact"), 0.0) for item in norm_checks)
        derived_score = max(0.0, min(1.0, conf + total_delta))
        final_score = derived_score if norm_checks else llm_reported_score

        strengths = [item["message"] for item in norm_checks if item.get("impact", 0.0) > 0 and item.get("message")]
        weaknesses = [item["message"] for item in norm_checks if item.get("impact", 0.0) < 0 and item.get("message")]
        if not strengths:
            raw_strengths = raw.get("critic_strengths", [])
            if isinstance(raw_strengths, list):
                strengths = [str(x) for x in raw_strengths if str(x).strip()]
        if not weaknesses:
            raw_weaknesses = raw.get("critic_weaknesses", [])
            if isinstance(raw_weaknesses, list):
                weaknesses = [str(x) for x in raw_weaknesses if str(x).strip()]

        critic_reasoning = raw.get("critic_reasoning", {})
        if not isinstance(critic_reasoning, dict):
            raise ValueError("CriticAgent LLM review critic_reasoning must be an object.")
        critic_reasoning.setdefault("critic_type", "openai_llm")
        critic_reasoning.setdefault("raw_json", json.dumps(raw, ensure_ascii=False)[:1500])
        critic_reasoning["start_confidence"] = round(conf, 6)
        critic_reasoning["total_delta"] = round(total_delta, 6)
        critic_reasoning["derived_score"] = round(derived_score, 6)
        critic_reasoning["llm_reported_score"] = round(llm_reported_score, 6)
        critic_reasoning["final_score"] = round(final_score, 6)
        critic_reasoning["score_source"] = "checks_derived" if norm_checks else "llm_reported"

        merged["confidence"] = conf
        merged["critic_score"] = final_score
        merged["critic_checks"] = norm_checks
        merged["critic_strengths"] = [str(x) for x in strengths if str(x).strip()]
        merged["critic_weaknesses"] = [str(x) for x in weaknesses if str(x).strip()]
        merged["critic_reasoning"] = critic_reasoning
        metadata = dict(merged.get("metadata", {}))
        metadata["critic_type"] = "openai_llm"
        metadata["critic_score_source"] = critic_reasoning.get("score_source")
        metadata["llm_reported_confidence"] = round(conf, 6)
        metadata["llm_reported_critic_score"] = round(llm_reported_score, 6)
        merged["metadata"] = metadata
        return merged

    def _review_single(
        self,
        proposal: Dict[str, Any],
        features: Dict[str, Any],
        similar_cases: List[Dict[str, Any]],
        pattern_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        p = proposal.copy()
        action = str(p.get("action", "hold")).lower()
        size_pct = self._safe_float(p.get("size_pct"), 0.0)
        start_conf = self._safe_float(p.get("confidence"), 0.50)

        checks: List[Dict[str, Any]] = []
        checks.extend(self._check_directional_alignment(action, features))
        checks.extend(self._check_sentiment_alignment(action, features))
        checks.extend(self._check_size_and_volatility(action, size_pct, features))
        checks.extend(self._check_similar_cases(action, similar_cases))
        checks.extend(self._check_pattern_regime(action, features, pattern_stats))
        checks.extend(self._check_reasoning_trace(p))

        total_delta = sum(self._safe_float(item.get("impact"), 0.0) for item in checks)
        final_conf = max(0.0, min(1.0, start_conf + total_delta))

        weaknesses = [item["message"] for item in checks if item.get("impact", 0.0) < 0]
        strengths = [item["message"] for item in checks if item.get("impact", 0.0) > 0]

        p["critic_checks"] = checks
        p["critic_strengths"] = strengths
        p["critic_weaknesses"] = weaknesses
        p["critic_score"] = final_conf
        p["confidence"] = final_conf
        p["critic_reasoning"] = {
            "critic_type": "explainable_critic_v1",
            "start_confidence": round(start_conf, 6),
            "total_delta": round(total_delta, 6),
            "final_confidence": round(final_conf, 6),
            "penalty_count": len(weaknesses),
            "credit_count": len(strengths),
        }
        return p

    def _check_directional_alignment(
        self,
        action: str,
        features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        trend = str(features.get("trend", "neutral")).lower()
        if action not in {"buy", "sell"}:
            return [
                self._check(
                    name="direction_alignment",
                    impact=0.01,
                    message="Non-directional action avoids trend mismatch risk.",
                    category="market",
                )
            ]

        if action == "buy" and trend == "down":
            return [
                self._check(
                    name="direction_alignment",
                    impact=-0.12,
                    message="BUY conflicts with downtrend.",
                    category="market",
                )
            ]
        if action == "sell" and trend == "up":
            return [
                self._check(
                    name="direction_alignment",
                    impact=-0.12,
                    message="SELL conflicts with uptrend.",
                    category="market",
                )
            ]
        return [
            self._check(
                name="direction_alignment",
                impact=0.02,
                message=f"{action.upper()} aligns with current trend.",
                category="market",
            )
        ]

    def _check_sentiment_alignment(
        self,
        action: str,
        features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if action not in {"buy", "sell"}:
            return []

        news_sent = self._safe_float(features.get("news_sentiment"), 0.0)
        social_sent = self._safe_float(features.get("social_sentiment"), 0.0)
        blended = news_sent * 0.6 + social_sent * 0.4

        if action == "buy" and blended < -0.12:
            return [
                self._check(
                    name="sentiment_alignment",
                    impact=-0.08,
                    message=f"BUY faces adverse sentiment (blended={blended:.3f}).",
                    category="sentiment",
                )
            ]
        if action == "sell" and blended > 0.12:
            return [
                self._check(
                    name="sentiment_alignment",
                    impact=-0.08,
                    message=f"SELL opposes constructive sentiment (blended={blended:.3f}).",
                    category="sentiment",
                )
            ]

        if abs(blended) < 0.08:
            return [
                self._check(
                    name="sentiment_alignment",
                    impact=0.0,
                    message=f"Sentiment is neutral (blended={blended:.3f}).",
                    category="sentiment",
                )
            ]
        return [
            self._check(
                name="sentiment_alignment",
                impact=0.02,
                message=f"Sentiment supports {action.upper()} (blended={blended:.3f}).",
                category="sentiment",
            )
        ]

    def _check_size_and_volatility(
        self,
        action: str,
        size_pct: float,
        features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        checks: List[Dict[str, Any]] = []
        if action not in {"buy", "sell"}:
            return checks

        atr = self._safe_float(features.get("atr"), 0.0)
        if size_pct > 0.12:
            checks.append(
                self._check(
                    name="size_aggressiveness",
                    impact=-0.05,
                    message=f"Position size {size_pct:.3f} is aggressive.",
                    category="risk",
                )
            )
        elif size_pct <= 0.10:
            checks.append(
                self._check(
                    name="size_aggressiveness",
                    impact=0.01,
                    message=f"Position size {size_pct:.3f} is conservative.",
                    category="risk",
                )
            )

        if atr >= 0.03 and size_pct >= 0.12:
            checks.append(
                self._check(
                    name="volatility_sizing",
                    impact=-0.06,
                    message=f"High ATR ({atr:.4f}) with large size increases execution risk.",
                    category="risk",
                )
            )
        return checks

    def _check_similar_cases(
        self,
        action: str,
        similar_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if action not in {"buy", "sell"}:
            return []

        matched = [
            case for case in similar_cases
            if str(case.get("action", "")).lower() == action
        ]
        if len(matched) < 2:
            return []

        wins = sum(1 for case in matched if str(case.get("outcome", "")).lower() == "win")
        losses = sum(1 for case in matched if str(case.get("outcome", "")).lower() == "loss")
        outcome_count = wins + losses
        if outcome_count < 2:
            return []

        loss_ratio = losses / outcome_count
        win_ratio = wins / outcome_count
        if loss_ratio >= 0.6:
            return [
                self._check(
                    name="similar_case_outcome",
                    impact=-0.10,
                    message=f"Similar {action.upper()} cases underperformed (loss_ratio={loss_ratio:.2f}).",
                    category="memory",
                )
            ]
        if win_ratio >= 0.6:
            return [
                self._check(
                    name="similar_case_outcome",
                    impact=0.03,
                    message=f"Similar {action.upper()} cases were supportive (win_ratio={win_ratio:.2f}).",
                    category="memory",
                )
            ]
        return []

    def _check_pattern_regime(
        self,
        action: str,
        features: Dict[str, Any],
        pattern_stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if action not in {"buy", "sell"}:
            return []
        if not isinstance(pattern_stats, dict):
            return []

        key = self._build_pattern_key(features=features, action=action)
        stats = pattern_stats.get(key, {})
        if not isinstance(stats, dict):
            return []

        count = int(stats.get("count", 0))
        if count < 3:
            return []
        wins = self._safe_float(stats.get("wins"), 0.0)
        losses = self._safe_float(stats.get("losses"), 0.0)
        total = wins + losses
        if total <= 0:
            return []
        win_rate = wins / total
        if losses > wins:
            return [
                self._check(
                    name="pattern_regime",
                    impact=-0.10,
                    message=f"Pattern memory disfavors regime ({key}, win_rate={win_rate:.2f}).",
                    category="memory",
                )
            ]
        if wins > losses and win_rate >= 0.60:
            return [
                self._check(
                    name="pattern_regime",
                    impact=0.03,
                    message=f"Pattern memory supports regime ({key}, win_rate={win_rate:.2f}).",
                    category="memory",
                )
            ]
        return []

    def _check_reasoning_trace(self, proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        trace = proposal.get("reasoning_trace", {})
        if not isinstance(trace, dict):
            return []

        conflict = self._safe_float(trace.get("conflict_index"), 0.0)
        confidence = self._safe_float(trace.get("confidence"), 0.5)
        action = str(proposal.get("action", "hold")).lower()

        checks: List[Dict[str, Any]] = []
        if action in {"buy", "sell"} and conflict >= 0.70:
            checks.append(
                self._check(
                    name="planner_conflict_index",
                    impact=-0.04,
                    message=f"Planner evidence is highly conflicting (conflict={conflict:.2f}).",
                    category="planner_trace",
                )
            )
        if action in {"buy", "sell"} and confidence >= 0.65 and conflict <= 0.45:
            checks.append(
                self._check(
                    name="planner_conflict_index",
                    impact=0.02,
                    message="Planner trace shows clear and coherent evidence.",
                    category="planner_trace",
                )
            )
        return checks

    def _build_pattern_key(self, features: Dict[str, Any], action: str) -> str:
        trend = str(features.get("trend", "down")).lower()
        sentiment_bucket = self._sentiment_bucket(features.get("news_sentiment", 0.0))
        return f"trend={trend}|sentiment={sentiment_bucket}|action={action}"

    def _sentiment_bucket(self, value: Any) -> str:
        numeric = self._safe_float(value, 0.0)
        if numeric >= 0.2:
            return "positive"
        if numeric <= -0.2:
            return "negative"
        return "neutral"

    def _check(
        self,
        name: str,
        impact: float,
        message: str,
        category: str,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "category": category,
            "impact": round(impact, 6),
            "message": message,
        }

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _sort_reviews(self, reviewed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reviewed.sort(
            key=lambda item: (
                self._safe_float(
                    item.get("critic_score"),
                    self._safe_float(item.get("confidence"), 0.0),
                ),
                self._safe_float(item.get("confidence"), 0.0),
            ),
            reverse=True,
        )
        return reviewed
