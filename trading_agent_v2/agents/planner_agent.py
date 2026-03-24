# trading_agent_v2/agents/planner_agent.py

from typing import List, Dict, Any


class PlannerAgent:
    def _extract_view_fields(self, view):
        if isinstance(view, dict):
            return (
                view.get("bias", "neutral"),
                float(view.get("confidence", 0.0)),
            )

        return (
            getattr(view, "bias", "neutral"),
            float(getattr(view, "confidence", 0.0)),
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
        bullish_score = 0.0
        bearish_score = 0.0

        for view in analyst_views:
            bias, confidence = self._extract_view_fields(view)

            if bias == "bullish":
                bullish_score += confidence
            elif bias == "bearish":
                bearish_score += confidence

        net_score = bullish_score - bearish_score

        proposals = []

        proposals.append({
            "proposal_id": "defensive",
            "symbol": symbol,
            "action": "hold",
            "size_pct": 0.0,
            "confidence": 0.55,
            "thesis": "Signals are mixed, preserve capital.",
            "style": "defensive",
        })

        if net_score > 0.2:
            proposals.append({
                "proposal_id": "base",
                "symbol": symbol,
                "action": "buy",
                "size_pct": 0.08,
                "confidence": min(0.75, 0.55 + net_score * 0.1),
                "thesis": "Bullish analyst alignment supports a modest long position.",
                "style": "base",
            })
            proposals.append({
                "proposal_id": "aggressive",
                "symbol": symbol,
                "action": "buy",
                "size_pct": 0.15,
                "confidence": min(0.72, 0.52 + net_score * 0.08),
                "thesis": "Strong alignment may justify a larger position.",
                "style": "aggressive",
            })
        elif net_score < -0.2:
            proposals.append({
                "proposal_id": "base",
                "symbol": symbol,
                "action": "sell",
                "size_pct": 0.08,
                "confidence": min(0.75, 0.55 + abs(net_score) * 0.1),
                "thesis": "Bearish analyst alignment supports reducing exposure.",
                "style": "base",
            })
            proposals.append({
                "proposal_id": "aggressive",
                "symbol": symbol,
                "action": "sell",
                "size_pct": 0.15,
                "confidence": min(0.72, 0.52 + abs(net_score) * 0.08),
                "thesis": "Stronger bearish alignment supports more aggressive de-risking.",
                "style": "aggressive",
            })
        else:
            proposals.append({
                "proposal_id": "base",
                "symbol": symbol,
                "action": "hold",
                "size_pct": 0.0,
                "confidence": 0.58,
                "thesis": "No clear directional edge.",
                "style": "base",
            })
            proposals.append({
                "proposal_id": "aggressive",
                "symbol": symbol,
                "action": "hold",
                "size_pct": 0.0,
                "confidence": 0.50,
                "thesis": "High uncertainty; avoid forced trades.",
                "style": "aggressive",
            })

        pattern_stats = {}
        if strategy_memory and isinstance(strategy_memory, dict):
            pattern_stats = strategy_memory.get("pattern_stats", {}) or {}

        proposals = self._apply_memory_adjustments(
            proposals=proposals,
            features=features,
            pattern_stats=pattern_stats,
            similar_cases=similar_cases or [],
        )
        return proposals

    def _apply_memory_adjustments(
        self,
        proposals: List[Dict[str, Any]],
        features: Dict[str, Any],
        pattern_stats: Dict[str, Any],
        similar_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        adjusted: List[Dict[str, Any]] = []
        for proposal in proposals:
            p = proposal.copy()
            action = str(p.get("action", "hold")).lower()
            if action not in {"buy", "sell"}:
                p["memory_adjustment"] = {"applied": False, "reason": "non_directional"}
                adjusted.append(p)
                continue

            confidence = float(p.get("confidence", 0.5))
            size_pct = float(p.get("size_pct", 0.0))
            notes: List[str] = []

            pattern_adj = self._pattern_adjustment(
                action=action,
                features=features,
                pattern_stats=pattern_stats,
            )
            if pattern_adj["applied"]:
                confidence += float(pattern_adj["confidence_delta"])
                size_pct *= float(pattern_adj["size_multiplier"])
                notes.extend(pattern_adj["notes"])

            similar_adj = self._similar_case_adjustment(
                action=action,
                similar_cases=similar_cases,
            )
            if similar_adj["applied"]:
                confidence += float(similar_adj["confidence_delta"])
                size_pct *= float(similar_adj["size_multiplier"])
                notes.extend(similar_adj["notes"])

            p["confidence"] = max(0.0, min(1.0, confidence))
            p["size_pct"] = max(0.0, min(0.30, size_pct))
            p["memory_adjustment"] = {
                "applied": bool(pattern_adj["applied"] or similar_adj["applied"]),
                "notes": notes,
                "pattern": pattern_adj,
                "similar_cases": similar_adj,
            }
            adjusted.append(p)
        return adjusted

    def _pattern_adjustment(
        self,
        action: str,
        features: Dict[str, Any],
        pattern_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        trend = str(features.get("trend", "down")).lower()
        sentiment_bucket = self._sentiment_bucket(features.get("news_sentiment", 0.0))
        key = f"trend={trend}|sentiment={sentiment_bucket}|action={action}"
        stats = pattern_stats.get(key) if isinstance(pattern_stats, dict) else None
        if not isinstance(stats, dict):
            return {
                "applied": False,
                "pattern_key": key,
                "notes": [],
                "confidence_delta": 0.0,
                "size_multiplier": 1.0,
            }

        count = int(stats.get("count", 0))
        wins = int(stats.get("wins", 0))
        losses = int(stats.get("losses", 0))
        win_rate = (wins / count) if count > 0 else 0.0

        confidence_delta = 0.0
        size_multiplier = 1.0
        notes: List[str] = []

        if count >= 3 and losses > wins:
            confidence_delta -= 0.10
            size_multiplier *= 0.75
            notes.append(f"pattern_underperforming({key}, win_rate={win_rate:.2f})")
        elif count >= 3 and wins > losses and win_rate >= 0.60:
            confidence_delta += 0.05
            size_multiplier *= 1.10
            notes.append(f"pattern_supportive({key}, win_rate={win_rate:.2f})")

        return {
            "applied": bool(notes),
            "pattern_key": key,
            "notes": notes,
            "confidence_delta": confidence_delta,
            "size_multiplier": size_multiplier,
            "sample_count": count,
            "win_rate": round(win_rate, 4),
        }

    def _similar_case_adjustment(
        self,
        action: str,
        similar_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        matched = [
            case for case in similar_cases
            if str(case.get("action", "")).lower() == action
        ]
        if not matched:
            return {
                "applied": False,
                "notes": [],
                "confidence_delta": 0.0,
                "size_multiplier": 1.0,
                "matched_count": 0,
            }

        losses = sum(1 for case in matched if str(case.get("outcome", "")).lower() == "loss")
        wins = sum(1 for case in matched if str(case.get("outcome", "")).lower() == "win")
        ratio = losses / len(matched)

        confidence_delta = 0.0
        size_multiplier = 1.0
        notes: List[str] = []

        if len(matched) >= 2 and ratio >= 0.5:
            confidence_delta -= 0.08
            size_multiplier *= 0.8
            notes.append(f"similar_cases_lossy(loss_ratio={ratio:.2f})")
        elif len(matched) >= 2 and wins > losses:
            confidence_delta += 0.04
            size_multiplier *= 1.05
            notes.append("similar_cases_supportive")

        return {
            "applied": bool(notes),
            "notes": notes,
            "confidence_delta": confidence_delta,
            "size_multiplier": size_multiplier,
            "matched_count": len(matched),
        }

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
