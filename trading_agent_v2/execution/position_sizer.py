from __future__ import annotations

from dataclasses import replace

from trading_agent_v2.schemas import FinalDecision


class PositionSizer:
    """
    Final guardrail before validation/execution.
    """

    def __init__(
        self,
        min_size_pct: float = 0.01,
        max_size_pct: float = 0.45,
        enforce_min_size: bool = False,
    ):
        self.min_size_pct = min_size_pct
        self.max_size_pct = max_size_pct
        self.enforce_min_size = enforce_min_size

    def apply(self, decision: FinalDecision) -> FinalDecision:
        action = (decision.action or "").lower().strip()
        if action not in {"buy", "sell"}:
            return decision

        original_size = max(0.0, float(decision.size_pct or 0.0))
        metadata = dict(decision.metadata or {})
        metadata["position_sizer_original_size_pct"] = original_size

        if original_size <= 0.0:
            return replace(
                decision,
                action="hold",
                reason=f"{decision.reason} Position size is non-positive after risk adjustment.",
                size_pct=0.0,
                stop_loss_pct=None,
                take_profit_pct=None,
                metadata=metadata,
            )

        adjusted_size = min(original_size, self.max_size_pct)

        if 0.0 < adjusted_size < self.min_size_pct:
            if self.enforce_min_size:
                adjusted_size = self.min_size_pct
                metadata["position_sizer_enforced_min_size"] = True
            else:
                metadata["position_sizer_skip_reason"] = "below_min_size"
                return replace(
                    decision,
                    action="hold",
                    reason=(
                        f"{decision.reason} Position size ({adjusted_size:.4f}) is below "
                        f"minimum executable size ({self.min_size_pct:.4f})."
                    ),
                    size_pct=0.0,
                    stop_loss_pct=None,
                    take_profit_pct=None,
                    metadata=metadata,
                )

        metadata["position_sizer_final_size_pct"] = adjusted_size
        return replace(decision, size_pct=adjusted_size, metadata=metadata)
