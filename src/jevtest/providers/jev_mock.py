"""Jev mock provider：确定性启发式打标，仅为 M2 管道验证，不是真分诊（PRD §11 S1 --mock 模式）。

确定性纪律：同输入 → 同输出。禁止 random / 时间 / 环境读取；
模拟延迟与费用只做纯记录，不真 sleep、不计费。
"""

from __future__ import annotations

from typing import Any, Literal

from jevtest.models import FailureState, TriageResult
from jevtest.providers.base import Provider

FailMode = Literal[None, "missing_reason", "bad_enum"]

# confidence=Probs[Value]（正确概率）语义；confidence_sharpness=分布尖锐度，单独承载（PRD §10.2）。
_CONFIDENCE_HIT = 0.9
_CONFIDENCE_UNKNOWN = 0.3
_SHARPNESS_HIT = 0.8
_SHARPNESS_UNKNOWN = 0.3

_REASON_TAG = "jev-mock heuristic"


def _flip_count(previous_runs: list[str]) -> int:
    """PASS<->FAIL 翻转次数（与 rule_engine 同口径：相邻运行结果不同的位置数）。"""
    return sum(1 for before, after in zip(previous_runs, previous_runs[1:]) if before != after)


def _first_false_env_key(state: FailureState) -> str | None:
    for key, value in state.env.items():
        if value is False:
            return key
    return None


class JevMockProvider(Provider):
    """确定性 Jev mock。启发式顺序：env → flaky → timeout → UNKNOWN，先命中先输出。"""

    arm_name = "jev"
    is_mock = True

    def __init__(
        self,
        simulated_latency_ms: float = 0.0,
        simulated_cost: float = 0.0,
        fail_mode: FailMode = None,
    ) -> None:
        # 纯记录字段：predict 不 sleep、不产生真实费用，仅累计供实验计量管道联调。
        self.simulated_latency_ms = float(simulated_latency_ms)
        self.simulated_cost = float(simulated_cost)
        self.calls_made = 0
        self.accumulated_latency_ms = 0.0
        self.accumulated_cost = 0.0
        if fail_mode not in (None, "missing_reason", "bad_enum"):
            raise ValueError(f"unknown fail_mode: {fail_mode!r}")
        self.fail_mode: FailMode = fail_mode

    def predict(self, state: FailureState) -> TriageResult:
        self.calls_made += 1
        self.accumulated_latency_ms += self.simulated_latency_ms
        self.accumulated_cost += self.simulated_cost

        if self.fail_mode is not None:
            return self._build_fail_mode_output(state)

        false_key = _first_false_env_key(state)
        if false_key is not None:
            return self._hit(
                category="ENVIRONMENT",
                action="CHECK_ENVIRONMENT",
                priority=3,
                reason=f"{_REASON_TAG}: env probe {false_key}=false indicates environment failure",
            )
        if _flip_count(state.previous_runs) >= 2 and state.retry == "not_executed":
            flips = _flip_count(state.previous_runs)
            return self._hit(
                category="FLAKY",
                action="RERUN",
                priority=1,
                reason=(
                    f"{_REASON_TAG}: {flips} PASS<->FAIL flips with retry=not_executed "
                    "suggests flaky test"
                ),
            )
        if "timeout" in state.assertion.lower():
            return self._hit(
                category="PRODUCT_DEFECT",
                action="ASSIGN_PRODUCT_DEVELOPER",
                priority=4,
                reason=f"{_REASON_TAG}: assertion {state.assertion[:80]!r} matches timeout defect signature",
            )
        return self._hit(
            category="UNKNOWN",
            action="HUMAN_TRIAGE",
            priority=3,
            reason=f"{_REASON_TAG}: no heuristic matched for assertion {state.assertion[:80]!r}",
        )

    def _hit(
        self, category: str, action: str, priority: int, reason: str
    ) -> TriageResult:
        # confidence 与 confidence_sharpness 分字段承载：前者是 Probs[Value]（正确概率），
        # 后者是分布尖锐度（Jev Confidence 原值），禁止混用（PRD §10.2）。
        if category == "UNKNOWN":
            confidence, sharpness = _CONFIDENCE_UNKNOWN, _SHARPNESS_UNKNOWN
        else:
            confidence, sharpness = _CONFIDENCE_HIT, _SHARPNESS_HIT
        probs: dict[str, Any] = {category: confidence, "other": round(1.0 - confidence, 2)}
        return TriageResult(
            category=category,  # type: ignore[arg-type]
            action=action,  # type: ignore[arg-type]
            priority=priority,
            confidence=confidence,
            reason=reason[:200],
            probs=probs,
            confidence_sharpness=sharpness,
        )

    def _build_fail_mode_output(self, state: FailureState) -> TriageResult:
        """故意产出违反契约的输出（绕过模型校验），供测试 InvalidProviderOutput / sanitize 兜底路径。"""
        base = self._hit(
            category="ENVIRONMENT",
            action="CHECK_ENVIRONMENT",
            priority=3,
            reason=f"{_REASON_TAG}: injected fail_mode={self.fail_mode}",
        )
        if self.fail_mode == "missing_reason":
            return base.model_copy(update={"reason": ""})
        # bad_enum：category 落在冻结枚举之外
        return base.model_copy(update={"category": "NOT_A_CATEGORY"})
