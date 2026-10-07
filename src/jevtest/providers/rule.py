"""Rule 臂的统一接口包装：Rule baseline v0 引擎接入 Provider ABC（PRD §5.1 条款 2）。

包装而非重写——判定逻辑单一事实源仍在 rule_engine + contracts/rule-baseline-v0.yaml，
T09 四臂集成时 Rule 臂与其他臂走完全相同的 `predict_validated` 入口，无特例。
"""

from __future__ import annotations

from jevtest.models import FailureState, TriageResult
from jevtest.providers.base import Provider
from jevtest.rule_engine import triage_by_rules


class RuleProvider(Provider):
    arm_name = "rule"
    is_mock = False

    def predict(self, state: FailureState) -> TriageResult:
        # triage_by_rules 自身已在边界做显式模型校验并抛 ValidationError；
        # predict_validated 会把 ValidationError 统一收敛为 InvalidProviderOutput。
        return triage_by_rules(state)
