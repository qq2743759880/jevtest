from __future__ import annotations

from typing import Any

import pytest

import jevtest.cascade as cascade_module
from jevtest.cascade import run_cascade
from jevtest.execution_contracts import RoutingPolicyV1
from jevtest.models import FailureState, TriageResult
from jevtest.providers.base import Provider
from jevtest.providers.jev_mock import JevMockProvider
from jevtest.providers.rule import RuleProvider


def _state(**overrides: Any) -> FailureState:
    payload: dict[str, Any] = {
        "case_id": "case-cascade-1",
        "result": "FAIL",
        "previous_runs": ["FAIL", "FAIL"],
        "device": "WA6638",
        "sw_version": "V200R022C01",
        "assertion": "unexpected test failure",
        "syslog_excerpt": "failure details",
        "env": {"AP_reachable": True, "AC_reachable": True},
        "retry": "passed",
        "provenance": {"dataset_version": "synthetic-v0", "synthetic": True},
    }
    payload.update(overrides)
    return FailureState.model_validate(payload)


def _policy(**overrides: Any) -> RoutingPolicyV1:
    payload: dict[str, Any] = {
        "schema_version": "routing-policy-v1",
        "policy_id": "offline-demo",
        "version": "1",
        "execution_mode": "offline",
        "run_purpose": "demo",
        "basis": "demo",
        "allowed_stages": ["rule", "jev", "llm"],
        "direct_output_rule_ids": ["R2_ENVIRONMENT"],
        "jev_high_threshold": 0.8,
        "jev_escalate_threshold": 0.5,
        "max_total_attempts": 4,
        "deadline_ms": 30_000,
    }
    payload.update(overrides)
    return RoutingPolicyV1.model_validate(payload)


class LLMStub(Provider):
    arm_name = "llm"
    is_mock = True

    def __init__(self, category: str = "UNKNOWN") -> None:
        self.calls_made = 0
        self.category = category

    def predict(self, state: FailureState) -> TriageResult:
        self.calls_made += 1
        return TriageResult(
            category=self.category,  # type: ignore[arg-type]
            action="HUMAN_TRIAGE" if self.category == "UNKNOWN" else "COLLECT_MORE_LOGS",
            priority=3,
            confidence=0.72,
            reason="offline LLM test stub",
        )


class LiveJevStub(Provider):
    arm_name = "jev"
    is_mock = False

    def __init__(self) -> None:
        self.calls_made = 0

    def predict(self, state: FailureState) -> TriageResult:
        self.calls_made += 1
        return TriageResult(
            category="UNKNOWN",
            action="HUMAN_TRIAGE",
            priority=3,
            confidence=0.2,
            reason="must never be called offline",
        )


class RetryingLLMStub(LLMStub):
    def predict_with_budget(self, state: FailureState, max_attempts: int) -> TriageResult:
        self.reset_attempts()
        call_limit = min(3, max_attempts)
        for index in range(call_limit):
            self.calls_made += 1
            self.record_attempt(
                status="failed" if index < call_limit - 1 else "succeeded",
                latency_ms=1.0,
            )
        return TriageResult(
            category="UNKNOWN",
            action="HUMAN_TRIAGE",
            priority=3,
            confidence=0.2,
            reason="retrying offline LLM stub",
        )


class ConfidenceJevStub(Provider):
    arm_name = "jev"
    is_mock = True

    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        self.calls_made = 0

    def predict(self, state: FailureState) -> TriageResult:
        self.calls_made += 1
        return TriageResult(
            category="FLAKY",
            action="RERUN",
            priority=2,
            confidence=self.confidence,
            reason="confidence band test stub",
        )


class FailingProvider(Provider):
    def __init__(self, arm_name: str, *, is_mock: bool) -> None:
        self.arm_name = arm_name
        self.is_mock = is_mock
        self.calls_made = 0

    def predict(self, state: FailureState) -> TriageResult:
        self.calls_made += 1
        raise RuntimeError(f"{self.arm_name} fixture failure")


def test_direct_r2_does_not_call_jev_or_llm() -> None:
    jev = JevMockProvider()
    llm = LLMStub()
    envelope = run_cascade(
        _state(env={"AP_reachable": False, "AC_reachable": True}),
        _policy(),
        {"rule": RuleProvider(), "jev": jev, "llm": llm},
        request_id="req-r2",
        result_version="result-v1",
    )

    assert envelope.result is not None and envelope.result.category == "ENVIRONMENT"
    assert envelope.terminal_stage == "rule"
    assert [step.stage for step in envelope.route_steps] == ["rule"]
    assert envelope.route_steps[0].reason_code == "direct_output:R2_ENVIRONMENT"
    assert jev.calls_made == llm.calls_made == 0
    assert len(envelope.attempts) == 1
    assert envelope.attempts[0].stage == "rule" and not envelope.attempts[0].provider_mock
    assert envelope.delivery_status == "not_requested"
    assert envelope.resolution_status == "unknown"


def test_r4_jev_unknown_llm_unknown_routes_to_human_with_mock_labels() -> None:
    jev = JevMockProvider()
    llm = LLMStub()
    envelope = run_cascade(
        _state(),
        _policy(),
        {"rule": RuleProvider(), "jev": jev, "llm": llm},
        request_id="req-escalate",
        result_version="result-v1",
        created_at="2026-09-27T00:00:00+00:00",
    )

    assert envelope.result is not None and envelope.result.category == "UNKNOWN"
    assert envelope.requires_human_review is True
    assert envelope.terminal_stage == "human"
    assert [step.stage for step in envelope.route_steps] == ["rule", "jev", "llm", "human"]
    assert envelope.route_steps[0].reason_code == "escalate_rule_result"
    assert envelope.route_steps[1].reason_code == "unknown_escalation_requested"
    assert envelope.route_steps[2].reason_code == "unknown_requires_human_review"
    assert jev.calls_made == llm.calls_made == 1
    assert [(item.stage, item.provider_mock) for item in envelope.attempts] == [
        ("rule", False),
        ("jev", True),
        ("llm", True),
    ]
    assert envelope.pricing_status == "mock"
    assert envelope.total_cost == 0
    assert envelope.delivery_status == "not_requested"
    assert envelope.local_handoff_id is None
    assert envelope.resolution_status == "unknown"


def test_global_budget_prevents_next_stage_request() -> None:
    jev = JevMockProvider()
    llm = LLMStub()
    envelope = run_cascade(
        _state(),
        _policy(max_total_attempts=1),
        {"rule": RuleProvider(), "jev": jev, "llm": llm},
        request_id="req-budget",
        result_version="result-v1",
    )

    assert len(envelope.attempts) == 1
    assert envelope.attempts[0].stage == "rule"
    assert jev.calls_made == llm.calls_made == 0
    assert [step.stage for step in envelope.route_steps] == ["rule", "human"]
    assert envelope.requires_human_review is True
    assert envelope.run_status == "degraded"


def test_all_provider_failures_are_recorded_and_end_in_human_review() -> None:
    providers = {
        "rule": FailingProvider("rule", is_mock=False),
        "jev": FailingProvider("jev", is_mock=True),
        "llm": FailingProvider("llm", is_mock=True),
    }

    envelope = run_cascade(
        _state(),
        _policy(),
        providers,
        request_id="req-all-providers-fail",
        result_version="result-v1",
    )

    assert envelope.run_status == "degraded"
    assert envelope.result is not None and envelope.result.category == "UNKNOWN"
    assert envelope.result.action == "HUMAN_TRIAGE"
    assert envelope.requires_human_review is True
    assert [step.stage for step in envelope.route_steps] == ["rule", "jev", "llm", "human"]
    assert all(step.status == "failed" for step in envelope.route_steps[:3])
    assert [(attempt.stage, attempt.status, attempt.invoked) for attempt in envelope.attempts] == [
        ("rule", "failed", True),
        ("jev", "failed", True),
        ("llm", "failed", True),
    ]


def test_remaining_global_budget_caps_retries_inside_each_stage() -> None:
    jev = JevMockProvider()
    llm = RetryingLLMStub()
    envelope = run_cascade(
        _state(),
        _policy(max_total_attempts=3),
        {"rule": RuleProvider(), "jev": jev, "llm": llm},
        request_id="req-inner-budget",
        result_version="result-v1",
    )

    # Rule and Jev each invoke once, leaving one request for LLM even though it can retry thrice.
    assert llm.calls_made == 1
    assert len(envelope.attempts) == 3
    assert sum(item.invoked for item in envelope.attempts) == 3
    assert envelope.route_steps[2].attempts == 1


@pytest.mark.parametrize(
    ("confidence", "expected_reason"),
    [
        (0.4, "low_confidence_escalation_requested"),
        (0.6, "mid_confidence_escalation_requested"),
    ],
)
def test_jev_escalation_threshold_records_low_and_middle_bands(
    confidence: float, expected_reason: str
) -> None:
    llm = LLMStub()
    envelope = run_cascade(
        _state(),
        _policy(),
        {
            "rule": RuleProvider(),
            "jev": ConfidenceJevStub(confidence),
            "llm": llm,
        },
        request_id=f"req-threshold-{confidence}",
        result_version="result-v1",
    )

    assert envelope.route_steps[1].reason_code == expected_reason
    assert llm.calls_made == 1


def test_r4_never_direct_outputs_even_if_policy_lists_it() -> None:
    jev = JevMockProvider()
    envelope = run_cascade(
        _state(),
        _policy(direct_output_rule_ids=["R4_FALLBACK"]),
        {"rule": RuleProvider(), "jev": jev},
        request_id="req-r4-never-direct",
        result_version="result-v1",
    )

    assert envelope.route_steps[0].reason_code == "escalate_rule_result"
    assert jev.calls_made == 1
    assert envelope.terminal_stage == "human"


def test_direct_rule_id_must_match_the_canonical_fired_rule() -> None:
    jev = JevMockProvider()
    envelope = run_cascade(
        _state(env={"AP_reachable": False}),
        _policy(direct_output_rule_ids=["R2_NOT_THE_BASELINE_ID"]),
        {"rule": RuleProvider(), "jev": jev},
        request_id="req-wrong-rule-id",
        result_version="result-v1",
    )

    assert envelope.route_steps[0].reason_code == "escalate_rule_result"
    assert jev.calls_made == 1


def test_deadline_skips_later_stages_after_current_sync_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_time = [0.0]

    class SlowRuleProvider(RuleProvider):
        def predict(self, state: FailureState) -> TriageResult:
            result = super().predict(state)
            fake_time[0] = 0.02
            return result

    jev = JevMockProvider()
    llm = LLMStub()
    monkeypatch.setattr(cascade_module, "perf_counter", lambda: fake_time[0])
    envelope = run_cascade(
        _state(),
        _policy(deadline_ms=10),
        {"rule": SlowRuleProvider(), "jev": jev, "llm": llm},
        request_id="req-deadline",
        result_version="result-v1",
    )

    assert [step.stage for step in envelope.route_steps] == ["rule", "human"]
    assert envelope.route_steps[-1].reason_code == "deadline_exceeded"
    assert jev.calls_made == llm.calls_made == 0
    assert envelope.requires_human_review is True


def test_offline_mode_rejects_live_jev_before_any_provider_call() -> None:
    jev = LiveJevStub()
    llm = LLMStub()
    with pytest.raises(ValueError, match="rejects non-mock jev"):
        run_cascade(
            _state(env={"AP_reachable": False}),
            _policy(),
            {"rule": RuleProvider(), "jev": jev, "llm": llm},
            request_id="req-offline-guard",
            result_version="result-v1",
        )
    assert jev.calls_made == llm.calls_made == 0


def test_non_offline_policy_is_rejected_before_provider_call() -> None:
    llm = LLMStub()
    with pytest.raises(ValueError, match="runtime authorization"):
        run_cascade(
            _state(env={"AP_reachable": False}),
            _policy(execution_mode="live"),
            {"rule": RuleProvider(), "llm": llm},
            request_id="req-live-policy",
            result_version="result-v1",
        )
    assert llm.calls_made == 0


def test_llm_non_unknown_remains_a_human_review_suggestion() -> None:
    llm = LLMStub(category="FLAKY")
    envelope = run_cascade(
        _state(),
        _policy(allowed_stages=["rule", "jev", "llm"]),
        {"rule": RuleProvider(), "jev": JevMockProvider(), "llm": llm},
        request_id="req-suggestion",
        result_version="result-v1",
    )

    assert envelope.result is not None and envelope.result.category == "FLAKY"
    assert envelope.requires_human_review is True
    assert envelope.terminal_stage == "human"
    assert envelope.route_steps[-2].reason_code == "suggestion_requires_human_review"
    assert envelope.delivery_status == "not_requested"
    assert envelope.resolution_status == "unknown"
