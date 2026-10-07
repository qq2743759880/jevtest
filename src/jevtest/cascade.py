"""Explicit-policy, offline-first cascade over the existing provider adapters."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from time import perf_counter

from jevtest.execution_contracts import (
    AttemptRecordV1,
    RouteStepV1,
    RoutingPolicyV1,
    TriageEnvelopeV1,
)
from jevtest.input_projection import (
    PROJECTION_VERSION,
    evidence_refs_for_facts,
    project_failure_state,
    projection_sha256,
)
from jevtest.metrics import MeasureRunner
from jevtest.models import FailureState, TriageResult
from jevtest.providers.base import Provider
from jevtest.runtime_authorization import LiveRuntimeAuthorization

_RULE_LABEL = re.compile(r"^(R[1-4])(?:\b|_)")
_RULE_ID_BY_LABEL = {
    "R1": "R1_FLAKY",
    "R2": "R2_ENVIRONMENT",
    "R3": "R3_ERROR_CODE",
}


def _input_hash(state: FailureState) -> str:
    payload = json.dumps(
        state.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _allowed_rule_id(reason: str, policy: RoutingPolicyV1) -> str | None:
    match = _RULE_LABEL.match(reason.strip())
    if not match:
        return None
    label = match.group(1)
    # R4 is the no-evidence fallback and must always escalate. For the frozen v0 baseline,
    # require the exact canonical rule id so an unrelated R2_* policy entry cannot authorize R2.
    canonical_id = _RULE_ID_BY_LABEL.get(label)
    if canonical_id is not None and canonical_id in policy.direct_output_rule_ids:
        return canonical_id
    return None


def _unknown_result(reason: str) -> TriageResult:
    return TriageResult(
        category="UNKNOWN",
        action="HUMAN_TRIAGE",
        priority=3,
        confidence=0.0,
        reason=reason[:200],
    )


def _attempt_totals(attempts: list[AttemptRecordV1]) -> tuple[float, float | None, str | None, str]:
    invoked = [attempt for attempt in attempts if attempt.invoked]
    latency_ms = sum(attempt.latency_ms for attempt in invoked)
    statuses = {attempt.pricing_status for attempt in invoked}
    known_currencies = {
        attempt.cost_currency
        for attempt in invoked
        if attempt.pricing_status == "known" and attempt.cost_currency
    }
    if "unknown" in statuses or len(known_currencies) > 1:
        return latency_ms, None, None, "unknown"
    if "known" in statuses:
        if len(known_currencies) != 1 or any(
            attempt.pricing_status == "known" and attempt.cost is None for attempt in invoked
        ):
            return latency_ms, None, None, "unknown"
        total_cost = sum(
            attempt.cost or 0.0
            for attempt in invoked
            if attempt.pricing_status in {"known", "no_external_cost", "mock"}
        )
        return latency_ms, round(total_cost, 12), next(iter(known_currencies)), "known"
    if "mock" in statuses:
        return latency_ms, 0.0, None, "mock"
    return latency_ms, 0.0, None, "no_external_cost"


def run_cascade(
    state: FailureState,
    policy: RoutingPolicyV1,
    providers: Mapping[str, Provider],
    *,
    request_id: str,
    result_version: str,
    created_at: str | None = None,
    runtime_authorization: LiveRuntimeAuthorization | None = None,
) -> TriageEnvelopeV1:
    """Run the Rule → Jev → LLM decision path under a cross-stage request cap.

    Offline execution requires mock Jev/LLM providers. Live execution is only
    accepted with a separate bounded runtime authorization. The envelope records
    suggestions and actual attempts but does not deliver, resolve, or externally
    apply a result. The deadline is checked between synchronous stages; provider
    client timeouts bound in-flight calls.
    """
    if policy.execution_mode == "offline":
        if runtime_authorization is not None:
            raise ValueError("offline cascade does not accept live runtime authorization")
        for stage in ("jev", "llm"):
            provider = providers.get(stage)
            if provider is not None and not provider.is_mock:
                raise ValueError(f"offline cascade rejects non-mock {stage} provider")
    else:
        if runtime_authorization is None:
            raise ValueError("live cascade requires explicit runtime authorization")
        if (
            policy.run_purpose != "advisory"
            or policy.jev_high_threshold is not None
            or policy.direct_output_rule_ids
            or "human" not in policy.allowed_stages
        ):
            raise ValueError("live-advisory requires advisory purpose and mandatory human review")
        for stage in ("jev", "llm"):
            provider = providers.get(stage)
            if stage not in runtime_authorization.provider_allowlist:
                raise ValueError(f"live cascade provider {stage!r} is not authorized")
            if provider is None or provider.is_mock:
                raise ValueError(f"live cascade requires a real {stage} provider")
            if provider.arm_name != stage:
                raise ValueError(
                    f"provider mapping key {stage!r} does not match arm_name {provider.arm_name!r}"
                )
    for stage, provider in providers.items():
        if stage in {"rule", "jev", "llm"} and provider.arm_name != stage:
            raise ValueError(
                f"provider mapping key {stage!r} does not match arm_name {provider.arm_name!r}"
            )

    deadline_at = perf_counter() + policy.deadline_ms / 1000.0
    projection = project_failure_state(state)
    projection_hash = projection_sha256(projection)
    input_hash = _input_hash(state)
    attempts: list[AttemptRecordV1] = []
    route_steps: list[RouteStepV1] = []
    current_result: TriageResult | None = None
    terminal_stage: str | None = None
    requires_human_review = False
    degraded = False
    deadline_exceeded = False
    def has_deadline_expired() -> bool:
        return perf_counter() >= deadline_at

    def remaining_budget() -> int:
        return policy.max_total_attempts - sum(1 for item in attempts if item.invoked)

    def remaining_provider_budget(stage: str) -> int:
        if runtime_authorization is None or stage not in {"jev", "llm"}:
            return remaining_budget()
        used = sum(1 for item in attempts if item.stage == stage and item.invoked)
        return min(remaining_budget(), runtime_authorization.budget_for(stage) - used)

    def visit(stage: str) -> tuple[TriageResult | None, bool]:
        nonlocal degraded, deadline_exceeded
        if has_deadline_expired():
            deadline_exceeded = True
            degraded = True
            return None, False
        provider = providers.get(stage)
        if provider is None:
            degraded = True
            return None, False
        budget = remaining_provider_budget(stage)
        if budget < 1:
            degraded = True
            return None, False
        runner = MeasureRunner(provider)
        result: TriageResult | None = None
        failed = False
        try:
            result, sample = runner.run_one(state, max_attempts=budget)
            failed = sample.parse_failure or bool(runner.last_attempts) and all(
                attempt.status == "failed" for attempt in runner.last_attempts
            )
        except Exception:
            failed = True
            result = None
        stage_attempts = list(runner.last_attempts)
        attempts.extend(stage_attempts)
        route_steps.append(
            RouteStepV1(
                stage=stage,  # type: ignore[arg-type]
                status="failed" if failed else "completed",
                attempts=sum(1 for item in stage_attempts if item.invoked),
                provider_mock=provider.is_mock,
                reason_code="provider_failure" if failed else None,
            )
        )
        if failed:
            degraded = True
        return result, failed

    if "rule" in policy.allowed_stages:
        rule_result, rule_failed = visit("rule")
        if rule_result is not None:
            current_result = rule_result
            rule_id = _allowed_rule_id(rule_result.reason, policy)
            if (
                not rule_failed
                and rule_result.category != "UNKNOWN"
                and rule_id is not None
            ):
                route_steps[-1] = route_steps[-1].model_copy(
                    update={"reason_code": f"direct_output:{rule_id}"}
                )
                terminal_stage = "rule"
                return _envelope(
                    state,
                    policy,
                    request_id,
                    result_version,
                    created_at,
                    input_hash,
                    projection_hash,
                    current_result,
                    route_steps,
                    attempts,
                    terminal_stage,
                    False,
                    degraded,
                )
            if route_steps:
                route_steps[-1] = route_steps[-1].model_copy(
                    update={"reason_code": "escalate_rule_result"}
                )

    # Rule output is terminal only for a policy-authorized R1/R2/R3. R4 and unknowns escalate.
    if "jev" in policy.allowed_stages and remaining_budget() > 0:
        jev_result, jev_failed = visit("jev")
        if jev_result is not None:
            current_result = jev_result
        high_threshold = policy.jev_high_threshold
        jev_is_high = (
            not jev_failed
            and jev_result is not None
            and jev_result.category != "UNKNOWN"
            and high_threshold is not None
            and jev_result.confidence >= high_threshold
        )
        if jev_is_high:
            route_steps[-1] = route_steps[-1].model_copy(
                update={"reason_code": "high_confidence_result"}
            )
            terminal_stage = "jev"
            return _envelope(
                state,
                policy,
                request_id,
                result_version,
                created_at,
                input_hash,
                projection_hash,
                current_result,
                route_steps,
                attempts,
                terminal_stage,
                False,
                degraded,
            )
        if route_steps and route_steps[-1].stage == "jev":
            if jev_result is None or jev_result.category == "UNKNOWN":
                escalation_reason = "unknown_escalation_requested"
            # The lower threshold distinguishes low from middle confidence for reporting;
            # both bands escalate to LLM when allowed and within budget.
            elif policy.jev_escalate_threshold is not None and (
                jev_result.confidence < policy.jev_escalate_threshold
            ):
                escalation_reason = "low_confidence_escalation_requested"
            elif policy.jev_escalate_threshold is not None:
                escalation_reason = "mid_confidence_escalation_requested"
            else:
                escalation_reason = "confidence_escalation_requested"
            route_steps[-1] = route_steps[-1].model_copy(
                update={"reason_code": escalation_reason}
            )
    elif "jev" in policy.allowed_stages and remaining_budget() == 0:
        degraded = True

    if "llm" in policy.allowed_stages and remaining_budget() > 0:
        llm_result, llm_failed = visit("llm")
        if llm_result is not None and not llm_failed:
            current_result = llm_result
        if route_steps and route_steps[-1].stage == "llm":
            route_steps[-1] = route_steps[-1].model_copy(
                update={
                    "reason_code": (
                        "provider_failure_human_review"
                        if llm_failed or llm_result is None
                        else "unknown_requires_human_review"
                        if llm_result.category == "UNKNOWN"
                        else "suggestion_requires_human_review"
                    )
                }
            )
    elif "llm" in policy.allowed_stages:
        degraded = True

    # No handoff is created here. A human terminal step is a routing decision only.
    requires_human_review = True
    if current_result is None:
        current_result = _unknown_result("Cascade produced no valid result; human review required")
    if current_result.category == "UNKNOWN" and current_result.action != "HUMAN_TRIAGE":
        current_result = current_result.model_copy(update={"action": "HUMAN_TRIAGE"})
    route_steps.append(
        RouteStepV1(
            stage="human",
            status="completed",
            attempts=0,
            provider_mock=False,
            reason_code="deadline_exceeded" if deadline_exceeded else "review_required",
        )
    )
    terminal_stage = "human"
    return _envelope(
        state,
        policy,
        request_id,
        result_version,
        created_at,
        input_hash,
        projection_hash,
        current_result,
        route_steps,
        attempts,
        terminal_stage,
        requires_human_review,
        degraded,
    )


def _envelope(
    state: FailureState,
    policy: RoutingPolicyV1,
    request_id: str,
    result_version: str,
    created_at: str | None,
    input_hash: str,
    projection_hash: str,
    result: TriageResult,
    route_steps: list[RouteStepV1],
    attempts: list[AttemptRecordV1],
    terminal_stage: str | None,
    requires_human_review: bool,
    degraded: bool,
) -> TriageEnvelopeV1:
    total_latency, total_cost, cost_currency, pricing_status = _attempt_totals(attempts)
    facts = project_failure_state(state)
    missing_evidence = (
        [
            "设备侧的实际关联状态（当前没有独立结构化字段；请从源日志核实）",
            "失败断言前后的带时间事件序列（如当前摘录未覆盖，请补充）",
            "完整诊断日志或日志覆盖范围（当前只记录为 syslog_excerpt 摘录）",
        ]
        if result.category == "UNKNOWN" or result.action == "HUMAN_TRIAGE"
        else []
    )
    return TriageEnvelopeV1(
        schema_version="triage-envelope-v1",
        request_id=request_id,
        result_version=result_version,
        created_at=created_at or _now_iso(),
        input_hash=input_hash,
        input_projection_version=PROJECTION_VERSION,
        projection_hash=projection_hash,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        execution_mode=policy.execution_mode,
        run_purpose=policy.run_purpose,
        run_status="degraded" if degraded else "result_ready",
        result=result,
        route_steps=route_steps,
        terminal_stage=terminal_stage,  # type: ignore[arg-type]
        attempts=attempts,
        evidence_refs=evidence_refs_for_facts(
            facts, request_id=request_id, result_version=result_version
        ),
        missing_evidence=missing_evidence,
        total_latency_ms=total_latency,
        total_cost=total_cost,
        cost_currency=cost_currency,
        pricing_status=pricing_status,  # type: ignore[arg-type]
        requires_human_review=requires_human_review,
        review_status="pending" if requires_human_review else "not_requested",
        delivery_status="not_requested",
        resolution_status="unknown",
    )
