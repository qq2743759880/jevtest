from __future__ import annotations

import pytest
from pydantic import ValidationError

from jevtest.execution_contracts import (
    EvaluationEligibilityV1,
    HandoffRecordV1,
    RunReportV2,
    TriageEnvelopeV1,
)
from jevtest.models import TriageResult


def _result(category: str = "UNKNOWN") -> TriageResult:
    return TriageResult(
        category=category,
        action="HUMAN_TRIAGE" if category == "UNKNOWN" else "CHECK_ENVIRONMENT",
        priority=3,
        confidence=0.0 if category == "UNKNOWN" else 0.8,
        reason="Evidence is insufficient" if category == "UNKNOWN" else "R2: AP_reachable=false",
    )


def _envelope(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "triage-envelope-v1",
        "request_id": "req-001",
        "result_version": "result-v1",
        "created_at": "2026-09-27T12:00:00Z",
        "input_hash": "a" * 64,
        "input_projection_version": "facts-v1",
        "projection_hash": "c" * 64,
        "policy_id": "offline-demo",
        "policy_version": "1",
        "execution_mode": "offline",
        "run_purpose": "demo",
        "run_status": "result_ready",
        "result": _result(),
        "route_steps": [],
        "terminal_stage": "human",
        "attempts": [],
        "evidence_refs": [],
        "missing_evidence": [],
        "total_latency_ms": 1.0,
        "total_cost": 0.0,
        "cost_currency": None,
        "pricing_status": "mock",
        "requires_human_review": True,
        "review_status": "pending",
        "delivery_status": "pending",
        "resolution_status": "unknown",
        "local_handoff_id": None,
    }
    payload.update(overrides)
    return payload


def test_unknown_result_requires_review_and_handoff_is_not_falsely_persisted():
    envelope = TriageEnvelopeV1.model_validate(_envelope())
    assert envelope.requires_human_review is True
    assert envelope.delivery_status == "pending"
    assert envelope.local_handoff_id is None


def test_unknown_result_cannot_be_marked_no_review():
    with pytest.raises(ValidationError, match="UNKNOWN result requires human review"):
        TriageEnvelopeV1.model_validate(_envelope(requires_human_review=False))


def test_local_queue_requires_a_real_local_receipt_id():
    with pytest.raises(ValidationError, match="local_queued requires local_handoff_id"):
        TriageEnvelopeV1.model_validate(_envelope(delivery_status="local_queued"))


def test_unlabeled_run_report_keeps_accuracy_and_unknown_price_null():
    report = RunReportV2.model_validate(
        {
            "schema_version": "run-report-v2",
            "report_id": "report-001",
            "run_purpose": "real_exploratory",
            "dataset_kind": "real_unlabeled",
            "accuracy_mode": "triage_v2",
            "dataset_version": None,
            "sample_count": 2,
            "evaluated_count": 0,
            "accuracy": None,
            "accuracy_null_reason": "No independent labels",
            "eligibility": {
                "status": "ineligible",
                "criteria_version": "formal-v1",
                "reasons": ["Independent labels and arbitration are absent"],
                "evidence_manifest_id": None,
            },
            "attempts": 3,
            "failed_count": 0,
            "pricing_status": "unknown",
            "total_cost": None,
            "cost_currency": None,
            "route_counts": {"jev": 2, "llm": 1},
        }
    )
    assert report.accuracy is None
    assert report.total_cost is None
    assert report.pricing_status == "unknown"


def test_eligibility_requires_manifest_and_formal_purpose():
    with pytest.raises(ValidationError, match="eligible evaluation requires evidence_manifest_id"):
        EvaluationEligibilityV1(
            status="eligible", criteria_version="formal-v1", reasons=[], evidence_manifest_id=None
        )

    with pytest.raises(ValidationError, match="only formal_eval can be evaluation eligible"):
        RunReportV2.model_validate(
            {
                "schema_version": "run-report-v2",
                "report_id": "report-002",
                "run_purpose": "synthetic_eval",
                "dataset_kind": "synthetic",
                "accuracy_mode": "triage_v2",
                "sample_count": 1,
                "evaluated_count": 1,
                "accuracy": 1.0,
                "accuracy_null_reason": None,
                "eligibility": {
                    "status": "eligible",
                    "criteria_version": "formal-v1",
                    "reasons": [],
                    "evidence_manifest_id": "manifest-001",
                },
                "attempts": 1,
                "failed_count": 0,
                "pricing_status": "no_external_cost",
                "total_cost": 0.0,
                "cost_currency": None,
                "route_counts": {"rule": 1},
            }
        )


def test_legacy_accuracy_mode_is_limited_to_nonformal_synthetic_runs():
    base = {
        "schema_version": "run-report-v2",
        "report_id": "report-synthetic",
        "run_purpose": "synthetic_eval",
        "dataset_kind": "synthetic",
        "accuracy_mode": "legacy_exact_synthetic",
        "sample_count": 1,
        "evaluated_count": 1,
        "accuracy": 1.0,
        "accuracy_null_reason": None,
        "eligibility": {
            "status": "ineligible",
            "criteria_version": "formal-v1",
            "reasons": ["Synthetic runs are not formal evaluations"],
            "evidence_manifest_id": None,
        },
        "attempts": 1,
        "failed_count": 0,
        "pricing_status": "no_external_cost",
        "total_cost": 0.0,
        "cost_currency": None,
        "route_counts": {"rule": 1},
    }
    assert RunReportV2.model_validate(base).accuracy_mode == "legacy_exact_synthetic"

    with pytest.raises(ValidationError, match="limited to non-formal synthetic reports"):
        RunReportV2.model_validate({**base, "dataset_kind": "real_labeled"})


def test_local_handoff_status_requires_receipt_and_uses_scoped_key():
    base = {
        "schema_version": "handoff-record-v1",
        "handoff_id": "handoff-001",
        "request_id": "req-001",
        "result_version": "result-v1",
        "target_role": "test_owner",
        "delivery_status": "local_queued",
        "local_receipt_id": "receipt-001",
        "idempotency_scope": "local-demo",
        "idempotency_key": "key-001",
        "input_hash": "b" * 64,
        "created_at": "2026-09-27T12:00:00Z",
    }
    record = HandoffRecordV1.model_validate(base)
    assert record.delivery_status == "local_queued"
    assert record.local_receipt_id == "receipt-001"

    with pytest.raises(ValidationError, match="local_queued requires local_receipt_id"):
        HandoffRecordV1.model_validate({**base, "local_receipt_id": None})
