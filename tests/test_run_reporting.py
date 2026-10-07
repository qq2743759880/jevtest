from __future__ import annotations

import pytest

from jevtest.execution_contracts import (
    AttemptRecordV1,
    TriageEnvelopeV1,
)
from jevtest.models import TriageResult
from jevtest.run_reporting import EvaluationEvidence, build_run_report


def _result(category: str = "PRODUCT_DEFECT") -> TriageResult:
    return TriageResult(
        category=category,
        action="HUMAN_TRIAGE" if category == "UNKNOWN" else "ASSIGN_PRODUCT_DEVELOPER",
        priority=3,
        confidence=0.7,
        reason="offline test fixture",
    )


def _envelope(
    request_id: str = "req-1",
    *,
    category: str | None = "PRODUCT_DEFECT",
    purpose: str = "demo",
    status: str = "result_ready",
    pricing_status: str = "no_external_cost",
    total_cost: float | None = 0.0,
    cost_currency: str | None = None,
    attempts: list[AttemptRecordV1] | None = None,
    terminal_stage: str | None = "rule",
    requires_human_review: bool = False,
) -> TriageEnvelopeV1:
    effective_attempts = attempts
    if effective_attempts is None and purpose == "formal_eval":
        effective_attempts = [
            AttemptRecordV1(
                stage="rule",
                attempt_index=1,
                invoked=True,
                provider_mock=False,
                provider_id="jevtest.rule",
                model_id="rule-baseline-v0",
                status="succeeded",
                latency_ms=1,
                cost=0.0,
                pricing_status="no_external_cost",
            )
        ]
    return TriageEnvelopeV1(
        schema_version="triage-envelope-v1",
        request_id=request_id,
        result_version="result-v1",
        created_at="2026-09-27T00:00:00Z",
        input_hash="a" * 64,
        input_projection_version="projection-v1",
        projection_hash="b" * 64,
        policy_id="offline-demo",
        policy_version="1",
        execution_mode="offline",
        run_purpose=purpose,
        run_status=status,
        result=None if category is None else _result(category),
        route_steps=[],
        terminal_stage=terminal_stage,
        attempts=effective_attempts or [],
        evidence_refs=[],
        missing_evidence=[],
        total_latency_ms=1.0,
        total_cost=total_cost,
        cost_currency=cost_currency,
        pricing_status=pricing_status,
        requires_human_review=requires_human_review or category == "UNKNOWN",
        review_status="pending"
        if requires_human_review or category == "UNKNOWN"
        else "not_requested",
        delivery_status="not_requested",
        resolution_status="unknown",
    )


def test_unlabeled_real_report_has_null_accuracy_and_is_not_formal_eligible():
    report = build_run_report(
        [
            _envelope("r1", purpose="advisory"),
            _envelope(
                "r2",
                category="UNKNOWN",
                purpose="advisory",
                requires_human_review=True,
            ),
        ],
        report_id="report-unlabeled",
        run_purpose="advisory",
        dataset_kind="real_unlabeled",
        dataset_version="capture-v1",
    )

    assert report.accuracy is None
    assert "labels were not supplied" in report.accuracy_null_reason
    assert report.eligibility.status == "not_requested"
    assert report.route_counts == {"human": 1, "rule": 1}


def test_synthetic_demo_without_labels_is_inference_not_evaluation():
    report = build_run_report(
        [_envelope("demo-r1")],
        report_id="report-synthetic-demo",
        run_purpose="demo",
        dataset_kind="synthetic",
        dataset_version="synthetic-v0",
    )

    assert report.accuracy is None
    assert "labels were not supplied" in (report.accuracy_null_reason or "")
    assert report.eligibility.status == "not_requested"


def test_synthetic_legacy_accuracy_can_score_expected_unknown_but_v2_cannot():
    samples = [_envelope("r1", category="UNKNOWN", purpose="synthetic_eval")]
    legacy = build_run_report(
        samples,
        report_id="report-legacy",
        run_purpose="synthetic_eval",
        dataset_kind="synthetic",
        dataset_version="synthetic-v0",
        expected_labels={"r1": "UNKNOWN"},
        accuracy_mode="legacy_exact_synthetic",
    )
    current = build_run_report(
        samples,
        report_id="report-v2",
        run_purpose="synthetic_eval",
        dataset_kind="synthetic",
        dataset_version="synthetic-v0",
        expected_labels={"r1": "UNKNOWN"},
    )

    assert legacy.accuracy_mode == "legacy_exact_synthetic"
    assert legacy.accuracy == 1.0
    assert current.accuracy_mode == "triage_v2"
    assert current.accuracy == 0.0


def test_unknown_price_on_any_request_keeps_batch_total_null():
    known_attempt = AttemptRecordV1(
        stage="jev",
        attempt_index=1,
        invoked=True,
        provider_mock=False,
        provider_id="typesafe",
        model_id="test-model",
        prompt_version="test-prompt-v1",
        status="succeeded",
        latency_ms=10,
        input_tokens=20,
        output_tokens=3,
        cost=0.02,
        cost_currency="USD",
        pricing_status="known",
    )
    unknown_attempt = AttemptRecordV1(
        stage="llm",
        attempt_index=1,
        invoked=True,
        provider_mock=False,
        provider_id="ark",
        model_id="test-model",
        prompt_version="test-prompt-v1",
        status="succeeded",
        latency_ms=15,
        input_tokens=30,
        output_tokens=4,
        cost=None,
        cost_currency=None,
        pricing_status="unknown",
    )
    report = build_run_report(
        [
            _envelope(
                "r1",
                pricing_status="known",
                total_cost=0.02,
                cost_currency="USD",
                attempts=[known_attempt],
            ),
            _envelope(
                "r2",
                pricing_status="unknown",
                total_cost=None,
                cost_currency=None,
                attempts=[unknown_attempt],
            ),
        ],
        report_id="report-pricing",
        run_purpose="demo",
        dataset_kind="synthetic",
        dataset_version="synthetic-v0",
        expected_labels={"r1": "PRODUCT_DEFECT", "r2": "PRODUCT_DEFECT"},
    )

    assert report.pricing_status == "unknown"
    assert report.total_cost is None
    assert report.cost_currency is None
    assert report.attempts == 2


def test_formal_gate_does_not_pass_from_real_data_and_sample_count_alone():
    envelopes = [
        _envelope(f"r{index}", purpose="formal_eval")
        for index in range(100)
    ]
    report = build_run_report(
        envelopes,
        report_id="formal-insufficient-evidence",
        run_purpose="formal_eval",
        dataset_kind="real_labeled",
        dataset_version="real-v1",
        expected_labels={item.request_id: "PRODUCT_DEFECT" for item in envelopes},
    )

    assert report.eligibility.status == "ineligible"
    assert report.eligibility.evidence_manifest_id is None
    assert any("manifest" in reason for reason in report.eligibility.reasons)
    assert any("independent annotators" in reason for reason in report.eligibility.reasons)
    assert any("preregistered" in reason for reason in report.eligibility.reasons)


def test_h3c_formal_gate_keeps_10_of_15_not_met():
    envelopes = [
        _envelope(f"r{index}", purpose="formal_eval")
        for index in range(100)
    ]
    evidence = EvaluationEvidence(
        evidence_manifest_id="manifest-h3c-v1",
        claim_scope="h3c_specific",
        protocol_preregistered=True,
        source_governance_complete=True,
        independent_annotator_count=2,
        annotators_independent=True,
        double_annotated_sample_count=100,
        arbitration_complete=True,
        strategy_frozen=True,
        input_projection_frozen=True,
        h3c_direct_real_strict_unique_count=10,
    )
    report = build_run_report(
        envelopes,
        report_id="formal-h3c-short",
        run_purpose="formal_eval",
        dataset_kind="real_labeled",
        dataset_version="h3c-eval-v1",
        expected_labels={item.request_id: "PRODUCT_DEFECT" for item in envelopes},
        evaluation_evidence=evidence,
    )

    assert report.eligibility.status == "ineligible"
    assert any("15 strict-unique" in reason for reason in report.eligibility.reasons)


def test_complete_non_h3c_formal_evidence_can_pass():
    envelopes = [
        _envelope(f"r{index}", purpose="formal_eval")
        for index in range(100)
    ]
    evidence = EvaluationEvidence(
        evidence_manifest_id="manifest-general-v1",
        claim_scope="general_network_automation",
        protocol_preregistered=True,
        source_governance_complete=True,
        independent_annotator_count=2,
        annotators_independent=True,
        double_annotated_sample_count=100,
        arbitration_complete=True,
        strategy_frozen=True,
        input_projection_frozen=True,
    )
    report = build_run_report(
        envelopes,
        report_id="formal-general-valid",
        run_purpose="formal_eval",
        dataset_kind="real_labeled",
        dataset_version="real-v1",
        expected_labels={item.request_id: "PRODUCT_DEFECT" for item in envelopes},
        evaluation_evidence=evidence,
    )

    assert report.eligibility.status == "eligible"
    assert report.eligibility.reasons == []
    assert report.eligibility.evidence_manifest_id == "manifest-general-v1"


def test_formal_eligibility_requires_actual_provider_model_and_prompt_versions():
    unversioned_jev = AttemptRecordV1(
        stage="jev",
        attempt_index=1,
        invoked=True,
        provider_mock=False,
        provider_id="typesafe",
        model_id=None,
        prompt_version=None,
        status="succeeded",
        latency_ms=1,
        pricing_status="unknown",
    )
    envelopes = [
        _envelope(
            f"r{index}",
            purpose="formal_eval",
            pricing_status="unknown",
            total_cost=None,
            attempts=[unversioned_jev],
        )
        for index in range(100)
    ]
    evidence = EvaluationEvidence(
        evidence_manifest_id="manifest-general-v1",
        claim_scope="general_network_automation",
        protocol_preregistered=True,
        source_governance_complete=True,
        independent_annotator_count=2,
        annotators_independent=True,
        double_annotated_sample_count=100,
        arbitration_complete=True,
        strategy_frozen=True,
        input_projection_frozen=True,
    )
    report = build_run_report(
        envelopes,
        report_id="formal-unversioned-provider",
        run_purpose="formal_eval",
        dataset_kind="real_labeled",
        dataset_version="real-v1",
        expected_labels={item.request_id: "PRODUCT_DEFECT" for item in envelopes},
        evaluation_evidence=evidence,
    )

    assert report.eligibility.status == "ineligible"
    assert any("model and prompt/question versions" in reason for reason in report.eligibility.reasons)


def test_legacy_accuracy_mode_rejects_real_or_formal_use():
    with pytest.raises(ValueError, match="only valid for synthetic data"):
        build_run_report(
            [_envelope()],
            report_id="bad-real-legacy",
            run_purpose="advisory",
            dataset_kind="real_labeled",
            dataset_version="real-v1",
            expected_labels={"req-1": "PRODUCT_DEFECT"},
            accuracy_mode="legacy_exact_synthetic",
        )
    with pytest.raises(ValueError, match="formal evaluation cannot use legacy"):
        build_run_report(
            [_envelope(purpose="formal_eval")],
            report_id="bad-formal-legacy",
            run_purpose="formal_eval",
            dataset_kind="synthetic",
            dataset_version="synthetic-v0",
            expected_labels={"req-1": "PRODUCT_DEFECT"},
            accuracy_mode="legacy_exact_synthetic",
        )
