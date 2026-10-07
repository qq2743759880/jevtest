"""Versioned outer contracts for request execution, routing, reporting and local review.

The frozen FailureState, TriageResult, rule baseline and benchmark v1 are embedded
or referenced without changing their meaning.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevtest.models import FailureState, TriageResult

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
UnitInterval = Annotated[float, Field(ge=0, le=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]

ExecutionMode = Literal["offline", "live"]
RunPurpose = Literal[
    "demo", "advisory", "synthetic_eval", "real_exploratory", "formal_eval"
]
Stage = Literal["rule", "jev", "llm", "human"]
PricingStatus = Literal["no_external_cost", "mock", "known", "unknown"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TriageRequestV1(_StrictModel):
    schema_version: Literal["triage-request-v1"]
    request_id: str = Field(min_length=1)
    failure_state: FailureState
    execution_mode: ExecutionMode
    run_purpose: RunPurpose
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    input_projection_version: str = Field(min_length=1)
    input_hash: Sha256
    projection_hash: Sha256
    idempotency_key: str | None = Field(default=None, min_length=1)


class RoutingPolicyV1(_StrictModel):
    schema_version: Literal["routing-policy-v1"]
    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    execution_mode: ExecutionMode
    run_purpose: RunPurpose
    basis: Literal["fixed_rule", "demo", "synthetic_preview", "formal_calibrated"]
    allowed_stages: list[Stage] = Field(min_length=1)
    direct_output_rule_ids: list[str]
    jev_high_threshold: UnitInterval | None = None
    jev_escalate_threshold: UnitInterval | None = None
    max_total_attempts: int = Field(ge=1)
    deadline_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "RoutingPolicyV1":
        if (
            self.jev_high_threshold is not None
            and self.jev_escalate_threshold is not None
            and self.jev_high_threshold < self.jev_escalate_threshold
        ):
            raise ValueError("jev_high_threshold must be >= jev_escalate_threshold")
        if self.execution_mode == "offline" and self.basis == "formal_calibrated":
            raise ValueError("formal_calibrated policy cannot run in offline demo mode")
        return self


class CommonFactsV1(_StrictModel):
    """Inference view shared by Rule/Jev/LLM; omits dataset and sample labels."""

    schema_version: Literal["common-facts-v1"]
    result: Literal["FAIL"]
    previous_runs: list[Literal["PASS", "FAIL"]] = Field(max_length=10)
    device: str = Field(min_length=1)
    sw_version: str = Field(min_length=1)
    assertion: str = Field(min_length=1)
    syslog_excerpt: str = Field(max_length=8192)
    env: dict[str, bool | str]
    retry: Literal["not_executed", "passed", "failed"]


class RouteStepV1(_StrictModel):
    stage: Stage
    status: Literal["completed", "failed", "skipped"]
    attempts: NonNegativeInt
    provider_mock: bool
    reason_code: str | None = None


class AttemptRecordV1(_StrictModel):
    stage: Stage
    attempt_index: int = Field(ge=1)
    invoked: bool
    provider_mock: bool
    provider_id: str = Field(min_length=1)
    model_id: str | None = None
    prompt_version: str | None = None
    output_schema_version: str = Field(default="triage-result-v1", min_length=1)
    status: Literal["succeeded", "failed", "skipped"]
    latency_ms: float = Field(ge=0)
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    cost: float | None = Field(default=None, ge=0)
    cost_currency: str | None = None
    pricing_status: PricingStatus
    error_code: str | None = None
    error_summary: str | None = Field(default=None, max_length=500)


class EvidenceRefV1(_StrictModel):
    evidence_id: str = Field(min_length=1)
    field_path: str = Field(min_length=1)
    content_sha256: Sha256
    capture_scope: Literal["submitted_field", "provided_excerpt", "source_artifact"]
    evidence_kind: Literal["observed", "derived", "model_output"]
    start_offset: NonNegativeInt | None = None
    end_offset: NonNegativeInt | None = None
    artifact_id: str | None = None


class TriageEnvelopeV1(_StrictModel):
    schema_version: Literal["triage-envelope-v1"]
    request_id: str = Field(min_length=1)
    result_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    input_hash: Sha256
    input_projection_version: str = Field(min_length=1)
    projection_hash: Sha256
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    execution_mode: ExecutionMode
    run_purpose: RunPurpose
    run_status: Literal["result_ready", "degraded", "failed", "cancelled"]
    result: TriageResult | None = None
    route_steps: list[RouteStepV1]
    terminal_stage: Stage | None = None
    attempts: list[AttemptRecordV1]
    evidence_refs: list[EvidenceRefV1]
    missing_evidence: list[str]
    total_latency_ms: float = Field(ge=0)
    total_cost: float | None = Field(default=None, ge=0)
    cost_currency: str | None = None
    pricing_status: PricingStatus
    requires_human_review: bool
    review_status: Literal["not_requested", "pending", "accepted", "corrected", "deferred"]
    delivery_status: Literal["not_requested", "pending", "local_queued", "failed"]
    resolution_status: Literal["unknown", "open", "resolved"]
    local_handoff_id: str | None = None

    @model_validator(mode="after")
    def status_matches_result_and_handoff(self) -> "TriageEnvelopeV1":
        if self.run_status in {"result_ready", "degraded"} and self.result is None:
            raise ValueError("result_ready/degraded requires a TriageResult")
        if self.run_status in {"failed", "cancelled"} and self.result is not None:
            raise ValueError("failed/cancelled cannot claim a completed TriageResult")
        if self.delivery_status == "local_queued" and not self.local_handoff_id:
            raise ValueError("local_queued requires local_handoff_id")
        if self.delivery_status != "local_queued" and self.local_handoff_id is not None:
            raise ValueError("local_handoff_id is valid only after local queue persistence")
        if self.result is not None and self.result.category == "UNKNOWN" and not self.requires_human_review:
            raise ValueError("UNKNOWN result requires human review")
        return self


class EvaluationEligibilityV1(_StrictModel):
    status: Literal["not_requested", "ineligible", "eligible"]
    criteria_version: str = Field(min_length=1)
    reasons: list[str]
    evidence_manifest_id: str | None = None

    @model_validator(mode="after")
    def eligible_requires_manifest(self) -> "EvaluationEligibilityV1":
        if self.status == "eligible" and not self.evidence_manifest_id:
            raise ValueError("eligible evaluation requires evidence_manifest_id")
        return self


class RunReportV2(_StrictModel):
    schema_version: Literal["run-report-v2"]
    report_id: str = Field(min_length=1)
    run_purpose: RunPurpose
    dataset_kind: Literal["synthetic", "real_labeled", "real_unlabeled", "mixed"]
    accuracy_mode: Literal["triage_v2", "legacy_exact_synthetic"]
    dataset_version: str | None = None
    sample_count: int = Field(ge=0)
    evaluated_count: int = Field(ge=0)
    accuracy: UnitInterval | None = None
    accuracy_null_reason: str | None = None
    eligibility: EvaluationEligibilityV1
    attempts: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    pricing_status: PricingStatus
    total_cost: float | None = Field(default=None, ge=0)
    cost_currency: str | None = None
    route_counts: dict[str, NonNegativeInt]

    @model_validator(mode="after")
    def report_fields_are_consistent(self) -> "RunReportV2":
        if self.accuracy is None and not self.accuracy_null_reason:
            raise ValueError("null accuracy requires accuracy_null_reason")
        if self.accuracy is not None and (self.evaluated_count == 0 or self.accuracy_null_reason is not None):
            raise ValueError("numeric accuracy requires evaluated samples and no null reason")
        if self.evaluated_count > self.sample_count:
            raise ValueError("evaluated_count cannot exceed sample_count")
        if self.failed_count > self.sample_count:
            raise ValueError("failed_count cannot exceed sample_count")
        if self.pricing_status == "unknown" and (self.total_cost is not None or self.cost_currency is not None):
            raise ValueError("unknown pricing must have null total_cost and cost_currency")
        if self.pricing_status == "known" and (self.total_cost is None or not self.cost_currency):
            raise ValueError("known pricing requires total_cost and cost_currency")
        if self.eligibility.status == "eligible" and self.run_purpose != "formal_eval":
            raise ValueError("only formal_eval can be evaluation eligible")
        if self.accuracy_mode == "legacy_exact_synthetic" and (
            self.dataset_kind != "synthetic" or self.run_purpose == "formal_eval"
        ):
            raise ValueError("legacy_exact_synthetic is limited to non-formal synthetic reports")
        return self


class ProviderExecutionMetadataV1(_StrictModel):
    """Requested alias and provider-returned model version for one attempt."""

    schema_version: Literal["provider-execution-metadata-v1"]
    request_id: str = Field(min_length=1)
    result_version: str = Field(min_length=1)
    stage: Literal["jev", "llm"]
    attempt_index: int = Field(ge=1)
    provider_id: str = Field(min_length=1)
    requested_model: str | None = Field(default=None, max_length=200)
    resolved_model: str | None = Field(default=None, max_length=200)
    resolution_source: Literal["provider_response", "not_returned"]

    @model_validator(mode="after")
    def model_value_matches_resolution_source(self) -> "ProviderExecutionMetadataV1":
        if (self.resolved_model is None) != (self.resolution_source == "not_returned"):
            raise ValueError("resolution_source must match whether resolved_model was returned")
        return self


class ReviewRecordV1(_StrictModel):
    schema_version: Literal["review-record-v1"]
    review_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    result_version: str = Field(min_length=1)
    actor_type: Literal["local_user", "model", "test"]
    actor_origin: str = Field(min_length=1)
    assistance_mode: Literal["unaided", "model_assisted", "model_only"]
    decision: Literal["accept", "correct", "defer"]
    before_result: TriageResult
    after_result: TriageResult | None = None
    confidence_origin: Literal["inherited_unreassessed"] | None = None
    reason: str | None = None
    reviewed_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def corrected_result_is_recorded(self) -> "ReviewRecordV1":
        if self.decision == "correct" and self.after_result is None:
            raise ValueError("correct decision requires after_result")
        if self.confidence_origin is not None and self.decision != "correct":
            raise ValueError("confidence_origin is only valid for a corrected result")
        return self


class HandoffRecordV1(_StrictModel):
    schema_version: Literal["handoff-record-v1"]
    handoff_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    result_version: str = Field(min_length=1)
    target_role: str = Field(min_length=1)
    delivery_status: Literal["pending", "local_queued", "failed"]
    local_receipt_id: str | None = None
    idempotency_scope: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    input_hash: Sha256
    created_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def local_receipt_matches_delivery(self) -> "HandoffRecordV1":
        if self.delivery_status == "local_queued" and not self.local_receipt_id:
            raise ValueError("local_queued requires local_receipt_id")
        if self.delivery_status != "local_queued" and self.local_receipt_id is not None:
            raise ValueError("local_receipt_id requires local_queued status")
        return self


VERSIONED_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("TriageRequestV1", TriageRequestV1, "triage-request-v1.schema.json"),
    ("RoutingPolicyV1", RoutingPolicyV1, "routing-policy-v1.schema.json"),
    ("CommonFactsV1", CommonFactsV1, "common-facts-v1.schema.json"),
    ("TriageEnvelopeV1", TriageEnvelopeV1, "triage-envelope-v1.schema.json"),
    ("RunReportV2", RunReportV2, "run-report-v2.schema.json"),
    (
        "ProviderExecutionMetadataV1",
        ProviderExecutionMetadataV1,
        "provider-execution-metadata-v1.schema.json",
    ),
    ("ReviewRecordV1", ReviewRecordV1, "review-record-v1.schema.json"),
    ("HandoffRecordV1", HandoffRecordV1, "handoff-record-v1.schema.json"),
)
