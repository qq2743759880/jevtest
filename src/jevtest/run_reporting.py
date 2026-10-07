"""Construction of honest run reports and fail-closed evaluation eligibility.

This module consumes completed execution envelopes.  It does not make provider
calls, infer evidence quality from ``synthetic=False``, or modify the frozen
benchmark-v1 report path.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose
from typing import Literal

from jevtest.execution_contracts import (
    EvaluationEligibilityV1,
    RunPurpose,
    RunReportV2,
    TriageEnvelopeV1,
)

AccuracyMode = Literal["triage_v2", "legacy_exact_synthetic"]
ClaimScope = Literal["general_network_automation", "h3c_specific"]

_CRITERIA_VERSION = "formal-effectiveness-v1"
_MIN_FORMAL_REAL_SAMPLES = 100
_MIN_H3C_STRICT_UNIQUE = 15


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    """Explicit evidence facts used by the formal-evaluation gate.

    Defaults deliberately fail closed.  The evidence manifest is the durable
    source for these attestations; this value object only carries the facts the
    report builder needs to decide whether the run may claim eligibility.
    """

    evidence_manifest_id: str | None = None
    claim_scope: ClaimScope | None = None
    protocol_preregistered: bool = False
    source_governance_complete: bool = False
    independent_annotator_count: int = 0
    annotators_independent: bool = False
    double_annotated_sample_count: int = 0
    arbitration_complete: bool = False
    strategy_frozen: bool = False
    input_projection_frozen: bool = False
    h3c_direct_real_strict_unique_count: int | None = None


def evaluate_eligibility(
    *,
    run_purpose: RunPurpose,
    dataset_kind: Literal["synthetic", "real_labeled", "real_unlabeled", "mixed"],
    dataset_version: str | None,
    sample_count: int,
    evaluated_count: int,
    envelopes: Sequence[TriageEnvelopeV1],
    evidence: EvaluationEvidence | None = None,
) -> EvaluationEligibilityV1:
    """Return an eligibility record; formal eligibility requires every gate.

    ``sample_count`` and ``synthetic=False`` are necessary facts at most.  The
    report also requires real labels, a minimum-sized run, a manifest,
    independent annotation and arbitration, registered protocol, governed
    sources, frozen strategy/input, and a declared claim scope.  H3C-specific
    claims additionally retain the existing 15 strict-unique direct-real gate.
    """

    facts = evidence or EvaluationEvidence()
    if run_purpose != "formal_eval":
        return EvaluationEligibilityV1(
            status="not_requested",
            criteria_version=_CRITERIA_VERSION,
            reasons=[f"Formal evaluation was not requested (run_purpose={run_purpose})."],
            evidence_manifest_id=facts.evidence_manifest_id,
        )

    reasons: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            reasons.append(message)

    require(
        dataset_kind == "real_labeled",
        "Formal evaluation requires a fully labeled real dataset.",
    )
    require(bool(dataset_version), "A versioned evaluation dataset is required.")
    require(
        sample_count >= _MIN_FORMAL_REAL_SAMPLES,
        f"Formal evaluation requires at least {_MIN_FORMAL_REAL_SAMPLES} real samples.",
    )
    require(
        evaluated_count == sample_count,
        "Every formal-evaluation sample must have a ground-truth label.",
    )
    require(
        bool(facts.evidence_manifest_id),
        "A formal-evaluation evidence manifest is required.",
    )
    require(facts.claim_scope is not None, "The evaluation claim scope must be declared.")
    require(
        facts.protocol_preregistered,
        "The evaluation protocol must be preregistered.",
    )
    require(
        facts.source_governance_complete,
        "Source authenticity and governance must be reviewed.",
    )
    require(
        facts.independent_annotator_count >= 2 and facts.annotators_independent,
        "At least two independent annotators are required.",
    )
    require(
        facts.double_annotated_sample_count >= evaluated_count,
        "Every evaluated sample must have two independent labels.",
    )
    require(facts.arbitration_complete, "Label disagreements must be arbitrated.")
    require(facts.strategy_frozen, "The evaluation strategy must be frozen.")
    require(facts.input_projection_frozen, "The evaluated input projection must be frozen.")

    purposes_match = all(item.run_purpose == "formal_eval" for item in envelopes)
    require(purposes_match, "Every execution envelope must be marked formal_eval.")
    all_have_invoked_attempts = bool(envelopes) and all(
        any(attempt.invoked for attempt in item.attempts) for item in envelopes
    )
    require(
        all_have_invoked_attempts,
        "Every formal-evaluation sample must have at least one actual provider attempt.",
    )
    invoked_attempts = [
        attempt for item in envelopes for attempt in item.attempts if attempt.invoked
    ]
    require(
        bool(invoked_attempts) and all(not attempt.provider_mock for attempt in invoked_attempts),
        "Formal evaluation cannot use mock provider attempts.",
    )
    require(
        all(
            attempt.provider_id and attempt.output_schema_version
            for attempt in invoked_attempts
        ),
        "Every formal provider attempt must bind a provider and output-schema version.",
    )
    require(
        all(
            attempt.stage not in {"jev", "llm"}
            or (attempt.model_id and attempt.prompt_version)
            for attempt in invoked_attempts
        ),
        "Formal Jev/LLM attempts must bind model and prompt/question versions.",
    )
    policy_versions = {(item.policy_id, item.policy_version) for item in envelopes}
    require(
        len(policy_versions) == 1 and bool(envelopes),
        "Formal evaluation requires one frozen policy id and version across the run.",
    )
    projection_versions = {item.input_projection_version for item in envelopes}
    require(
        len(projection_versions) == 1 and bool(envelopes),
        "Formal evaluation requires one frozen input projection version across the run.",
    )

    if facts.claim_scope == "h3c_specific":
        count = facts.h3c_direct_real_strict_unique_count
        require(
            count is not None and count >= _MIN_H3C_STRICT_UNIQUE,
            "H3C-specific effectiveness requires at least 15 strict-unique direct-real roots.",
        )

    return EvaluationEligibilityV1(
        status="eligible" if not reasons else "ineligible",
        criteria_version=_CRITERIA_VERSION,
        reasons=reasons,
        evidence_manifest_id=facts.evidence_manifest_id,
    )


def build_run_report(
    envelopes: Sequence[TriageEnvelopeV1],
    *,
    report_id: str,
    run_purpose: RunPurpose,
    dataset_kind: Literal["synthetic", "real_labeled", "real_unlabeled", "mixed"],
    dataset_version: str | None,
    expected_labels: Mapping[str, str] | None = None,
    accuracy_mode: AccuracyMode = "triage_v2",
    evaluation_evidence: EvaluationEvidence | None = None,
) -> RunReportV2:
    """Build a RunReportV2 from already-completed requests, without I/O.

    Missing predictions remain in the labeled denominator and therefore do
    not disappear from accuracy.  ``triage_v2`` treats UNKNOWN fallback as an
    incorrect answer; ``legacy_exact_synthetic`` preserves v1's exact-category
    comparison solely for synthetic demo scores.
    """

    items = tuple(envelopes)
    labels = dict(expected_labels or {})
    request_ids = [item.request_id for item in items]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_id values must be unique within a run report")
    unknown_label_ids = sorted(set(labels) - set(request_ids))
    if unknown_label_ids:
        raise ValueError(f"labels reference requests outside this run: {unknown_label_ids}")
    if any(not isinstance(label, str) or not label for label in labels.values()):
        raise ValueError("expected labels must be non-empty strings")

    if dataset_kind == "real_unlabeled" and labels:
        raise ValueError("real_unlabeled reports cannot include ground-truth labels")
    if dataset_kind == "synthetic" and items:
        if run_purpose == "synthetic_eval" and len(labels) != len(items):
            raise ValueError("synthetic_eval reports require a label for every request")
        if labels and len(labels) != len(items):
            raise ValueError("synthetic reports require either labels for all requests or none")
    if dataset_kind == "real_labeled" and items and len(labels) != len(items):
        raise ValueError("real_labeled reports require a label for every request")
    if dataset_kind == "mixed" and items and not (0 < len(labels) < len(items)):
        raise ValueError("mixed reports require both labeled and unlabeled requests")

    if accuracy_mode not in {"triage_v2", "legacy_exact_synthetic"}:
        raise ValueError(f"unsupported accuracy_mode: {accuracy_mode!r}")
    if accuracy_mode == "legacy_exact_synthetic":
        if dataset_kind != "synthetic":
            raise ValueError("legacy_exact_synthetic accuracy is only valid for synthetic data")
        if run_purpose == "formal_eval":
            raise ValueError("formal evaluation cannot use legacy synthetic accuracy")

    purposes = {item.run_purpose for item in items}
    if purposes and purposes != {run_purpose}:
        raise ValueError("run_purpose must match every execution envelope")

    by_request = {item.request_id: item for item in items}
    evaluated_count = len(labels)
    if evaluated_count:
        correct_count = 0
        for request_id, expected in labels.items():
            envelope = by_request[request_id]
            predicted = envelope.result.category if envelope.result is not None else None
            if accuracy_mode == "legacy_exact_synthetic":
                correct_count += predicted == expected
            else:
                correct_count += predicted == expected and predicted != "UNKNOWN"
        accuracy: float | None = correct_count / evaluated_count
        accuracy_null_reason: str | None = None
    else:
        accuracy = None
        accuracy_null_reason = (
            "Ground-truth labels were not supplied; accuracy was not computed."
        )

    eligibility = evaluate_eligibility(
        run_purpose=run_purpose,
        dataset_kind=dataset_kind,
        dataset_version=dataset_version,
        sample_count=len(items),
        evaluated_count=evaluated_count,
        envelopes=items,
        evidence=evaluation_evidence,
    )
    attempts = sum(sum(attempt.invoked for attempt in item.attempts) for item in items)
    failed_count = sum(item.run_status in {"failed", "cancelled"} for item in items)
    route_counts: Counter[str] = Counter()
    for item in items:
        if item.result is None:
            continue
        if item.requires_human_review or item.result.category == "UNKNOWN":
            route = "human"
        elif item.terminal_stage is not None:
            route = item.terminal_stage
        else:
            route = "unknown"
        route_counts[route] += 1

    pricing_status, total_cost, cost_currency = _aggregate_pricing(items)
    return RunReportV2(
        schema_version="run-report-v2",
        report_id=report_id,
        run_purpose=run_purpose,
        dataset_kind=dataset_kind,
        dataset_version=dataset_version,
        sample_count=len(items),
        evaluated_count=evaluated_count,
        accuracy=accuracy,
        accuracy_null_reason=accuracy_null_reason,
        accuracy_mode=accuracy_mode,
        eligibility=eligibility,
        attempts=attempts,
        failed_count=failed_count,
        pricing_status=pricing_status,
        total_cost=total_cost,
        cost_currency=cost_currency,
        route_counts=dict(sorted(route_counts.items())),
    )


def _aggregate_pricing(
    envelopes: Sequence[TriageEnvelopeV1],
) -> tuple[Literal["no_external_cost", "mock", "known", "unknown"], float | None, str | None]:
    """Aggregate pricing without converting unknown or inconsistent data to zero."""

    if not envelopes or any(item.pricing_status == "unknown" for item in envelopes):
        return "unknown", None, None

    for envelope in envelopes:
        invoked = [attempt for attempt in envelope.attempts if attempt.invoked]
        if not invoked:
            continue
        if any(attempt.pricing_status == "unknown" for attempt in invoked):
            return "unknown", None, None

        known_attempts = [attempt for attempt in invoked if attempt.pricing_status == "known"]
        non_known_attempts = [attempt for attempt in invoked if attempt.pricing_status != "known"]
        if any(
            attempt.cost not in (None, 0, 0.0) or attempt.cost_currency is not None
            for attempt in non_known_attempts
        ):
            return "unknown", None, None
        if known_attempts:
            currencies = {attempt.cost_currency for attempt in known_attempts}
            if (
                len(currencies) != 1
                or None in currencies
                or any(attempt.cost is None for attempt in known_attempts)
                or envelope.pricing_status != "known"
                or envelope.total_cost is None
                or envelope.cost_currency not in currencies
            ):
                return "unknown", None, None
            attempt_total = round(sum(attempt.cost or 0.0 for attempt in known_attempts), 12)
            if not isclose(attempt_total, envelope.total_cost, rel_tol=1e-9, abs_tol=1e-9):
                return "unknown", None, None
        else:
            attempt_statuses = {attempt.pricing_status for attempt in invoked}
            expected_status = "mock" if "mock" in attempt_statuses else "no_external_cost"
            if envelope.pricing_status != expected_status:
                return "unknown", None, None
            if any(
                attempt.cost not in (None, 0, 0.0) or attempt.cost_currency is not None
                for attempt in invoked
            ):
                return "unknown", None, None

    known = [item for item in envelopes if item.pricing_status == "known"]
    for item in envelopes:
        if item.pricing_status in {"mock", "no_external_cost"} and (
            item.total_cost not in (None, 0, 0.0) or item.cost_currency is not None
        ):
            return "unknown", None, None

    if known:
        currencies = {item.cost_currency for item in known}
        if len(currencies) != 1 or None in currencies or any(item.total_cost is None for item in known):
            return "unknown", None, None
        currency = next(iter(currencies))
        total = sum(item.total_cost or 0.0 for item in known)
        return "known", round(total, 12), currency

    if any(item.pricing_status == "mock" for item in envelopes):
        return "mock", 0.0, None
    return "no_external_cost", 0.0, None
