from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from jevtest.execution_contracts import CommonFactsV1, EvidenceRefV1, ReviewRecordV1, RouteStepV1, TriageEnvelopeV1
from jevtest.local_store import (
    EnvelopeConflictError,
    HandoffConflictError,
    IdempotencyConflictError,
    LocalStore,
    ReviewConflictError,
    StaleResultError,
)
from jevtest.models import TriageResult


def _result(category: str = "UNKNOWN", reason: str | None = None) -> TriageResult:
    defaults = {
        "UNKNOWN": ("HUMAN_TRIAGE", 3, 0.0, "Evidence is insufficient"),
        "PRODUCT_DEFECT": ("ASSIGN_PRODUCT_DEVELOPER", 4, 0.91, "R1: assertion reproduces"),
        "FLAKY": ("RERUN", 2, 0.78, "R2: previous run passed"),
    }
    action, priority, confidence, default_reason = defaults[category]
    return TriageResult(
        category=category,
        action=action,
        priority=priority,
        confidence=confidence,
        reason=reason or default_reason,
    )


def _envelope(**overrides: object) -> TriageEnvelopeV1:
    payload: dict[str, object] = {
        "schema_version": "triage-envelope-v1",
        "request_id": "req-001",
        "result_version": "result-v1",
        "created_at": "2026-09-27T12:00:00Z",
        "input_hash": "a" * 64,
        "input_projection_version": "common-facts-v1",
        "projection_hash": "b" * 64,
        "policy_id": "offline-demo",
        "policy_version": "1",
        "execution_mode": "offline",
        "run_purpose": "demo",
        "run_status": "result_ready",
        "result": _result(),
        "route_steps": [
            {
                "stage": "rule",
                "status": "completed",
                "attempts": 1,
                "provider_mock": False,
                "reason_code": "rule_unknown",
            },
            {
                "stage": "human",
                "status": "skipped",
                "attempts": 0,
                "provider_mock": False,
                "reason_code": "requires_review",
            },
        ],
        "terminal_stage": "human",
        "attempts": [],
        "evidence_refs": [
            {
                "evidence_id": "ev-001",
                "field_path": "syslog_excerpt",
                "content_sha256": "c" * 64,
                "capture_scope": "provided_excerpt",
                "evidence_kind": "observed",
            }
        ],
        "missing_evidence": ["device reachability confirmation"],
        "total_latency_ms": 12.0,
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
    return TriageEnvelopeV1.model_validate(payload)


def _facts(**overrides: object) -> CommonFactsV1:
    payload: dict[str, object] = {
        "schema_version": "common-facts-v1",
        "result": "FAIL",
        "previous_runs": ["PASS", "PASS", "FAIL"],
        "device": "AP-660",
        "sw_version": "V6R1",
        "assertion": "association timeout",
        "syslog_excerpt": "CAPWAP association failed\nstatus=timeout",
        "env": {"AP_reachable": True, "lab": "rack-a"},
        "retry": "not_executed",
    }
    payload.update(overrides)
    return CommonFactsV1.model_validate(payload)


def _review(
    envelope: TriageEnvelopeV1,
    *,
    review_id: str = "review-001",
    decision: str = "defer",
    actor_type: str = "test",
    actor_origin: str = "pytest:test_local_store",
    before_result: TriageResult | None = None,
    after_result: TriageResult | None = None,
    reason: str | None = None,
) -> ReviewRecordV1:
    return ReviewRecordV1(
        schema_version="review-record-v1",
        review_id=review_id,
        request_id=envelope.request_id,
        result_version=envelope.result_version,
        actor_type=actor_type,
        actor_origin=actor_origin,
        assistance_mode="unaided" if actor_type == "local_user" else "model_only",
        decision=decision,
        before_result=before_result or envelope.result,
        after_result=after_result,
        reason=reason,
        reviewed_at="2026-09-27T12:01:00Z",
    )


def _queue(store: LocalStore, envelope: TriageEnvelopeV1, **overrides):
    args = {
        "idempotency_scope": "local-demo",
        "idempotency_key": "key-001",
        "target_role": "test_owner",
    }
    args.update(overrides)
    return store.queue_handoff(envelope, **args)


def test_queue_is_idempotent_scoped_and_persists_only_local_receipt(tmp_path):
    db = tmp_path / "triage.sqlite3"
    first_input = _envelope()
    with LocalStore(db) as store:
        first_envelope, first_handoff = store.queue_handoff(
            first_input,
            idempotency_scope="local-demo",
            idempotency_key="key-001",
            target_role="test_owner",
            common_facts=_facts(),
        )
        replay_envelope, replay_handoff = _queue(store, first_input)

        assert first_handoff.delivery_status == "local_queued"
        assert first_handoff.local_receipt_id.startswith("local-receipt-")
        assert first_envelope.delivery_status == "local_queued"
        assert first_envelope.local_handoff_id == first_handoff.handoff_id
        assert replay_handoff == first_handoff
        assert replay_envelope == first_envelope
        assert len(store.list_queued_handoffs()) == 1
        assert store.get_common_facts("req-001", "result-v1") == _facts()


def test_same_scoped_key_with_different_input_hash_conflicts_without_overwrite(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        first_envelope, first_handoff = _queue(store, _envelope())
        different_input = _envelope(request_id="req-002", input_hash="d" * 64)
        with pytest.raises(IdempotencyConflictError, match="different input_hash"):
            _queue(store, different_input)

        assert store.get_envelope("req-001").local_handoff_id == first_handoff.handoff_id
        assert store.get_envelope("req-002") is None
        assert store.get_handoff(first_handoff.handoff_id) == first_handoff
        assert len(store.list_queued_handoffs()) == 1
        assert first_envelope.delivery_status == "local_queued"


def test_failed_queue_write_rolls_back_envelope_and_handoff(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        with pytest.raises(ValidationError):
            store.queue_handoff(
                _envelope(),
                idempotency_scope="local-demo",
                idempotency_key="key-rollback",
                target_role="test_owner",
                common_facts={"schema_version": "common-facts-v1"},
            )
        assert store.get_envelope("req-001") is None
        assert store.list_queued_handoffs() == []


def test_reopen_restores_queued_state(tmp_path):
    db = tmp_path / "triage.sqlite3"
    with LocalStore(db) as store:
        envelope, handoff = _queue(store, _envelope())

    with LocalStore(db) as reopened:
        restored = reopened.get_envelope("req-001")
        assert restored == envelope
        assert reopened.get_handoff(handoff.handoff_id) == handoff
        assert reopened.list_queued_handoffs() == [(envelope, handoff)]


def test_queue_rejects_results_that_do_not_require_human_review(tmp_path):
    needs_no_review = _envelope(
        result=_result("FLAKY"),
        requires_human_review=False,
        review_status="not_requested",
    )
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        with pytest.raises(HandoffConflictError, match="requires human review"):
            _queue(store, needs_no_review)
        assert store.get_envelope("req-001") is None
        assert store.list_queued_handoffs() == []


def test_put_envelope_is_immutable_for_request_result_key(tmp_path):
    envelope = _envelope()
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        assert store.put_envelope(envelope) == envelope
        assert store.put_envelope(envelope) == envelope
        changed = _envelope(result=_result("FLAKY"))
        with pytest.raises(EnvelopeConflictError, match="immutable"):
            store.put_envelope(changed)
        assert store.get_envelope("req-001", "result-v1") == envelope


def test_put_envelope_cannot_import_preapproved_workflow_status(tmp_path):
    forged = _envelope(review_status="accepted")
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        with pytest.raises(ReviewConflictError, match="unreviewed result"):
            store.put_envelope(forged)
        assert store.get_envelope("req-001") is None


def test_new_result_version_becomes_current_and_old_review_is_stale(tmp_path):
    original = _envelope()
    newer = _envelope(result_version="result-v2", result=_result("FLAKY"))
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        store.put_envelope(original)
        store.put_envelope(newer)
        assert store.get_envelope("req-001") == newer
        with pytest.raises(StaleResultError, match="stale"):
            store.append_review(_review(original))


def test_review_history_retains_defer_then_accept_and_actor_origin(tmp_path):
    envelope = _envelope()
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        store.put_envelope(envelope)
        deferred = _review(envelope, review_id="review-defer", decision="defer")
        accepted = _review(
            envelope,
            review_id="review-accept",
            decision="accept",
            actor_type="local_user",
            actor_origin="tty:local-user-asserted",
        )
        store.append_review(deferred)
        store.append_review(accepted)

        assert store.list_reviews("req-001") == [deferred, accepted]
        assert store.get_envelope("req-001").review_status == "accepted"
        assert store.get_envelope("req-001").result == envelope.result
        assert store.list_reviews("req-001")[1].actor_type == "local_user"
        assert store.list_reviews("req-001")[1].actor_origin == "tty:local-user-asserted"

        with pytest.raises(ReviewConflictError, match="terminal review status"):
            store.append_review(
                _review(envelope, review_id="review-late-defer", decision="defer")
            )
        assert len(store.list_reviews("req-001")) == 2


def test_model_review_origin_is_kept_distinct_from_local_user(tmp_path):
    envelope = _envelope()
    model_event = _review(
        envelope,
        review_id="review-model",
        decision="accept",
        actor_type="model",
        actor_origin="ark:test-stub-v1",
    )
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        store.put_envelope(envelope)
        saved = store.append_review(model_event)

        assert saved.actor_type == "model"
        assert saved.actor_origin == "ark:test-stub-v1"
        assert saved.actor_type != "local_user"
        assert store.get_envelope("req-001").review_status == "pending"


def test_correct_requires_reason_and_persists_new_version_with_after_result(tmp_path):
    envelope = _envelope()
    correction = _result("PRODUCT_DEFECT")
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        store.put_envelope(envelope)
        blank_reason = _review(
            envelope,
            decision="correct",
            after_result=correction,
            reason="  ",
        )
        with pytest.raises(ReviewConflictError, match="non-empty reason"):
            store.append_review(blank_reason)
        with pytest.raises(ValidationError, match="correct decision requires after_result"):
            _review(envelope, decision="correct", reason="corrected after checking evidence")

        event = _review(
            envelope,
            review_id="review-correct",
            decision="correct",
            actor_type="local_user",
            actor_origin="tty:confirmed-correction",
            after_result=correction,
            reason="Checked the failure against the provided log excerpt.",
        )
        saved = store.append_review(event)
        current = store.get_envelope("req-001")
        parent = store.get_envelope("req-001", "result-v1")

        assert saved.request_id == "req-001"
        assert saved.result_version == "result-v1"
        assert parent.review_status == "corrected"
        assert current.result_version != "result-v1"
        assert current.result_version.startswith("result-v1:correction:review-correct")
        assert current.result == correction
        assert current.review_status == "pending"
        assert store.list_reviews("req-001", "result-v1") == [event]
        with pytest.raises(StaleResultError, match="stale"):
            store.append_review(_review(parent, review_id="review-stale-after-correction"))


def test_corrected_version_carries_forward_common_facts(tmp_path):
    envelope = _envelope()
    facts = _facts()
    event = _review(
        envelope,
        decision="correct",
        after_result=_result("PRODUCT_DEFECT"),
        reason="Corrected from provided evidence.",
    )
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        store.put_envelope(envelope)
        store.put_common_facts(envelope.request_id, envelope.result_version, facts)
        store.append_review(event)
        current = store.get_envelope(envelope.request_id)
        assert store.get_common_facts(envelope.request_id, current.result_version) == facts


def test_review_id_is_idempotent_but_conflicting_reuse_is_rejected(tmp_path):
    envelope = _envelope()
    event = _review(envelope, review_id="review-same-id", decision="defer")
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        store.put_envelope(envelope)
        assert store.append_review(event) == event
        assert store.append_review(event) == event
        assert store.list_reviews("req-001") == [event]
        changed = _review(
            envelope,
            review_id="review-same-id",
            decision="defer",
            reason="different event body",
        )
        with pytest.raises(ReviewConflictError, match="different event content"):
            store.append_review(changed)


def test_review_before_result_must_match_stored_result(tmp_path):
    envelope = _envelope()
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        store.put_envelope(envelope)
        with pytest.raises(ReviewConflictError, match="before_result"):
            store.append_review(
                _review(envelope, before_result=_result("FLAKY"))
            )
        assert store.list_reviews("req-001") == []


def test_review_events_are_append_only_in_sqlite(tmp_path):
    envelope = _envelope()
    event = _review(envelope)
    db = tmp_path / "triage.sqlite3"
    with LocalStore(db) as store:
        store.put_envelope(envelope)
        store.append_review(event)

    connection = sqlite3.connect(db)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE reviews SET review_json = '{}' WHERE review_id = ?", (event.review_id,))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM reviews WHERE review_id = ?", (event.review_id,))
    finally:
        connection.close()
    with LocalStore(db) as reopened:
        assert reopened.list_reviews("req-001") == [event]


def test_corrected_current_head_is_in_review_queue_after_restart(tmp_path):
    db = tmp_path / "triage.sqlite3"
    original = _envelope()
    correction = _review(
        original,
        review_id="review-correct-queue",
        decision="correct",
        actor_type="local_user",
        after_result=_result("PRODUCT_DEFECT"),
        reason="Checked the submitted evidence and corrected the classification.",
    )
    with LocalStore(db) as store:
        _, old_handoff = _queue(store, original)
        store.append_review(correction)
        corrected = store.get_envelope(original.request_id)
        assert corrected is not None
        assert corrected.result_version != original.result_version
        assert corrected.review_status == "pending"
        assert store.list_current_review_queue() == [(corrected, None)]

    with LocalStore(db) as reopened:
        corrected = reopened.get_envelope(original.request_id)
        assert corrected is not None
        assert reopened.list_current_review_queue() == [(corrected, None)]
        assert reopened.get_handoff(old_handoff.handoff_id) == old_handoff
        assert reopened.list_queued_handoffs()[0][0].review_status == "corrected"
        assert reopened.list_reviews(original.request_id) == [correction]
        accepted = _review(
            corrected,
            review_id="review-accept-corrected-queue",
            decision="accept",
            actor_type="local_user",
            actor_origin="test:local-review",
        )
        reopened.append_review(accepted)
        assert reopened.list_current_review_queue() == []


def test_card_json_and_markdown_render_stored_projection_and_export(tmp_path):
    db = tmp_path / "triage.sqlite3"
    with LocalStore(db) as store:
        _, handoff = store.queue_handoff(
            _envelope(),
            idempotency_scope="local-demo",
            idempotency_key="key-001",
            target_role="test_owner",
            common_facts=_facts(),
        )
        card_json = json.loads(store.render_handoff_json(handoff.handoff_id))
        card_markdown = store.render_handoff_markdown(handoff.handoff_id)
        json_path = store.export_handoff_json(handoff.handoff_id, tmp_path / "export" / "card.json")
        md_path = store.export_handoff_markdown(handoff.handoff_id, tmp_path / "export" / "card.md")

    assert card_json["handoff"]["delivery_status"] == "local_queued"
    assert card_json["common_facts"]["assertion"] == "association timeout"
    assert "external person or system received it" in card_json["trust_boundary"]
    assert "case_id" not in card_json["common_facts"]
    assert "provenance" not in card_json["common_facts"]
    assert "association timeout" in card_markdown
    assert "CAPWAP association failed" in card_markdown
    assert "device reachability confirmation" in card_markdown
    assert "rule: completed" in card_markdown
    assert "local_queued" in card_markdown
    assert "not externally delivered" in card_markdown
    assert json.loads(json_path.read_text(encoding="utf-8")) == card_json
    assert md_path.read_text(encoding="utf-8") == card_markdown


def test_common_facts_reject_source_labels_and_conflicting_rewrite(tmp_path):
    envelope = _envelope()
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        store.put_envelope(envelope)
        with pytest.raises(ValidationError):
            store.put_common_facts(
                envelope.request_id,
                envelope.result_version,
                {**_facts().model_dump(mode="json"), "case_id": "source-label"},
            )
        store.put_common_facts(envelope.request_id, envelope.result_version, _facts())
        with pytest.raises(EnvelopeConflictError, match="common facts are immutable"):
            store.put_common_facts(
                envelope.request_id,
                envelope.result_version,
                _facts(assertion="a changed input projection"),
            )


def test_concurrent_same_key_submissions_create_one_handoff(tmp_path):
    db = tmp_path / "concurrent.sqlite3"
    first = LocalStore(db)
    second = LocalStore(db)
    barrier = threading.Barrier(2)

    def submit(store: LocalStore):
        barrier.wait(timeout=5)
        return _queue(store, _envelope())

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            left_future = pool.submit(submit, first)
            right_future = pool.submit(submit, second)
            left = left_future.result(timeout=10)
            right = right_future.result(timeout=10)
        assert left == right
        assert len(first.list_queued_handoffs()) == 1
    finally:
        first.close()
        second.close()
