from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from jevtest.app_service import (
    ApplicationIdempotencyConflict,
    SubmittedFactNotFound,
    TriageApplicationService,
    UnsupportedApplicationPolicy,
    default_offline_policy,
)
from jevtest.execution_contracts import RoutingPolicyV1
from jevtest.local_store import LocalStore
from jevtest.models import FailureState, Provenance, TriageResult


def _state(**overrides: object) -> FailureState:
    payload: dict[str, object] = {
        "case_id": "SYN-APP-0001",
        "result": "FAIL",
        "previous_runs": ["PASS", "PASS", "PASS"],
        "device": "AP-660",
        "sw_version": "V6R1",
        "assertion": "association did not complete",
        "syslog_excerpt": "CAPWAP association timed out",
        "env": {"AP_reachable": True, "AC_reachable": True, "lab": "rack-2"},
        "retry": "failed",
        "provenance": Provenance(
            dataset_version="synthetic-v1", synthetic=True, source="local-test"
        ),
    }
    payload.update(overrides)
    return FailureState.model_validate(payload)


def _unknown_state() -> FailureState:
    return _state(assertion="association did not complete", syslog_excerpt="no matching cause")


def _direct_state() -> FailureState:
    return _state(
        case_id="SYN-APP-DIRECT",
        assertion="station unreachable",
        env={"AP_reachable": False, "AC_reachable": True},
    )


def test_default_service_is_offline_mock_only_and_builds_unlabeled_report(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        service = TriageApplicationService(store)
        result = service.submit(_unknown_state(), idempotency_key="key-001")

        assert result.envelope.execution_mode == "offline"
        assert result.envelope.run_purpose == "demo"
        assert result.envelope.requires_human_review is True
        assert result.envelope.delivery_status == "local_queued"
        assert [step.stage for step in result.envelope.route_steps] == ["rule", "jev", "llm", "human"]
        assert all(attempt.provider_mock for attempt in result.envelope.attempts if attempt.stage in {"jev", "llm"})
        assert result.report.accuracy is None
        assert result.report.accuracy_null_reason
        assert result.report.dataset_kind == "synthetic"
        assert result.report.sample_count == 1
        assert result.report_result_version == result.envelope.result_version
        assert len(result.envelope.missing_evidence) == 3
        assert {item.field_path for item in result.envelope.evidence_refs} >= {
            "env.AP_reachable", "env.AC_reachable", "previous_runs", "retry", "syslog_excerpt"
        }
        assert all(item.evidence_kind == "observed" for item in result.envelope.evidence_refs)
        assert next(
            item for item in result.envelope.evidence_refs if item.field_path == "syslog_excerpt"
        ).capture_scope == "provided_excerpt"
        assert result.replayed is False


def test_same_key_replays_stored_result_without_second_cascade(tmp_path, monkeypatch):
    import jevtest.app_service as app_service

    real_run = app_service.run_cascade
    calls = 0

    def count_runs(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(app_service, "run_cascade", count_runs)
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        service = TriageApplicationService(store, idempotency_scope="tenant-local")
        first = service.submit(_unknown_state(), idempotency_key="key-replay")
        assert first.report is not None
        frozen_report = first.report.model_dump(mode="json")

        def no_rebuild(*_args, **_kwargs):
            raise AssertionError("replay must load the persisted report")

        monkeypatch.setattr(service, "_build_report", no_rebuild)
        second = service.submit(_unknown_state(), idempotency_key="key-replay")

        assert calls == 1
        assert second.replayed is True
        assert second.envelope == first.envelope
        assert second.report is not None
        assert second.report.accuracy is None
        assert second.report.model_dump(mode="json") == frozen_report
        assert second.report_result_version == first.envelope.result_version
        assert service.get_result(first.envelope.request_id) == first.envelope
        assert service.get_result_by_key(
            "key-replay", idempotency_scope="local-app"
        ) is None
        assert service.get_result_by_key("key-replay", idempotency_scope="tenant-local") == first.envelope


def test_concurrent_same_key_serializes_before_running_cascade(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        service = TriageApplicationService(store, idempotency_scope="threaded-local")
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(
                workers.map(
                    lambda _index: service.submit(
                        _unknown_state(), idempotency_key="concurrent-key"
                    ),
                    range(4),
                )
            )

        assert len({result.envelope.request_id for result in results}) == 1
        assert sum(result.replayed for result in results) == 3
        assert len(store.list_queued_handoffs()) == 1


def test_same_scope_key_with_changed_input_or_policy_conflicts(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        service = TriageApplicationService(store, idempotency_scope="scope-a")
        first = service.submit(_unknown_state(), idempotency_key="same-key")
        changed_state = _unknown_state().model_copy(update={"assertion": "new submitted assertion"})
        with pytest.raises(ApplicationIdempotencyConflict, match="different input or policy"):
            service.submit(changed_state, idempotency_key="same-key")

        changed_policy = default_offline_policy().model_copy(update={"jev_high_threshold": 0.95})
        with pytest.raises(ApplicationIdempotencyConflict, match="different input or policy"):
            service.submit(_unknown_state(), idempotency_key="same-key", policy=changed_policy)
        assert service.get_result(first.envelope.request_id) == first.envelope


def test_different_keys_with_same_input_create_distinct_requests(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        service = TriageApplicationService(store)
        first = service.submit(_unknown_state(), idempotency_key="key-a")
        second = service.submit(_unknown_state(), idempotency_key="key-b")
        assert first.envelope.request_id != second.envelope.request_id
        assert len(store.list_queued_handoffs()) == 2


def test_non_human_result_is_persisted_without_creating_handoff(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        service = TriageApplicationService(store)
        result = service.submit(_direct_state(), idempotency_key="direct-key")

        assert result.envelope.result.category == "ENVIRONMENT"
        assert result.envelope.requires_human_review is False
        assert result.envelope.delivery_status == "not_requested"
        assert result.envelope.missing_evidence == []
        assert store.list_queued_handoffs() == []
        assert service.get_result_by_key("direct-key") == result.envelope


def test_policy_defaults_offline_and_rejects_live_or_evaluation_policy(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        assert default_offline_policy().execution_mode == "offline"
        live = RoutingPolicyV1.model_validate(
            {
                **default_offline_policy().model_dump(mode="json"),
                "execution_mode": "live",
            }
        )
        with pytest.raises(UnsupportedApplicationPolicy, match="offline"):
            TriageApplicationService(store, policy=live)

        synthetic_eval = default_offline_policy().model_copy(
            update={"run_purpose": "synthetic_eval"}
        )
        with pytest.raises(UnsupportedApplicationPolicy, match="labels/evidence"):
            TriageApplicationService(store, policy=synthetic_eval)


def test_submitted_facts_lookup_is_field_scoped_hashed_and_not_root_cause(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        service = TriageApplicationService(store)
        result = service.submit(_unknown_state(), idempotency_key="evidence-key")

        assertion = service.lookup_submitted_facts_evidence(
            result.envelope.request_id, result.envelope.result_version, "assertion"
        )
        log = service.lookup_submitted_facts_evidence(
            result.envelope.request_id, result.envelope.result_version, "syslog_excerpt"
        )
        env_probe = service.lookup_submitted_facts_evidence(
            result.envelope.request_id, result.envelope.result_version, "env.AP_reachable"
        )

        expected_hash = hashlib.sha256(
            json.dumps(assertion.value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert assertion.field_path == "assertion"
        assert assertion.content_sha256 == expected_hash
        assert assertion.capture_scope == "submitted_field"
        assert assertion.root_cause_verified is False
        assert log.capture_scope == "provided_excerpt"
        assert log.value == "no matching cause"
        assert env_probe.field_path == "env.AP_reachable"
        assert env_probe.value is True
        ref = next(
            item for item in result.envelope.evidence_refs if item.field_path == env_probe.field_path
        )
        assert ref.content_sha256 == env_probe.content_sha256
        assert ref.evidence_kind == "observed"
        for linked in result.envelope.evidence_refs:
            value = service.lookup_submitted_facts_evidence(
                result.envelope.request_id,
                result.envelope.result_version,
                linked.field_path,
            )
            assert value.content_sha256 == linked.content_sha256
        with pytest.raises(SubmittedFactNotFound):
            service.lookup_submitted_facts_evidence(
                result.envelope.request_id, result.envelope.result_version, "provenance"
            )


def test_evidence_lookup_supports_correction_current_version_and_stale_parent(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        service = TriageApplicationService(store)
        result = service.submit(_unknown_state(), idempotency_key="review-key")
        before = result.envelope.result
        after = TriageResult(
            category="PRODUCT_DEFECT",
            action="ASSIGN_PRODUCT_DEVELOPER",
            priority=4,
            confidence=0.9,
            reason="Reviewer correction from submitted facts",
        )
        from jevtest.execution_contracts import ReviewRecordV1

        store.append_review(
            ReviewRecordV1(
                schema_version="review-record-v1",
                review_id="review-correct",
                request_id=result.envelope.request_id,
                result_version=result.envelope.result_version,
                actor_type="local_user",
                actor_origin="tty:test-asserted-correction",
                assistance_mode="unaided",
                decision="correct",
                before_result=before,
                after_result=after,
                reason="test correction with explicit reason",
                reviewed_at="2026-09-27T12:10:00Z",
            )
        )
        current = service.get_result(result.envelope.request_id)

        assert current.result == after
        assert current.result_version != result.envelope.result_version
        report_record = service.get_run_report(current.request_id)
        assert report_record is not None
        assert report_record[1] == result.envelope.result_version
        assert service.get_run_report(current.request_id, current.result_version) is None
        evidence = service.lookup_submitted_facts_evidence(
            current.request_id, current.result_version, "assertion"
        )
        assert evidence.value == "association did not complete"
        replay = service.submit(_unknown_state(), idempotency_key="review-key")
        assert replay.replayed is True
        assert replay.envelope == current
        assert replay.report.accuracy is None


def test_result_lookup_is_version_scoped(tmp_path):
    with LocalStore(tmp_path / "triage.sqlite3") as store:
        service = TriageApplicationService(store)
        result = service.submit(_unknown_state(), idempotency_key="lookup-key")

        assert service.get_result(result.envelope.request_id, result.envelope.result_version) == result.envelope
        assert service.get_result(result.envelope.request_id, "missing-version") is None


def test_report_and_provider_metadata_survive_store_reopen(tmp_path):
    db_path = tmp_path / "persistent-report.sqlite3"
    with LocalStore(db_path) as store:
        service = TriageApplicationService(store)
        result = service.submit(_unknown_state(), idempotency_key="persistent-report")
        request_id = result.envelope.request_id
        result_version = result.envelope.result_version
        assert store.get_run_report(request_id, result_version) is not None
        assert store.get_provider_execution_metadata(request_id, result_version) is not None

    with LocalStore(db_path) as reopened:
        restored = reopened.get_run_report(request_id, result_version)
        metadata = reopened.get_provider_execution_metadata(request_id, result_version)
        assert restored is not None and restored[1] == result_version
        assert metadata is not None and metadata[1] == result_version
