from __future__ import annotations

import json
from pathlib import Path

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from jevtest.app_service import TriageApplicationService
from jevtest.http_api import _parse_server_args, create_app
from jevtest.local_store import LocalStore
from jevtest.models import FailureState, Provenance, TriageResult

_HOST = "127.0.0.1:8765"
_ORIGIN = f"http://{_HOST}"


def _state() -> FailureState:
    return FailureState(
        case_id="SYN-HTTP-001",
        result="FAIL",
        previous_runs=["PASS", "PASS", "PASS"],
        device="AP-660",
        sw_version="V6R1",
        assertion="association did not complete",
        syslog_excerpt="no matching cause",
        env={"AP_reachable": True, "AC_reachable": True},
        retry="failed",
        provenance=Provenance(dataset_version="synthetic-v1", synthetic=True),
    )


@pytest.fixture
def api(tmp_path):
    with LocalStore(tmp_path / "http.sqlite3") as store:
        service = TriageApplicationService(store, idempotency_scope="http-test")
        app = create_app(service, expected_host=_HOST)
        with TestClient(
            app,
            base_url=_ORIGIN,
            client=("127.0.0.1", 54321),
        ) as client:
            session = client.get("/v1/session")
            assert session.status_code == 200
            yield client, service, session.json()["csrf_token"]


def _headers(token: str, key: str, **extra: str) -> dict[str, str]:
    return {
        "Origin": _ORIGIN,
        "X-CSRF-Token": token,
        "Idempotency-Key": key,
        **extra,
    }


def _submit(client: TestClient, token: str, key: str = "triage-001"):
    return client.post(
        "/v1/triage",
        json={"failure_state": _state().model_dump(mode="json")},
        headers=_headers(token, key),
    )


def test_health_session_and_no_cors(api):
    client, _service, _token = api
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert page.headers["cache-control"] == "no-store"
    assert "本地分诊审核" in page.text
    assert "REQ-DEMO-0042" not in page.text
    script = client.get("/assets/review.js")
    assert script.status_code == 200
    assert "Idempotency-Key" in script.text
    assert "localStorage" not in script.text
    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok", "profile": "offline", "execution_mode": "offline",
        "run_purpose": "demo", "provider_mode": "mock",
    }
    assert "access-control-allow-origin" not in health.headers


def test_submit_is_offline_idempotent_and_result_is_readable(api):
    client, _service, token = api
    first = _submit(client, token)
    replay = _submit(client, token)

    assert first.status_code == 201
    assert replay.status_code == 200
    payload = first.json()
    replay_payload = replay.json()
    envelope = payload["envelope"]
    assert envelope["execution_mode"] == "offline"
    assert envelope["run_purpose"] == "demo"
    assert all(a["provider_mock"] for a in envelope["attempts"] if a["stage"] in {"jev", "llm"})
    assert all("error_summary" not in attempt for attempt in envelope["attempts"])
    assert payload["report"]["accuracy"] is None
    assert replay_payload["replayed"] is True
    assert replay_payload["envelope"]["request_id"] == envelope["request_id"]
    assert first.headers["etag"] == f'"{envelope["result_version"]}"'
    direct_capture = {
        "http_status": first.status_code,
        "replayed": payload["replayed"],
        "request_id": envelope["request_id"],
        "etag": first.headers["etag"],
    }
    assert direct_capture == {
        "http_status": 201,
        "replayed": False,
        "request_id": envelope["request_id"],
        "etag": f'"{envelope["result_version"]}"',
    }
    assert payload["report_result_version"] == envelope["result_version"]

    fetched = client.get(f"/v1/triage/{envelope['request_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["envelope"]["result_version"] == envelope["result_version"]
    report = client.get(f"/v1/triage/{envelope['request_id']}/report")
    assert report.status_code == 200
    assert report.json()["report_result_version"] == envelope["result_version"]
    assert report.json()["report"] == payload["report"]
    metadata = client.get(f"/v1/triage/{envelope['request_id']}/execution-metadata")
    assert metadata.status_code == 200
    assert metadata.json()["metadata_result_version"] == envelope["result_version"]
    assert {row["stage"] for row in metadata.json()["items"]} == {"jev", "llm"}
    assert client.get("/v1/triage/missing-request").status_code == 404


def test_evidence_is_field_scoped_and_not_root_cause_evidence(api):
    client, _service, token = api
    envelope = _submit(client, token).json()["envelope"]
    path = f"/v1/triage/{envelope['request_id']}/evidence"
    response = client.get(
        path,
        params={"result_version": envelope["result_version"], "field_path": "syslog_excerpt"},
    )
    assert response.status_code == 200
    evidence = response.json()
    assert evidence["value"] == "no matching cause"
    assert evidence["kind"] == "submitted_fact"
    assert evidence["capture_scope"] == "provided_excerpt"
    assert evidence["root_cause_verified"] is False

    rejected = client.get(
        path,
        params={"result_version": envelope["result_version"], "field_path": "provenance.source"},
    )
    assert rejected.status_code == 404
    assert rejected.json() == {"error": {"code": "evidence_not_found"}}

    facts = client.get(
        f"/v1/triage/{envelope['request_id']}/facts",
        params={"result_version": envelope["result_version"]},
    )
    assert facts.status_code == 200
    assert facts.json()["facts"]["assertion"] == "association did not complete"
    assert "case_id" not in facts.json()["facts"]
    assert "provenance" not in facts.json()["facts"]


def test_review_is_server_authored_confirmed_and_idempotent(api):
    client, _service, token = api
    envelope = _submit(client, token).json()["envelope"]
    body = {
        "decision": "accept",
        "operator_label": "operator-self-asserted",
        "confirmation": f"accept {envelope['request_id'][:8]}",
    }
    headers = _headers(
        token,
        "review-001",
        **{"If-Match": f'"{envelope["result_version"]}"'},
    )
    response = client.post(
        f"/v1/triage/{envelope['request_id']}/reviews", json=body, headers=headers
    )
    assert response.status_code == 200
    review = response.json()["review"]
    assert review["actor_type"] == "local_user"
    assert review["actor_origin"] == "operator-self-asserted"
    assert review["assistance_mode"] == "model_assisted"
    assert review["reviewed_at"]
    assert response.json()["actor_identity_verified"] is False
    assert response.json()["envelope"]["review_status"] == "accepted"

    replay = client.post(
        f"/v1/triage/{envelope['request_id']}/reviews", json=body, headers=headers
    )
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert replay.json()["review"]["review_id"] == review["review_id"]

    changed = {**body, "operator_label": "different-claim"}
    conflict = client.post(
        f"/v1/triage/{envelope['request_id']}/reviews", json=changed, headers=headers
    )
    assert conflict.status_code == 409
    assert conflict.json() == {"error": {"code": "idempotency_conflict"}}


def test_corrected_result_is_reachable_from_queue_after_restart(tmp_path):
    database = tmp_path / "triage.sqlite3"
    with LocalStore(database) as store:
        service = TriageApplicationService(store, idempotency_scope="queue-restart-test")
        with TestClient(
            create_app(service, expected_host=_HOST),
            base_url=_ORIGIN,
            client=("127.0.0.1", 54321),
        ) as client:
            token = client.get("/v1/session").json()["csrf_token"]
            original = _submit(client, token, "queue-restart-submit").json()["envelope"]
            assert original["requires_human_review"] is True
            initial_queue = client.get("/v1/review-queue").json()
            assert [item["envelope"]["result_version"] for item in initial_queue["items"]] == [
                original["result_version"]
            ]
            assert initial_queue["items"][0]["handoff"]["result_version"] == original[
                "result_version"
            ]

            corrected = client.post(
                f"/v1/triage/{original['request_id']}/reviews",
                json={
                    "decision": "correct",
                    "operator_label": "test-reviewer",
                    "confirmation": f"correct {original['request_id'][:8]}",
                    "reason": "The submitted failure evidence supports this classification.",
                    "after_result": {
                        "category": "PRODUCT_DEFECT",
                        "action": "ASSIGN_PRODUCT_DEVELOPER",
                        "priority": 4,
                        "confidence": 0.0,
                        "reason": "Reviewer correction based on submitted evidence.",
                    },
                },
                headers=_headers(
                    token,
                    "queue-restart-correct",
                    **{"If-Match": f'"{original["result_version"]}"'},
                ),
            )
            assert corrected.status_code == 200
            corrected_envelope = corrected.json()["envelope"]
            correction_event = corrected.json()["review"]
            assert correction_event["confidence_origin"] == "inherited_unreassessed"
            assert correction_event["after_result"]["confidence"] == original["result"]["confidence"]
            assert corrected_envelope["result_version"] != original["result_version"]
            assert corrected_envelope["review_status"] == "pending"

    # Reopen the same DB and prove the actual local todo entrance addresses the new head.
    with LocalStore(database) as reopened:
        service = TriageApplicationService(reopened, idempotency_scope="queue-restart-test")
        with TestClient(
            create_app(service, expected_host=_HOST),
            base_url=_ORIGIN,
            client=("127.0.0.1", 54322),
        ) as client:
            token = client.get("/v1/session").json()["csrf_token"]
            queue = client.get("/v1/review-queue")
            assert queue.status_code == 200
            items = queue.json()["items"]
            assert len(items) == 1
            current = items[0]["envelope"]
            assert current["request_id"] == original["request_id"]
            assert current["result_version"] == corrected_envelope["result_version"]
            assert current["delivery_status"] == "pending"
            assert items[0]["handoff"] is None

            history = client.get(
                f"/v1/triage/{current['request_id']}/reviews",
                params={"result_version": original["result_version"]},
            )
            assert history.status_code == 200
            assert len(history.json()["reviews"]) == 1
            assert history.json()["reviews"][0]["decision"] == "correct"
            assert history.json()["current_result_version"] == current["result_version"]
            assert history.json()["actor_identity_verified"] is False

            accepted = client.post(
                f"/v1/triage/{current['request_id']}/reviews",
                json={
                    "decision": "accept",
                    "operator_label": "test-reviewer",
                    "confirmation": f"accept {current['request_id'][:8]}",
                },
                headers=_headers(
                    token,
                    "queue-restart-accept",
                    **{"If-Match": f'"{current["result_version"]}"'},
                ),
            )
            assert accepted.status_code == 200
            assert accepted.json()["envelope"]["review_status"] == "accepted"
            assert client.get("/v1/review-queue").json() == {"items": [], "count": 0}


def test_review_requires_confirmation_reason_and_server_owned_actor_fields(api):
    client, _service, token = api
    envelope = _submit(client, token).json()["envelope"]
    path = f"/v1/triage/{envelope['request_id']}/reviews"
    headers = _headers(
        token,
        "review-invalid",
        **{"If-Match": f'"{envelope["result_version"]}"'},
    )

    bad_confirmation = client.post(
        path,
        json={
            "decision": "accept",
            "operator_label": "operator",
            "confirmation": "accept wrong-id",
        },
        headers=headers,
    )
    assert bad_confirmation.status_code == 422
    assert bad_confirmation.json() == {"error": {"code": "confirmation_mismatch"}}

    untrusted_actor = client.post(
        path,
        json={
            "decision": "accept",
            "operator_label": "operator",
            "confirmation": f"accept {envelope['request_id'][:8]}",
            "actor_type": "local_user",
            "reviewer_id": "owner",
        },
        headers=_headers(
            token,
            "review-extra",
            **{"If-Match": f'"{envelope["result_version"]}"'},
        ),
    )
    assert untrusted_actor.status_code == 422
    assert untrusted_actor.json() == {"error": {"code": "invalid_request"}}
    assert "owner" not in untrusted_actor.text

    invalid_correction = client.post(
        path,
        json={
            "decision": "correct",
            "operator_label": "operator",
            "confirmation": f"correct {envelope['request_id'][:8]}",
        },
        headers=_headers(
            token,
            "review-no-correction",
            **{"If-Match": f'"{envelope["result_version"]}"'},
        ),
    )
    assert invalid_correction.status_code == 422
    assert invalid_correction.json() == {"error": {"code": "invalid_request"}}


def test_correct_cannot_relabel_existing_model_confidence_as_manual(api):
    client, _service, token = api
    envelope = _submit(client, token).json()["envelope"]
    original_confidence = envelope["result"]["confidence"]
    changed_confidence = 0.123 if original_confidence != 0.123 else 0.124
    response = client.post(
        f"/v1/triage/{envelope['request_id']}/reviews",
        json={
            "decision": "correct",
            "operator_label": "test-reviewer",
            "confirmation": f"correct {envelope['request_id'][:8]}",
            "reason": "Classification changed from submitted facts.",
            "after_result": {
                "category": "PRODUCT_DEFECT",
                "action": "ASSIGN_PRODUCT_DEVELOPER",
                "priority": 4,
                "confidence": changed_confidence,
                "reason": "Reviewer correction from submitted facts.",
            },
        },
        headers=_headers(
            token,
            "changed-confidence-correction",
            **{"If-Match": f'"{envelope["result_version"]}"'},
        ),
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "confidence_reassessment_not_supported"}}
    current = client.get(f"/v1/triage/{envelope['request_id']}").json()["envelope"]
    assert current["result_version"] == envelope["result_version"]


def test_stale_if_match_and_review_state_conflict(api):
    client, _service, token = api
    envelope = _submit(client, token).json()["envelope"]
    body = {
        "decision": "defer",
        "operator_label": "operator",
        "confirmation": f"defer {envelope['request_id'][:8]}",
        "reason": "need one more log excerpt",
    }
    headers = _headers(
        token,
        "review-defer",
        **{"If-Match": '"stale-version"'},
    )
    stale = client.post(
        f"/v1/triage/{envelope['request_id']}/reviews", json=body, headers=headers
    )
    assert stale.status_code == 409
    assert stale.json() == {"error": {"code": "result_version_conflict"}}

    headers["If-Match"] = f'"{envelope["result_version"]}"'
    corrected = client.post(
        f"/v1/triage/{envelope['request_id']}/reviews",
        json={
            "decision": "correct",
            "operator_label": "operator",
            "confirmation": f"correct {envelope['request_id'][:8]}",
            "reason": "reviewer changed the category",
            "after_result": {
                "category": "PRODUCT_DEFECT",
                "action": "ASSIGN_PRODUCT_DEVELOPER",
                "priority": 4,
                "confidence": envelope["result"]["confidence"],
                "reason": "Correction from submitted facts",
            },
        },
        headers=headers,
    )
    assert corrected.status_code == 200
    assert corrected.json()["envelope"]["result_version"] != envelope["result_version"]
    assert corrected.json()["review"]["confidence_origin"] == "inherited_unreassessed"

    headers["Idempotency-Key"] = "review-after-accept"
    stale_head = client.post(
        f"/v1/triage/{envelope['request_id']}/reviews", json=body, headers=headers
    )
    assert stale_head.status_code == 409
    assert stale_head.json() == {"error": {"code": "result_version_conflict"}}


def test_host_origin_csrf_loopback_and_request_size_boundaries(api, tmp_path):
    client, _service, token = api
    state_json = _state().model_dump(mode="json")

    wrong_host = client.get("/v1/health", headers={"Host": "evil.example:8765"})
    assert wrong_host.status_code == 403
    assert wrong_host.json() == {"error": {"code": "host_rejected"}}

    bad_origin = client.post(
        "/v1/triage",
        json={"failure_state": state_json},
        headers={**_headers(token, "bad-origin"), "Origin": "http://evil.example"},
    )
    assert bad_origin.status_code == 403
    assert bad_origin.json() == {"error": {"code": "origin_rejected"}}

    no_csrf = client.post(
        "/v1/triage",
        json={"failure_state": state_json},
        headers={"Origin": _ORIGIN, "Idempotency-Key": "no-csrf"},
    )
    assert no_csrf.status_code == 403
    assert no_csrf.json() == {"error": {"code": "csrf_rejected"}}

    marker = "DO_NOT_ECHO_THIS_VALUE"
    oversized = client.post(
        "/v1/triage",
        content=json.dumps({"failure_state": state_json, "padding": marker + "x" * (65 * 1024)}),
        headers={
            **_headers(token, "too-large"),
            "Content-Type": "application/json",
        },
    )
    assert oversized.status_code == 413
    assert oversized.json() == {"error": {"code": "request_too_large"}}
    assert marker not in oversized.text

    def streamed_body():
        yield b'{"failure_state":'
        yield b'"' + b"x" * (65 * 1024) + b'"}'

    streamed = client.post(
        "/v1/triage",
        content=streamed_body(),
        headers={
            **_headers(token, "streamed-too-large"),
            "Content-Type": "application/json",
        },
    )
    assert streamed.status_code == 413
    assert streamed.json() == {"error": {"code": "request_too_large"}}

    with LocalStore(tmp_path / "remote.sqlite3") as remote_store:
        remote_app = create_app(
            TriageApplicationService(remote_store),
            expected_host=_HOST,
        )
        with TestClient(
            remote_app,
            base_url=_ORIGIN,
            client=("192.0.2.10", 54321),
        ) as remote_client:
            remote = remote_client.get("/v1/health")
    assert remote.status_code == 403
    assert remote.json() == {"error": {"code": "loopback_only"}}


def test_invalid_host_configuration_rejected(tmp_path):
    with LocalStore(tmp_path / "bad-host.sqlite3") as store:
        service = TriageApplicationService(store)
        for host in ("0.0.0.0:8000", "localhost:8000", "127.0.0.1"):
            with pytest.raises(ValueError, match="loopback IP literal"):
                create_app(service, expected_host=host)


def test_module_server_args_are_loopback_only_and_default_to_user_store(tmp_path):
    host, port, expected_host, store = _parse_server_args([])
    assert (host, port, expected_host) == ("127.0.0.1", 8765, "127.0.0.1:8765")
    assert store == Path.home() / ".jevtest" / "triage.sqlite3"

    host6, port6, expected6, custom_store = _parse_server_args(
        ["--host", "::1", "--port", "9001", "--store", "~/custom.sqlite3"]
    )
    assert (host6, port6, expected6) == ("::1", 9001, "[::1]:9001")
    assert str(custom_store).endswith("custom.sqlite3")

    for argv in (["--host", "0.0.0.0"], ["--host", "localhost"], ["--port", "0"]):
        with pytest.raises(SystemExit):
            _parse_server_args(list(argv))
