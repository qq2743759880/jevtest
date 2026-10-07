from __future__ import annotations

import pytest

from jevtest.live_smoke_capture import capture_live_advisory_smoke
from jevtest.models import FailureState, Provenance


class _Response:
    def __init__(self, status_code: int, body: dict, headers: dict | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._body


class _Client:
    def __init__(self):
        self.posts = 0

    def get(self, path: str):
        if path == "/v1/health":
            return _Response(200, {
                "status": "ok", "profile": "live-advisory", "execution_mode": "live",
                "run_purpose": "advisory", "provider_mode": "live",
            })
        if path == "/v1/session":
            return _Response(200, {"csrf_token": "test-csrf"})
        if path.endswith("/execution-metadata"):
            return _Response(200, {"items": []})
        if path.endswith("/report"):
            return _Response(200, {"report": {"schema_version": "run-report-v2"}})
        return _Response(200, {})

    def post(self, path: str, *, headers: dict, json: dict):
        self.posts += 1
        assert path == "/v1/triage"
        assert headers["Origin"] == "http://127.0.0.1:8766"
        assert headers["X-CSRF-Token"] == "test-csrf"
        assert json["live_confirmed"] is True
        return _Response(
            201,
            {
                "replayed": False,
                "envelope": {
                    "request_id": "app-owner-test",
                    "result_version": "result-v1",
                    "attempts": [
                        {"stage": "jev", "attempt_index": 1},
                        {"stage": "llm", "attempt_index": 1},
                    ],
                },
            },
            {"ETag": '"result-v1"'},
        )


def _state() -> FailureState:
    return FailureState(
        case_id="OWNER-EXP-TEST",
        result="FAIL",
        previous_runs=["PASS"],
        device="SYNTH-AP-01",
        sw_version="synthetic-build-1",
        assertion="synthetic failure",
        syslog_excerpt="synthetic only",
        env={"AP_reachable": True},
        retry="failed",
        provenance=Provenance(
            dataset_version="owner-experience-test-v1.1",
            synthetic=True,
            source="owner_experience_test",
        ),
    )


def test_smoke_capture_records_direct_http_fields_and_marks_automation():
    client = _Client()
    evidence = capture_live_advisory_smoke(
        client,
        origin="http://127.0.0.1:8766",
        failure_state=_state(),
        idempotency_key="owner-smoke-test-key",
    )

    assert client.posts == 1
    assert evidence["submission_http"] == {
        "http_status": 201,
        "replayed": False,
        "request_id": "app-owner-test",
        "etag": '"result-v1"',
        "capture_source": "direct_http_response",
    }
    assert evidence["readbacks"]["report"]["http_status"] == 200
    assert [item["stage"] for item in evidence["application_attempts"]] == ["jev", "llm"]
    assert [item["attempt_index"] for item in evidence["application_attempts"]] == [1, 1]
    assert evidence["wire_count"] == "unknown"
    assert evidence["automation_smoke"] is True
    assert evidence["actor_identity_verified"] is False
    assert evidence["counts_toward_human_acceptance"] is False
    assert evidence["formal_annotation"] is False
    assert evidence["screenshot_persistence"] == "unavailable"


def test_smoke_capture_refuses_non_live_service_before_post():
    client = _Client()
    original_get = client.get
    client.get = lambda path: (
        _Response(200, {"status": "ok", "profile": "offline"})
        if path == "/v1/health"
        else original_get(path)
    )

    with pytest.raises(RuntimeError, match="not the live-advisory profile"):
        capture_live_advisory_smoke(
            client, origin="http://127.0.0.1:8766", failure_state=_state()
        )
    assert client.posts == 0
