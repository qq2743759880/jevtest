from __future__ import annotations

import json
from pathlib import Path

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from jevtest.app_service import (
    LiveSubmissionConfirmationRequired,
    TriageApplicationService,
    UnsupportedApplicationPolicy,
    default_live_advisory_policy,
)
from jevtest.cascade import run_cascade
from jevtest.execution_contracts import RoutingPolicyV1
from jevtest.http_api import _parse_server_options, create_app
from jevtest.local_store import LocalStore
from jevtest.models import FailureState, Provenance, TriageResult
from jevtest.providers.base import Provider
from jevtest.providers.rule import RuleProvider
from jevtest.runtime_authorization import (
    LiveAuthorizationError,
    LiveRuntimeAuthorization,
    authorize_live_advisory,
)

HOST = "127.0.0.1:9876"
ORIGIN = f"http://{HOST}"
SYNTHETIC_FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "live-advisory-smoke.failure.json"


def _state() -> FailureState:
    return FailureState.model_validate(json.loads(SYNTHETIC_FIXTURE.read_text(encoding="utf-8")))


class FakeLiveProvider(Provider):
    is_mock = False

    def __init__(self, arm_name: str, result: TriageResult, *, fail: str | None = None) -> None:
        self.arm_name = arm_name
        self.provider_id = "typesafe" if arm_name == "jev" else "ark"
        self.model_id = "fake-live-model"
        self.result = result
        self.fail = fail
        self.calls = 0

    def predict(self, state: FailureState) -> TriageResult:
        return self.predict_with_budget(state, 1)

    def predict_with_budget(self, state: FailureState, max_attempts: int) -> TriageResult:
        self.calls += 1
        assert max_attempts == 1
        if self.fail:
            self.record_attempt(
                status="failed",
                latency_ms=2,
                pricing_status="unknown",
                error_code=self.fail,
                error_summary=self.fail,
            )
            raise TimeoutError(self.fail)
        if self.arm_name == "jev":
            self.record_attempt(
                status="succeeded", latency_ms=2, input_tokens=11,
                cost=0.000001, cost_currency="USD", pricing_status="known",
            )
        else:
            self.record_attempt(
                status="succeeded", latency_ms=3, input_tokens=12, output_tokens=8,
                pricing_status="unknown",
            )
        return self.result


def _results() -> tuple[TriageResult, TriageResult]:
    return (
        TriageResult(
            category="FLAKY", action="RERUN", priority=2, confidence=0.98,
            reason="Jev suggestion from the synthetic fixture.",
        ),
        TriageResult(
            category="PRODUCT_DEFECT", action="ASSIGN_PRODUCT_DEVELOPER", priority=4,
            confidence=0.99, reason="Strong LLM suggestion from the synthetic fixture.",
        ),
    )


def _service(store: LocalStore, *, jev_fail: str | None = None, llm_fail: str | None = None):
    jev_result, llm_result = _results()
    providers = {
        "jev": FakeLiveProvider("jev", jev_result, fail=jev_fail),
        "llm": FakeLiveProvider("llm", llm_result, fail=llm_fail),
    }
    auth = authorize_live_advisory(explicit_opt_in=True)
    service = TriageApplicationService(
        store,
        idempotency_scope="live-advisory-test",
        execution_profile="live-advisory",
        live_authorization=auth,
        live_provider_factory=lambda _authorization: providers,
    )
    return service, providers, auth


def test_credentials_never_change_the_offline_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-test-key")
    monkeypatch.setenv("ARK_API_KEY", "offline-test-key")
    with LocalStore(tmp_path / "offline.sqlite3") as store:
        service = TriageApplicationService(store)
        result = service.submit(_state(), idempotency_key="offline-with-credentials")

    assert service.execution_profile == "offline"
    assert service.provider_mode == "mock"
    assert result.envelope.execution_mode == "offline"
    assert all(attempt.provider_mock for attempt in result.envelope.attempts if attempt.stage in {"jev", "llm"})


def test_live_profile_requires_runtime_opt_in_provider_allowlist_and_budgets(tmp_path):
    with pytest.raises(LiveAuthorizationError, match="explicit"):
        authorize_live_advisory(explicit_opt_in=False)
    with pytest.raises(LiveAuthorizationError, match="allowlist"):
        LiveRuntimeAuthorization(
            profile="live-advisory", explicit_opt_in=True,
            provider_allowlist=frozenset({"jev"}), jev_attempt_budget=1,
            llm_attempt_budget=1, deadline_ms=30_000,
        )
    with pytest.raises(LiveAuthorizationError, match="one attempt"):
        LiveRuntimeAuthorization(
            profile="live-advisory", explicit_opt_in=True,
            provider_allowlist=frozenset({"jev", "llm"}), jev_attempt_budget=2,
            llm_attempt_budget=1, deadline_ms=30_000,
        )
    with LocalStore(tmp_path / "unauthorized.sqlite3") as store:
        with pytest.raises(UnsupportedApplicationPolicy, match="runtime authorization"):
            TriageApplicationService(
                store, policy=default_live_advisory_policy(), execution_profile="live-advisory"
            )


def test_live_policy_rejects_auto_approval_or_direct_rule_output(tmp_path):
    policy = default_live_advisory_policy().model_copy(
        update={"jev_high_threshold": 0.9}
    )
    auth = authorize_live_advisory(explicit_opt_in=True)
    with LocalStore(tmp_path / "bad-policy.sqlite3") as store:
        with pytest.raises(UnsupportedApplicationPolicy, match="human review"):
            TriageApplicationService(
                store,
                policy=policy,
                execution_profile="live-advisory",
                live_authorization=auth,
                live_provider_factory=lambda _auth: {},
            )


def test_live_confirmed_request_reaches_human_and_persists_real_attempt_ledger(tmp_path):
    with LocalStore(tmp_path / "live.sqlite3") as store:
        service, providers, auth = _service(store)
        with pytest.raises(LiveSubmissionConfirmationRequired):
            service.submit(_state(), idempotency_key="needs-confirmation")
        assert providers["jev"].calls == providers["llm"].calls == 0

        result = service.submit(
            _state(), idempotency_key="live-r4", live_confirmed=True
        )

        envelope = result.envelope
        assert envelope.execution_mode == "live"
        assert envelope.run_purpose == "advisory"
        assert envelope.run_status == "result_ready"
        assert envelope.terminal_stage == "human"
        assert envelope.requires_human_review is True
        assert envelope.delivery_status == "local_queued"
        assert [step.stage for step in envelope.route_steps] == ["rule", "jev", "llm", "human"]
        assert envelope.route_steps[0].reason_code == "escalate_rule_result"
        assert envelope.attempts and len(envelope.attempts) == 3
        assert sum(1 for row in envelope.attempts if row.stage == "jev") == auth.jev_attempt_budget
        assert sum(1 for row in envelope.attempts if row.stage == "llm") == auth.llm_attempt_budget
        assert all(not row.provider_mock for row in envelope.attempts if row.stage in {"jev", "llm"})
        assert envelope.result.category == "PRODUCT_DEFECT"
        assert envelope.result.confidence == 0.99
        assert envelope.pricing_status == "unknown"
        assert envelope.total_cost is None
        assert result.report.run_purpose == "advisory"
        assert result.report.accuracy is None
        assert result.report.pricing_status == "unknown"
        assert result.report.total_cost is None
        assert service.get_result(envelope.request_id) == envelope
        assert service.get_result_by_key("live-r4") == envelope
        assert providers["jev"].calls == providers["llm"].calls == 1


@pytest.mark.parametrize(
    ("jev_fail", "llm_fail", "expected_category", "expected_failed_stage"),
    [
        ("timeout", None, "PRODUCT_DEFECT", "jev"),
        (None, "http_429", "FLAKY", "llm"),
    ],
)
def test_provider_failures_are_degraded_and_do_not_stop_the_next_stage(
    tmp_path, jev_fail, llm_fail, expected_category, expected_failed_stage
):
    with LocalStore(tmp_path / f"failure-{expected_failed_stage}.sqlite3") as store:
        service, providers, _auth = _service(store, jev_fail=jev_fail, llm_fail=llm_fail)
        result = service.submit(
            _state(), idempotency_key="provider-failure", live_confirmed=True
        )

    envelope = result.envelope
    failed = next(step for step in envelope.route_steps if step.status == "failed")
    assert failed.stage == expected_failed_stage
    assert envelope.run_status == "degraded"
    assert envelope.terminal_stage == "human"
    assert envelope.requires_human_review is True
    assert envelope.result.category == expected_category
    assert providers["jev"].calls == providers["llm"].calls == 1
    assert envelope.pricing_status == "unknown"
    assert envelope.total_cost is None


def test_runtime_deadline_skips_later_provider_without_exceeding_budget(tmp_path, monkeypatch):
    import jevtest.cascade as cascade

    clock = iter([0.0, 0.0, 0.0, 0.002])
    monkeypatch.setattr(cascade, "perf_counter", lambda: next(clock))
    with LocalStore(tmp_path / "deadline.sqlite3") as store:
        auth = authorize_live_advisory(explicit_opt_in=True, deadline_ms=1)
        jev_result, llm_result = _results()
        providers = {
            "jev": FakeLiveProvider("jev", jev_result),
            "llm": FakeLiveProvider("llm", llm_result),
        }
        envelope = run_cascade(
            _state(),
            default_live_advisory_policy(deadline_ms=1),
            {"rule": RuleProvider(), **providers},
            request_id="deadline-test",
            result_version="deadline-test-v1",
            runtime_authorization=auth,
        )

    assert envelope.run_status == "degraded"
    assert envelope.route_steps[-1].reason_code == "deadline_exceeded"
    assert [step.stage for step in envelope.route_steps] == ["rule", "jev", "human"]
    assert providers["jev"].calls == 1
    assert providers["llm"].calls == 0


def test_live_initialization_fails_closed_and_redacts_provider_errors(tmp_path, monkeypatch):
    import jevtest.providers.jev_live as jev_live
    import jevtest.providers.llm_judge as llm_judge

    monkeypatch.setattr(jev_live, "_load_env_file", lambda: None)
    monkeypatch.setattr(jev_live, "validate_https_url", lambda url: url)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    auth = authorize_live_advisory(explicit_opt_in=True)
    with LocalStore(tmp_path / "missing-key.sqlite3") as store:
        with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY, ARK_API_KEY"):
            TriageApplicationService(
                store, execution_profile="live-advisory", live_authorization=auth
            )

    secret = "fake-test-secret-that-must-not-escape"
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)
    monkeypatch.setenv("ARK_API_KEY", secret)

    class ReadyJev:
        arm_name = "jev"
        is_mock = False
        def close(self):
            self.closed = True

    def reject_ark(**_kwargs):
        raise ValueError(secret)

    monkeypatch.setattr(jev_live, "JevLiveProvider", lambda **_kwargs: ReadyJev(), raising=False)
    monkeypatch.setattr(llm_judge, "LLMJudgeProvider", reject_ark, raising=False)
    with LocalStore(tmp_path / "bad-provider.sqlite3") as store:
        with pytest.raises(RuntimeError, match="ValueError") as error:
            TriageApplicationService(
                store, execution_profile="live-advisory", live_authorization=auth
            )
    assert secret not in str(error.value)


def test_live_health_and_http_submit_require_explicit_confirmation(tmp_path):
    with LocalStore(tmp_path / "live-http.sqlite3") as store:
        service, providers, _auth = _service(store)
        with TestClient(
            create_app(service, expected_host=HOST), base_url=ORIGIN,
            client=("127.0.0.1", 54321),
        ) as client:
            session = client.get("/v1/session")
            token = session.json()["csrf_token"]
            headers = {
                "Origin": ORIGIN,
                "X-CSRF-Token": token,
                "Idempotency-Key": "live-http-key",
            }
            health = client.get("/v1/health").json()
            assert health == {
                "status": "ok", "profile": "live-advisory", "execution_mode": "live",
                "run_purpose": "advisory", "provider_mode": "live",
            }
            assert "API_KEY" not in json.dumps(health)
            assert "fake-live-model" not in json.dumps(health)

            unconfirmed = client.post(
                "/v1/triage", json={"failure_state": _state().model_dump(mode="json")},
                headers=headers,
            )
            assert unconfirmed.status_code == 400
            assert unconfirmed.json() == {"error": {"code": "live_confirmation_required"}}
            assert providers["jev"].calls == providers["llm"].calls == 0

            confirmed = client.post(
                "/v1/triage",
                json={"failure_state": _state().model_dump(mode="json"), "live_confirmed": True},
                headers=headers,
            )
            assert confirmed.status_code == 201
            assert confirmed.json()["envelope"]["execution_mode"] == "live"
            assert providers["jev"].calls == providers["llm"].calls == 1
            script = client.get("/assets/review.js").text
            page = client.get("/").text
            assert "window.confirm" in script
            assert "live_confirmed: liveConfirmed" in script
            assert 'id="live-warning"' in page and "可能产生费用" in page


def test_offline_health_and_page_keep_live_warning_hidden(tmp_path):
    with LocalStore(tmp_path / "offline-http.sqlite3") as store:
        service = TriageApplicationService(store)
        with TestClient(
            create_app(service, expected_host=HOST), base_url=ORIGIN,
            client=("127.0.0.1", 54321),
        ) as client:
            health = client.get("/v1/health").json()
            assert health == {
                "status": "ok", "profile": "offline", "execution_mode": "offline",
                "run_purpose": "demo", "provider_mode": "mock",
            }
            page = client.get("/").text
            assert 'id="live-warning"' in page
            assert 'id="live-warning" class="inline-note live-warning" hidden' in page
            script = client.get("/assets/review.js").text
            assert "state.executionProfile = \"offline\"" in script


def test_http_entrypoint_requires_explicit_live_profile_opt_in():
    offline = _parse_server_options([])
    assert offline.profile == "offline"
    assert offline.allow_live_api is False

    with pytest.raises(SystemExit):
        _parse_server_options(["--profile", "live-advisory"])
    with pytest.raises(SystemExit):
        _parse_server_options(["--allow-live-api"])

    live = _parse_server_options(
        ["--profile", "live-advisory", "--allow-live-api", "--live-deadline-ms", "30000"]
    )
    assert live.profile == "live-advisory"
    assert live.allow_live_api is True
    assert live.live_deadline_ms == 30_000


def test_live_jev_rate_limit_uses_one_fake_wire_request(monkeypatch):
    import httpx2
    import jevtest.providers.jev_live as jev_live
    from jevtest.metrics import MeasureRunner
    from jevtest.providers.jev_live import JevLiveProvider

    monkeypatch.setattr(jev_live, "validate_https_url", lambda url: url)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(429, json={"error": {"message": "synthetic rate limit"}})

    provider = JevLiveProvider(
        api_key="offline-fake-key", base_url="https://example.invalid",
        max_retries=0, timeout_s=0.5, transport=httpx2.MockTransport(handler),
    )
    result, sample = MeasureRunner(provider).run_one(_state(), max_attempts=1)

    assert result.category == "UNKNOWN"
    assert sample.parse_failure is False
    assert len(requests) == provider.calls_made == len(provider.last_attempts) == 1
    assert provider.last_attempts[0].status == "failed"
    assert provider.last_attempts[0].error_code == "http_429"
    assert "offline-fake-key" not in str(provider.last_attempts[0].model_dump(mode="json"))
    provider.close()


def test_live_ark_invalid_json_uses_one_fake_wire_request_and_unknown_pricing(monkeypatch):
    import httpx
    import jevtest.providers.llm_judge as llm_judge
    from jevtest.metrics import MeasureRunner
    from jevtest.providers.llm_judge import LLMJudgeProvider

    monkeypatch.setattr(
        llm_judge,
        "load_ark_config",
        lambda _path=None: {
            "openai_compatible_base": "https://ark.invalid/v1",
            "model": "fake-strong-model",
        },
    )
    monkeypatch.setenv("ARK_API_KEY", "offline-fake-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "offline-test",
                "object": "chat.completion",
                "created": 1,
                "model": "fake-strong-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "not json"}}],
                "usage": {"prompt_tokens": 21, "completion_tokens": 4, "total_tokens": 25},
            },
        )

    provider = LLMJudgeProvider(
        transport=httpx.MockTransport(handler), timeout_s=0.5
    )
    result, sample = MeasureRunner(provider).run_one(_state(), max_attempts=1)

    assert result.category == "UNKNOWN"
    assert sample.parse_failure is True
    assert len(requests) == provider.calls_made == len(provider.last_attempts) == 1
    assert provider.last_attempts[0].status == "failed"
    assert provider.last_attempts[0].pricing_status == "unknown"
    assert provider.last_attempts[0].cost is None
    assert provider.last_attempts[0].input_tokens == 21
    assert provider.last_attempts[0].output_tokens == 4
    assert "offline-fake-key" not in str(provider.last_attempts[0].model_dump(mode="json"))
    provider.close()


def test_live_ark_json_mode_rejection_does_not_escape_one_wire_request_budget(monkeypatch):
    import httpx
    import jevtest.providers.llm_judge as llm_judge
    from jevtest.metrics import MeasureRunner
    from jevtest.providers.llm_judge import LLMJudgeProvider

    monkeypatch.setattr(
        llm_judge,
        "load_ark_config",
        lambda _path=None: {
            "openai_compatible_base": "https://ark.invalid/v1",
            "model": "fake-strong-model",
        },
    )
    monkeypatch.setenv("ARK_API_KEY", "offline-fake-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            400,
            json={"error": {"message": "synthetic unsupported response_format"}},
            request=request,
        )

    provider = LLMJudgeProvider(
        transport=httpx.MockTransport(handler), timeout_s=0.5
    )
    result, sample = MeasureRunner(provider).run_one(_state(), max_attempts=1)

    assert result.category == "UNKNOWN"
    assert sample.parse_failure is False
    assert len(requests) == provider.calls_made == len(provider.last_attempts) == 1
    assert provider.last_attempts[0].status == "failed"
    assert provider.last_attempts[0].error_code == "http_400_json_mode_rejected"
    assert "offline-fake-key" not in str(provider.last_attempts[0].model_dump(mode="json"))
    provider.close()
