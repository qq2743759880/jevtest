"""Host-independent, offline-first application service for triage requests.

This layer composes the existing cascade, local store, and report builder.
Offline is the default. Its opt-in live-advisory profile adds only the existing
real Jev and Ark providers under a bounded runtime authorization. Submitted
fact lookups return field-scoped hashes and capture scope; they do not establish
root cause or independently verify the submitted fact.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel

from jevtest.cascade import run_cascade
from jevtest.execution_contracts import (
    CommonFactsV1,
    HandoffRecordV1,
    ProviderExecutionMetadataV1,
    RunReportV2,
    RoutingPolicyV1,
    TriageEnvelopeV1,
)
from jevtest.input_projection import field_content_sha256, project_failure_state, projection_sha256
from jevtest.local_store import LocalStore, LocalStoreConflictError
from jevtest.models import FailureState
from jevtest.run_reporting import build_run_report
from jevtest.runtime_authorization import (
    LiveRuntimeAuthorization,
    LiveAuthorizationError,
)


class ApplicationIdempotencyConflict(ValueError):
    """A key was reused with different request input or policy identity."""


class UnsupportedApplicationPolicy(ValueError):
    """The selected application profile does not authorize this policy."""


class LiveSubmissionConfirmationRequired(ValueError):
    """A live request was submitted without the per-request operator confirmation."""


class LiveProviderInitializationError(RuntimeError):
    """A safe, actionable live-provider startup error with no provider details."""


class SubmittedFactNotFound(KeyError):
    """The requested projection field is absent from the stored facts view."""


@dataclass(frozen=True, slots=True)
class ApplicationRunResult:
    envelope: TriageEnvelopeV1
    report: RunReportV2 | None
    report_result_version: str | None
    replayed: bool


@dataclass(frozen=True, slots=True)
class SubmittedFactEvidence:
    request_id: str
    result_version: str
    field_path: str
    value: Any
    content_sha256: str
    capture_scope: str
    root_cause_verified: bool = False


def default_offline_policy() -> RoutingPolicyV1:
    """Return the repository's conservative offline demo policy."""

    return RoutingPolicyV1(
        schema_version="routing-policy-v1",
        policy_id="offline-demo",
        version="1",
        execution_mode="offline",
        run_purpose="demo",
        basis="demo",
        allowed_stages=["rule", "jev", "llm", "human"],
        direct_output_rule_ids=["R1_FLAKY", "R2_ENVIRONMENT", "R3_ERROR_CODE"],
        jev_high_threshold=0.9,
        jev_escalate_threshold=0.7,
        max_total_attempts=6,
        deadline_ms=120_000,
    )


def default_live_advisory_policy(*, deadline_ms: int = 30_000) -> RoutingPolicyV1:
    """A live policy that always ends in human review and never auto-approves."""

    return RoutingPolicyV1(
        schema_version="routing-policy-v1",
        policy_id="live-advisory",
        version="1",
        execution_mode="live",
        run_purpose="advisory",
        basis="demo",
        allowed_stages=["rule", "jev", "llm", "human"],
        direct_output_rule_ids=[],
        jev_high_threshold=None,
        jev_escalate_threshold=0.7,
        max_total_attempts=3,
        deadline_ms=deadline_ms,
    )


def _canonical_hash(value: BaseModel | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _full_input_hash(state: FailureState) -> str:
    # Keep this encoding aligned with cascade._input_hash: full input identity
    # includes request metadata; the separate CommonFacts projection excludes it.
    return _canonical_hash(state)


def _field_content_hash(value: Any) -> str:
    return field_content_sha256(value)


def _request_id(scope: str, key: str) -> str:
    stable_id = uuid5(NAMESPACE_URL, f"jevtest-app:{scope}\0{key}").hex
    return f"app-{stable_id}"


def _result_version(policy_hash: str) -> str:
    return f"policy-sha256-{policy_hash}-{uuid4().hex}"


def _result_binds_policy(result_version: str, policy_hash: str) -> bool:
    return result_version.startswith(f"policy-sha256-{policy_hash}-")


class TriageApplicationService:
    """Run and retrieve offline triage results through one host-neutral API.

    Idempotency identity is ``(idempotency_scope, idempotency_key)``. The
    derived request id is stable for that pair; the stored input hash and
    content-bound result version detect changed input or policy. Reusing the
    identity with the same request returns the stored envelope without
    constructing providers or running Cascade.
    The default policy and all supported execution paths are offline.
    """

    def __init__(
        self,
        store: LocalStore,
        *,
        policy: RoutingPolicyV1 | None = None,
        idempotency_scope: str = "local-app",
        target_role: str = "test_owner",
        execution_profile: Literal["offline", "live-advisory"] = "offline",
        live_authorization: LiveRuntimeAuthorization | None = None,
        live_provider_factory: Callable[[LiveRuntimeAuthorization], Mapping[str, object]] | None = None,
    ) -> None:
        if not idempotency_scope.strip():
            raise ValueError("idempotency_scope must be non-empty")
        if not target_role.strip():
            raise ValueError("target_role must be non-empty")
        if execution_profile not in {"offline", "live-advisory"}:
            raise ValueError("unsupported application execution profile")
        if execution_profile == "offline" and (
            live_authorization is not None or live_provider_factory is not None
        ):
            raise UnsupportedApplicationPolicy("offline profile cannot accept live authorization or providers")
        self.store = store
        self.execution_profile = execution_profile
        self.live_authorization = live_authorization
        self.policy = policy or (
            default_live_advisory_policy(deadline_ms=live_authorization.deadline_ms)
            if execution_profile == "live-advisory" and live_authorization is not None
            else default_offline_policy()
        )
        self.idempotency_scope = idempotency_scope
        self.target_role = target_role
        self._submit_lock = Lock()
        self._validate_policy(self.policy, execution_profile, live_authorization)
        self._live_providers: dict[str, object] | None = None
        if execution_profile == "live-advisory":
            try:
                providers = (
                    dict(live_provider_factory(live_authorization))
                    if live_provider_factory is not None and live_authorization is not None
                    else self._live_advisory_providers(live_authorization)
                )
                self._validate_live_providers(providers, live_authorization)
            except LiveProviderInitializationError:
                raise
            except Exception as exc:
                for provider in locals().get("providers", {}).values():
                    close = getattr(provider, "close", None)
                    if callable(close):
                        close()
                raise RuntimeError(
                    "live-advisory provider initialization failed (" + type(exc).__name__ + ")"
                ) from None
            self._live_providers = providers

    @staticmethod
    def _validate_policy(
        policy: RoutingPolicyV1,
        execution_profile: str = "offline",
        authorization: LiveRuntimeAuthorization | None = None,
    ) -> None:
        if execution_profile == "offline" and policy.execution_mode != "offline":
            raise UnsupportedApplicationPolicy(
                "TriageApplicationService only runs execution_mode='offline'"
            )
        if execution_profile == "live-advisory":
            if authorization is None:
                raise UnsupportedApplicationPolicy("live-advisory requires explicit runtime authorization")
            if policy.execution_mode != "live" or policy.run_purpose != "advisory":
                raise UnsupportedApplicationPolicy("live-advisory requires execution_mode='live' and run_purpose='advisory'")
            if (
                policy.jev_high_threshold is not None
                or policy.direct_output_rule_ids
                or policy.allowed_stages != ["rule", "jev", "llm", "human"]
                or policy.max_total_attempts != 1 + authorization.request_provider_attempt_budget
                or policy.deadline_ms > authorization.deadline_ms
            ):
                raise UnsupportedApplicationPolicy(
                    "live-advisory policy must keep human review and stay within the request budget"
                )
        if execution_profile not in {"offline", "live-advisory"}:
            raise UnsupportedApplicationPolicy("unsupported execution profile")
        if policy.run_purpose in {"synthetic_eval", "formal_eval"}:
            raise UnsupportedApplicationPolicy(
                "evaluation policies need labels/evidence that this inference service does not accept"
            )

    @staticmethod
    def _offline_providers() -> dict[str, object]:
        # Credentials never select this profile; offline providers stay the default.
        from jevtest.providers.jev_mock import JevMockProvider
        from jevtest.providers.llm_stub import OfflineLLMStubProvider
        from jevtest.providers.rule import RuleProvider

        return {
            "rule": RuleProvider(),
            "jev": JevMockProvider(),
            "llm": OfflineLLMStubProvider(),
        }

    @staticmethod
    def _live_advisory_providers(
        authorization: LiveRuntimeAuthorization | None,
    ) -> dict[str, object]:
        if authorization is None:
            raise LiveAuthorizationError("live-advisory requires runtime authorization")
        from jevtest.providers import jev_live

        jev_live._load_env_file()
        missing = [
            name
            for name in ("TYPESAFE_API_KEY", "ARK_API_KEY")
            if not os.environ.get(name)
        ]
        if missing:
            raise LiveProviderInitializationError(
                "live-advisory missing required credentials: " + ", ".join(missing)
            )
        jev_provider = None
        try:
            from jevtest.providers.jev_live import JevLiveProvider
            from jevtest.providers.llm_judge import LLMJudgeProvider

            # Each of two sequential provider calls may use at most half the
            # request deadline; SDK timeouts and retry budgets are both bounded.
            timeout_s = authorization.deadline_ms / 2000.0
            jev_provider = JevLiveProvider(max_retries=0, timeout_s=timeout_s)
            return {
                "jev": jev_provider,
                "llm": LLMJudgeProvider(timeout_s=timeout_s),
            }
        except Exception as exc:
            # Provider exception strings may include configuration or SDK request data.
            if jev_provider is not None:
                jev_provider.close()
            raise LiveProviderInitializationError(
                "live-advisory provider initialization failed (" + type(exc).__name__ + ")"
            ) from None

    @staticmethod
    def _validate_live_providers(
        providers: Mapping[str, object], authorization: LiveRuntimeAuthorization | None
    ) -> None:
        if authorization is None or set(providers) != authorization.provider_allowlist:
            raise UnsupportedApplicationPolicy("live provider factory must return only jev and llm")
        for stage in ("jev", "llm"):
            provider = providers[stage]
            if getattr(provider, "arm_name", None) != stage or getattr(provider, "is_mock", True):
                raise UnsupportedApplicationPolicy(f"live-advisory requires a real {stage} provider")

    @property
    def provider_mode(self) -> Literal["mock", "live"]:
        return "live" if self.execution_profile == "live-advisory" else "mock"

    def close(self) -> None:
        """Close the live provider clients owned by this application instance."""

        for provider in (self._live_providers or {}).values():
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def submit(
        self,
        state: FailureState | dict[str, Any],
        *,
        idempotency_key: str,
        policy: RoutingPolicyV1 | None = None,
        live_confirmed: bool = False,
    ) -> ApplicationRunResult:
        """Execute one advisory request, serializing submissions in this process.

        HTTP and MCP adapters share this service instance across worker threads.
        Holding the lock through persistence makes same-key concurrent requests
        replay the first result instead of running Cascade twice.
        """

        with self._submit_lock:
            return self._submit_unlocked(
                state,
                idempotency_key=idempotency_key,
                policy=policy,
                live_confirmed=live_confirmed,
            )

    def _submit_unlocked(
        self,
        state: FailureState | dict[str, Any],
        *,
        idempotency_key: str,
        policy: RoutingPolicyV1 | None = None,
        live_confirmed: bool = False,
    ) -> ApplicationRunResult:
        """Submit while holding ``_submit_lock``."""

        failure_state = (
            state if isinstance(state, FailureState) else FailureState.model_validate(state)
        )
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must be non-empty")
        selected_policy = policy or self.policy
        self._validate_policy(selected_policy, self.execution_profile, self.live_authorization)
        if self.execution_profile == "live-advisory" and not live_confirmed:
            raise LiveSubmissionConfirmationRequired("live provider confirmation is required")

        request_id = _request_id(self.idempotency_scope, idempotency_key)
        input_hash = _full_input_hash(failure_state)
        facts = project_failure_state(failure_state)
        facts_hash = projection_sha256(facts)
        policy_hash = _canonical_hash(selected_policy)

        stored = self.store.get_envelope(request_id)
        if stored is not None:
            if (
                stored.input_hash != input_hash
                or stored.projection_hash != facts_hash
                or stored.policy_id != selected_policy.policy_id
                or stored.policy_version != selected_policy.version
                or not _result_binds_policy(stored.result_version, policy_hash)
                or stored.execution_mode != selected_policy.execution_mode
                or stored.run_purpose != selected_policy.run_purpose
            ):
                raise ApplicationIdempotencyConflict(
                    "idempotency scope/key already belongs to different input or policy"
                )
            saved_facts = self.store.get_common_facts(request_id, stored.result_version)
            if saved_facts is None:
                # Repair a process interruption after result persistence but before projection save.
                self.store.put_common_facts(request_id, stored.result_version, facts)
            elif saved_facts != facts:
                raise ApplicationIdempotencyConflict(
                    "stored CommonFacts projection differs for this idempotency key"
                )
            report_record = self.store.get_run_report(request_id)
            return ApplicationRunResult(
                envelope=stored,
                report=report_record[0] if report_record else None,
                report_result_version=report_record[1] if report_record else None,
                replayed=True,
            )

        result_version = _result_version(policy_hash)
        providers = (
            {**(self._live_providers or {}), "rule": self._offline_providers()["rule"]}
            if self.execution_profile == "live-advisory"
            else self._offline_providers()
        )
        envelope = run_cascade(
            failure_state,
            selected_policy,
            providers,
            request_id=request_id,
            result_version=result_version,
            runtime_authorization=self.live_authorization,
        )
        report = self._build_report(envelope, failure_state)
        execution_metadata = self._provider_execution_metadata(envelope, providers)
        if envelope.requires_human_review:
            envelope, _handoff = self.store.queue_handoff(
                envelope,
                idempotency_scope=self.idempotency_scope,
                idempotency_key=idempotency_key,
                target_role=self.target_role,
                common_facts=facts,
                run_report=report,
                provider_execution_metadata=execution_metadata,
            )
        else:
            envelope = self.store.put_envelope(
                envelope,
                common_facts=facts,
                run_report=report,
                provider_execution_metadata=execution_metadata,
            )
        return ApplicationRunResult(
            envelope=envelope,
            report=report,
            report_result_version=envelope.result_version,
            replayed=False,
        )

    @staticmethod
    def _provider_execution_metadata(
        envelope: TriageEnvelopeV1, providers: Mapping[str, Any]
    ) -> list[ProviderExecutionMetadataV1]:
        rows: list[ProviderExecutionMetadataV1] = []
        for attempt in envelope.attempts:
            if attempt.stage not in {"jev", "llm"} or not attempt.invoked:
                continue
            provider = providers.get(attempt.stage)
            resolved = getattr(provider, "last_resolved_models", {}).get(attempt.attempt_index)
            rows.append(
                ProviderExecutionMetadataV1(
                    schema_version="provider-execution-metadata-v1",
                    request_id=envelope.request_id,
                    result_version=envelope.result_version,
                    stage=attempt.stage,
                    attempt_index=attempt.attempt_index,
                    provider_id=attempt.provider_id,
                    requested_model=attempt.model_id,
                    resolved_model=resolved,
                    resolution_source=(
                        "provider_response" if resolved is not None else "not_returned"
                    ),
                )
            )
        return rows

    @staticmethod
    def _build_report(envelope: TriageEnvelopeV1, state: FailureState) -> RunReportV2:
        return build_run_report(
            [envelope],
            report_id=f"{envelope.request_id}-report",
            run_purpose=envelope.run_purpose,
            dataset_kind="synthetic" if state.provenance.synthetic else "real_unlabeled",
            dataset_version=state.provenance.dataset_version,
        )

    def get_result(
        self, request_id: str, result_version: str | None = None
    ) -> TriageEnvelopeV1 | None:
        """Look up a stored result version; omit version to retrieve the current view."""

        return self.store.get_envelope(request_id, result_version)

    def get_run_report(
        self, request_id: str, result_version: str | None = None
    ) -> tuple[RunReportV2, str] | None:
        return self.store.get_run_report(request_id, result_version)

    def get_provider_execution_metadata(
        self, request_id: str, result_version: str | None = None
    ) -> tuple[list[ProviderExecutionMetadataV1], str] | None:
        return self.store.get_provider_execution_metadata(request_id, result_version)

    def list_current_review_queue(
        self,
    ) -> list[tuple[TriageEnvelopeV1, HandoffRecordV1 | None]]:
        """Return current pending/deferred human-review heads from local storage."""

        return self.store.list_current_review_queue()

    def get_result_by_key(
        self,
        idempotency_key: str,
        *,
        idempotency_scope: str | None = None,
        policy: RoutingPolicyV1 | None = None,
    ) -> TriageEnvelopeV1 | None:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must be non-empty")
        selected_policy = policy or self.policy
        self._validate_policy(selected_policy, self.execution_profile, self.live_authorization)
        scope = idempotency_scope or self.idempotency_scope
        return self.store.get_envelope(_request_id(scope, idempotency_key))

    def lookup_submitted_facts_evidence(
        self,
        request_id: str,
        result_version: str,
        field_path: str,
    ) -> SubmittedFactEvidence:
        """Read one stored projected field with a content hash and capture scope.

        Supported paths are CommonFactsV1 top-level fields and ``env.<key>``.
        These are submitted values, not verified causes.
        """

        facts = self.store.get_common_facts(request_id, result_version)
        if facts is None:
            raise SubmittedFactNotFound(
                f"no CommonFactsV1 projection for request/result {request_id}/{result_version}"
            )
        if field_path.startswith("env."):
            key = field_path.removeprefix("env.")
            if not key or key not in facts.env:
                raise SubmittedFactNotFound(field_path)
            value: Any = facts.env[key]
        elif field_path in CommonFactsV1.model_fields and field_path != "schema_version":
            value = getattr(facts, field_path)
        else:
            raise SubmittedFactNotFound(field_path)

        capture_scope = "provided_excerpt" if field_path == "syslog_excerpt" else "submitted_field"
        return SubmittedFactEvidence(
            request_id=request_id,
            result_version=result_version,
            field_path=field_path,
            value=value,
            content_sha256=_field_content_hash(value),
            capture_scope=capture_scope,
            root_cause_verified=False,
        )
