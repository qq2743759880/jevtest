"""Versioned shared inference facts, separate from dataset/governance metadata."""

from __future__ import annotations

import hashlib
import json

from jevtest.execution_contracts import CommonFactsV1, EvidenceRefV1
from jevtest.models import FailureState

PROJECTION_VERSION = "common-facts-v1"


def project_failure_state(state: FailureState) -> CommonFactsV1:
    """Return only failure facts; omit case_id and provenance to avoid label leakage."""
    return CommonFactsV1(
        schema_version=PROJECTION_VERSION,
        result=state.result,
        previous_runs=list(state.previous_runs),
        device=state.device,
        sw_version=state.sw_version,
        assertion=state.assertion,
        syslog_excerpt=state.syslog_excerpt,
        env={key: state.env[key] for key in sorted(state.env)},
        retry=state.retry,
    )


def projection_json(view: CommonFactsV1) -> str:
    """Canonical JSON bytes used for all adapter payloads and projection hashing."""
    return json.dumps(
        view.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def projection_sha256(view: CommonFactsV1) -> str:
    """Content integrity hash; this is not an anonymization mechanism."""
    return hashlib.sha256(projection_json(view).encode("utf-8")).hexdigest()


def field_content_sha256(value: object) -> str:
    """Hash the canonical JSON value used by the submitted-facts evidence API."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def evidence_refs_for_facts(
    facts: CommonFactsV1, *, request_id: str, result_version: str
) -> list[EvidenceRefV1]:
    """Bind each model-visible submitted field to its exact canonical value hash."""
    payload = facts.model_dump(mode="json")
    refs: list[EvidenceRefV1] = []
    for field_path, value in payload.items():
        if field_path == "schema_version":
            continue
        fields = (
            [(f"env.{key}", item) for key, item in sorted(value.items())]
            if field_path == "env"
            else [(field_path, value)]
        )
        for path, field_value in fields:
            refs.append(
                EvidenceRefV1(
                    evidence_id=f"{request_id}:{result_version}:{path}",
                    field_path=path,
                    content_sha256=field_content_sha256(field_value),
                    capture_scope=(
                        "provided_excerpt" if path == "syslog_excerpt" else "submitted_field"
                    ),
                    evidence_kind="observed",
                )
            )
    return refs
