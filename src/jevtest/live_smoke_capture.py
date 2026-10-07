"""One-request Live Advisory smoke capture with direct HTTP evidence.

This tool is opt-in and is never invoked by the application server or tests.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from jevtest.models import FailureState


_APPROVAL_PHRASE = "LIVE SMOKE APPROVED"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _json_or_error(response: Any) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError:
        return {"error": "non_json_response", "body_excerpt": response.text[:500]}
    return value if isinstance(value, dict) else {"value": value}


def capture_live_advisory_smoke(
    client: Any,
    *,
    origin: str,
    failure_state: FailureState,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Submit once and archive the status/replay/id/ETag observed on the response."""
    health_response = client.get("/v1/health")
    health = _json_or_error(health_response)
    if health_response.status_code != 200 or health != {
        "status": "ok",
        "profile": "live-advisory",
        "execution_mode": "live",
        "run_purpose": "advisory",
        "provider_mode": "live",
    }:
        raise RuntimeError("the target service is not the live-advisory profile; no request submitted")

    session_response = client.get("/v1/session")
    session = _json_or_error(session_response)
    csrf_token = session.get("csrf_token")
    if session_response.status_code != 200 or not isinstance(csrf_token, str) or not csrf_token:
        raise RuntimeError("could not establish the local session; no request submitted")

    idempotency_key = idempotency_key or f"owner-smoke-{uuid4().hex}"
    try:
        submission = client.post(
            "/v1/triage",
            headers={
                "Origin": origin,
                "X-CSRF-Token": csrf_token,
                "Idempotency-Key": idempotency_key,
            },
            json={"failure_state": failure_state.model_dump(mode="json"), "live_confirmed": True},
        )
    except Exception as exc:
        return {
            "schema_version": "live-advisory-smoke-evidence-v1",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "automation_smoke": True,
            "actor_identity_verified": False,
            "counts_toward_human_acceptance": False,
            "formal_annotation": False,
            "health": {"http_status": health_response.status_code, "body": health},
            "submission_http": {
                "http_status": None,
                "replayed": None,
                "request_id": None,
                "etag": None,
                "capture_source": "direct_http_response_unavailable",
                "outcome": "unknown_after_transport_error",
                "error_type": type(exc).__name__,
            },
            "idempotency_key": idempotency_key,
            "readbacks": {},
            "wire_count": "unknown",
            "screenshot_persistence": "unavailable",
            "review_events_created_by_harness": 0,
        }
    payload = _json_or_error(submission)
    envelope = payload.get("envelope") if isinstance(payload.get("envelope"), dict) else {}
    request_id = envelope.get("request_id")
    readbacks: dict[str, Any] = {}
    if isinstance(request_id, str) and request_id:
        for name, path in (
            ("envelope", f"/v1/triage/{request_id}"),
            ("facts", f"/v1/triage/{request_id}/facts?result_version={envelope.get('result_version', '')}"),
            ("reviews", f"/v1/triage/{request_id}/reviews"),
            ("report", f"/v1/triage/{request_id}/report"),
            ("execution_metadata", f"/v1/triage/{request_id}/execution-metadata"),
        ):
            try:
                response = client.get(path)
                readbacks[name] = {
                    "http_status": response.status_code,
                    "etag": response.headers.get("ETag"),
                    "body": _json_or_error(response),
                }
            except Exception as exc:
                readbacks[name] = {
                    "http_status": None,
                    "transport_error": type(exc).__name__,
                }

    return {
        "schema_version": "live-advisory-smoke-evidence-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "automation_smoke": True,
        "actor_identity_verified": False,
        "counts_toward_human_acceptance": False,
        "formal_annotation": False,
        "health": {"http_status": health_response.status_code, "body": health},
        "submission_http": {
            "http_status": submission.status_code,
            "replayed": payload.get("replayed"),
            "request_id": request_id,
            "etag": submission.headers.get("ETag"),
            "capture_source": "direct_http_response",
        },
        "idempotency_key": idempotency_key,
        "submission_body": payload,
        "readbacks": readbacks,
        "application_attempts": [
            {
                "stage": attempt.get("stage"),
                "attempt_index": attempt.get("attempt_index"),
                "provider_id": attempt.get("provider_id"),
                "status": attempt.get("status"),
            }
            for attempt in envelope.get("attempts", [])
            if isinstance(attempt, dict) and attempt.get("stage") in {"jev", "llm"}
        ],
        "wire_count": "unknown",
        "screenshot_persistence": "unavailable",
        "review_events_created_by_harness": 0,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one explicitly approved Live Advisory smoke"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    parser.add_argument(
        "--fixture",
        default="examples/owner-test-pack-v1.1/C-ambiguous-unknown.json",
        help="sanitized owner-experience FailureState JSON",
    )
    parser.add_argument("--output", help="evidence path under runs/; defaults to a timestamped file")
    parser.add_argument(
        "--allow-real-provider-call",
        action="store_true",
        help="required opt-in; the command also asks for the approval phrase",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.allow_real_provider_call:
        raise SystemExit("refusing to call real providers without --allow-real-provider-call")
    if input("This submits one fixture to real Jev and Ark. Type LIVE SMOKE APPROVED: ").strip() != _APPROVAL_PHRASE:
        raise SystemExit("approval phrase did not match; no request submitted")

    parsed = urlsplit(args.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.port is None
        or parsed.path not in {"", "/"}
    ):
        raise SystemExit("base URL must be an HTTP loopback origin with an explicit port")
    origin = f"{parsed.scheme}://{parsed.netloc}"

    fixture_path = Path(args.fixture)
    if not fixture_path.is_absolute():
        fixture_path = _REPOSITORY_ROOT / fixture_path
    try:
        failure_state = FailureState.model_validate_json(
            fixture_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not load a valid FailureState fixture: {exc}") from None

    output_path = Path(args.output) if args.output else Path(
        "runs",
        f"live-advisory-smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
        f"{uuid4().hex[:8]}.json",
    )
    if not output_path.is_absolute():
        output_path = _REPOSITORY_ROOT / output_path
    resolved_output = output_path.resolve()
    if not resolved_output.is_relative_to((_REPOSITORY_ROOT / "runs").resolve()):
        raise SystemExit("output must stay under the repository runs/ directory")
    temp_path = resolved_output.with_suffix(resolved_output.suffix + ".tmp")
    if resolved_output.exists() or temp_path.exists():
        raise SystemExit("output path already exists; choose a new evidence filename")

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("smoke capture requires httpx; install the jevtest[web] extra") from exc
    try:
        with httpx.Client(base_url=origin, timeout=90.0, follow_redirects=False) as client:
            artifact = capture_live_advisory_smoke(
                client, origin=origin, failure_state=failure_state
            )
    except Exception as exc:
        raise SystemExit(f"smoke request failed: {type(exc).__name__}; check local service logs") from None

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(resolved_output)
    print(f"Saved direct HTTP smoke evidence: {resolved_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
