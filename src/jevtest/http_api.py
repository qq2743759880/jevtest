"""Loopback-only HTTP adapter for the offline triage application service.

``create_app`` does not start a server; :func:`main` is an optional loopback-only
module entry point. Browser sessions and operator labels are local,
self-asserted workflow hints; they are not authentication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import ipaddress
import json
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from http.cookies import SimpleCookie
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from jevtest.app_service import (
    ApplicationIdempotencyConflict,
    LiveSubmissionConfirmationRequired,
    SubmittedFactNotFound,
    TriageApplicationService,
)
from jevtest.execution_contracts import ReviewRecordV1
from jevtest.local_store import LocalStoreConflictError
from jevtest.models import FailureState, TriageResult

MAX_REQUEST_BYTES = 64 * 1024
MAX_IDEMPOTENCY_KEY_LENGTH = 200
_SESSION_COOKIE = "jevtest_session"
_CSRF_HEADER = "x-csrf-token"
_IDEMPOTENCY_HEADER = "Idempotency-Key"
_SESSION_TTL_SECONDS = 12 * 60 * 60
_MAX_SESSIONS = 128
_STATIC_DIR = Path(__file__).with_name("static")


class _StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _TriageBody(_StrictBody):
    failure_state: FailureState
    live_confirmed: bool = False


class _ReviewBody(_StrictBody):
    decision: Literal["accept", "correct", "defer"]
    operator_label: str = Field(min_length=1, max_length=120)
    confirmation: str = Field(min_length=1, max_length=120)
    reason: str | None = Field(default=None, max_length=2000)
    after_result: TriageResult | None = None

    @model_validator(mode="after")
    def fields_match_decision(self) -> "_ReviewBody":
        if self.decision == "correct":
            if self.after_result is None:
                raise ValueError("correct requires after_result")
            if not (self.reason and self.reason.strip()):
                raise ValueError("correct requires a reason")
        elif self.after_result is not None:
            raise ValueError("after_result is only valid for correct")
        if self.decision == "defer" and not (self.reason and self.reason.strip()):
            raise ValueError("defer requires a reason")
        if not self.operator_label.strip():
            raise ValueError("operator_label cannot be empty")
        return self


class _ApiFailure(Exception):
    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


def _expected_host_and_origin(value: str) -> tuple[str, str]:
    candidate = value.strip()
    try:
        parsed = urlsplit(f"http://{candidate}")
        if (
            not candidate
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.port is None
        ):
            raise ValueError
        address = ipaddress.ip_address(parsed.hostname or "")
        if not address.is_loopback or not 1 <= parsed.port <= 65535:
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise ValueError("expected_host must be a loopback IP literal with an explicit port") from exc

    host = (
        f"[{address.compressed}]:{parsed.port}"
        if address.version == 6
        else f"{address.compressed}:{parsed.port}"
    )
    return host, f"http://{host}"


class _SessionRegistry:
    """Bounded in-process CSRF session registry; it authenticates nobody."""

    def __init__(self) -> None:
        self._items: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        with self._lock:
            item = self._items.get(session_id)
            now = time.monotonic()
            if item is None:
                return None
            token, expires_at = item
            if expires_at <= now:
                self._items.pop(session_id, None)
                return None
            self._items[session_id] = (token, now + _SESSION_TTL_SECONDS)
            self._items.move_to_end(session_id)
            return token

    def create(self) -> tuple[str, str]:
        with self._lock:
            now = time.monotonic()
            for session_id, (_token, expires_at) in list(self._items.items()):
                if expires_at <= now:
                    self._items.pop(session_id, None)
            while len(self._items) >= _MAX_SESSIONS:
                self._items.popitem(last=False)
            session_id = secrets.token_urlsafe(32)
            token = secrets.token_urlsafe(32)
            self._items[session_id] = (token, now + _SESSION_TTL_SECONDS)
            return session_id, token


def _raw_header(scope: Scope, name: bytes) -> list[str]:
    values: list[str] = []
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            try:
                values.append(value.decode("latin-1"))
            except UnicodeDecodeError:
                return []
    return values


def _is_loopback_peer(scope: Scope) -> bool:
    client = scope.get("client")
    if not client or not client[0]:
        return False
    try:
        return ipaddress.ip_address(str(client[0]).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


async def _send_error(scope: Scope, receive: Receive, send: Send, status: int, code: str) -> None:
    response = JSONResponse(status_code=status, content={"error": {"code": code}})
    await response(scope, receive, send)


class _LocalBoundaryMiddleware:
    """Enforce loopback peer, exact authority/origin, CSRF, and body size."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        expected_host: str,
        expected_origin: str,
        sessions: _SessionRegistry,
        max_body_bytes: int = MAX_REQUEST_BYTES,
    ) -> None:
        self.app = app
        self.expected_host = expected_host.casefold()
        self.expected_origin = expected_origin
        self.sessions = sessions
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not _is_loopback_peer(scope):
            await _send_error(scope, receive, send, 403, "loopback_only")
            return

        hosts = _raw_header(scope, b"host")
        if len(hosts) != 1 or hosts[0].strip().casefold() != self.expected_host:
            await _send_error(scope, receive, send, 403, "host_rejected")
            return

        method = str(scope.get("method", "GET")).upper()
        if method not in {"GET", "HEAD", "OPTIONS"}:
            origins = _raw_header(scope, b"origin")
            if len(origins) != 1 or not hmac.compare_digest(origins[0], self.expected_origin):
                await _send_error(scope, receive, send, 403, "origin_rejected")
                return

            csrf_values = _raw_header(scope, _CSRF_HEADER.encode("ascii"))
            cookies = _raw_header(scope, b"cookie")
            if len(csrf_values) != 1 or len(cookies) > 1:
                await _send_error(scope, receive, send, 403, "csrf_rejected")
                return
            cookie = SimpleCookie()
            try:
                cookie.load(cookies[0] if cookies else "")
            except Exception:
                await _send_error(scope, receive, send, 403, "csrf_rejected")
                return
            morsel = cookie.get(_SESSION_COOKIE)
            expected_token = self.sessions.get(morsel.value if morsel else None)
            if expected_token is None or not hmac.compare_digest(csrf_values[0], expected_token):
                await _send_error(scope, receive, send, 403, "csrf_rejected")
                return

        lengths = _raw_header(scope, b"content-length")
        if len(lengths) > 1:
            await _send_error(scope, receive, send, 422, "invalid_request")
            return
        if lengths:
            try:
                declared_size = int(lengths[0])
            except ValueError:
                await _send_error(scope, receive, send, 422, "invalid_request")
                return
            if declared_size < 0:
                await _send_error(scope, receive, send, 422, "invalid_request")
                return
            if declared_size > self.max_body_bytes:
                await _send_error(scope, receive, send, 413, "request_too_large")
                return

        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            if message.get("type") != "http.request":
                continue
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                await _send_error(scope, receive, send, 413, "request_too_large")
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


def _envelope_response(envelope: Any) -> dict[str, Any]:
    payload = envelope.model_dump(mode="json")
    # Provider exception summaries are internal diagnostics, not client-facing evidence.
    for attempt in payload.get("attempts", []):
        attempt.pop("error_summary", None)
    return payload


def _review_id(scope: str, key: str) -> str:
    digest = hashlib.sha256(f"jevtest-http-review\0{scope}\0{key}".encode("utf-8")).hexdigest()
    return f"http-review-{digest}"


def _same_review_intent(
    existing: ReviewRecordV1,
    *,
    decision: str,
    operator_label: str,
    reason: str | None,
    after_result: TriageResult | None,
) -> bool:
    return (
        existing.decision == decision
        and existing.actor_origin == operator_label
        and existing.actor_type == "local_user"
        and existing.assistance_mode == "model_assisted"
        and existing.reason == reason
        and existing.after_result == after_result
    )


def create_app(
    service: TriageApplicationService,
    *,
    expected_host: str,
) -> FastAPI:
    """Build a host-neutral ASGI app with a loopback-only browser boundary.

    ``expected_host`` must be the exact loopback host and port used by the ASGI
    server, such as ``127.0.0.1:8765``. This function does not launch or bind
    that server.
    """

    if not isinstance(service, TriageApplicationService):
        raise TypeError("service must be a TriageApplicationService")
    host, origin = _expected_host_and_origin(expected_host)
    sessions = _SessionRegistry()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        _LocalBoundaryMiddleware,
        expected_host=host,
        expected_origin=origin,
        sessions=sessions,
    )

    @app.get("/", include_in_schema=False)
    def review_page() -> FileResponse:
        return FileResponse(
            _STATIC_DIR / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    app.mount("/assets", StaticFiles(directory=_STATIC_DIR), name="review-assets")

    @app.exception_handler(_ApiFailure)
    async def api_failure_handler(_request: Request, exc: _ApiFailure) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code}},
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI's default includes rejected values; only expose a stable code.
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request"}})

    @app.exception_handler(ApplicationIdempotencyConflict)
    async def idempotency_conflict_handler(
        _request: Request, _exc: ApplicationIdempotencyConflict
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": {"code": "idempotency_conflict"}})

    @app.exception_handler(LocalStoreConflictError)
    async def store_conflict_handler(
        _request: Request, _exc: LocalStoreConflictError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": {"code": "state_conflict"}})

    @app.exception_handler(SubmittedFactNotFound)
    async def evidence_not_found_handler(
        _request: Request, _exc: SubmittedFactNotFound
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": {"code": "evidence_not_found"}})

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "request_rejected"
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": code}})

    @app.exception_handler(sqlite3.Error)
    async def sqlite_error_handler(_request: Request, _exc: sqlite3.Error) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": {"code": "storage_unavailable"}})

    @app.exception_handler(OSError)
    async def os_error_handler(_request: Request, _exc: OSError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": {"code": "storage_unavailable"}})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error"}})

    @app.get("/v1/session")
    def get_session(request: Request, response: Response) -> dict[str, str]:
        session_id = request.cookies.get(_SESSION_COOKIE)
        token = sessions.get(session_id)
        if session_id is None or token is None:
            session_id, token = sessions.create()
            response.set_cookie(
                _SESSION_COOKIE,
                session_id,
                max_age=_SESSION_TTL_SECONDS,
                httponly=True,
                secure=False,  # loopback HTTP only; never use this adapter remotely
                samesite="strict",
                path="/",
            )
        response.headers["Cache-Control"] = "no-store"
        return {"csrf_token": token}

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "profile": service.execution_profile,
            "execution_mode": service.policy.execution_mode,
            "run_purpose": service.policy.run_purpose,
            "provider_mode": service.provider_mode,
        }

    @app.get("/v1/review-queue")
    def list_review_queue() -> dict[str, Any]:
        items = [
            {
                "envelope": _envelope_response(envelope),
                "handoff": handoff.model_dump(mode="json") if handoff is not None else None,
            }
            for envelope, handoff in service.list_current_review_queue()
        ]
        return {"items": items, "count": len(items)}

    @app.post("/v1/triage")
    def submit_triage(
        body: _TriageBody,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias=_IDEMPOTENCY_HEADER),
    ) -> dict[str, Any]:
        key = _checked_idempotency_key(idempotency_key)
        if service.execution_profile == "live-advisory" and not body.live_confirmed:
            raise _ApiFailure(400, "live_confirmation_required")
        try:
            result = service.submit(
                body.failure_state,
                idempotency_key=key,
                live_confirmed=body.live_confirmed,
            )
        except LiveSubmissionConfirmationRequired as exc:
            raise _ApiFailure(400, "live_confirmation_required") from exc
        response.status_code = 200 if result.replayed else 201
        response.headers["ETag"] = f'"{result.envelope.result_version}"'
        return {
            "envelope": _envelope_response(result.envelope),
            "report": result.report.model_dump(mode="json") if result.report is not None else None,
            "report_result_version": result.report_result_version,
            "replayed": result.replayed,
        }

    @app.get("/v1/triage/{request_id}")
    def get_triage(
        request_id: str,
        response: Response,
        result_version: str | None = Query(default=None, min_length=1),
    ) -> dict[str, Any]:
        envelope = service.get_result(request_id, result_version)
        if envelope is None:
            raise _ApiFailure(404, "result_not_found")
        response.headers["ETag"] = f'"{envelope.result_version}"'
        return {"envelope": _envelope_response(envelope)}

    @app.get("/v1/triage/{request_id}/report")
    def get_run_report(
        request_id: str,
        result_version: str | None = Query(default=None, min_length=1),
    ) -> dict[str, Any]:
        record = service.get_run_report(request_id, result_version)
        if record is None:
            raise _ApiFailure(404, "run_report_not_found")
        report, report_result_version = record
        current = service.get_result(request_id)
        return {
            "request_id": request_id,
            "report_result_version": report_result_version,
            "current_result_version": current.result_version if current is not None else None,
            "report": report.model_dump(mode="json"),
        }

    @app.get("/v1/triage/{request_id}/execution-metadata")
    def get_provider_execution_metadata(
        request_id: str,
        result_version: str | None = Query(default=None, min_length=1),
    ) -> dict[str, Any]:
        record = service.get_provider_execution_metadata(request_id, result_version)
        if record is None:
            raise _ApiFailure(404, "execution_metadata_not_found")
        items, metadata_result_version = record
        current = service.get_result(request_id)
        return {
            "request_id": request_id,
            "metadata_result_version": metadata_result_version,
            "current_result_version": current.result_version if current is not None else None,
            "items": [item.model_dump(mode="json") for item in items],
        }

    @app.get("/v1/triage/{request_id}/facts")
    def get_common_facts(
        request_id: str,
        result_version: str | None = Query(default=None, min_length=1),
    ) -> dict[str, Any]:
        envelope = service.get_result(request_id, result_version)
        if envelope is None:
            raise _ApiFailure(404, "result_not_found")
        facts = service.store.get_common_facts(request_id, envelope.result_version)
        if facts is None:
            raise _ApiFailure(404, "facts_not_found")
        return {
            "request_id": request_id,
            "result_version": envelope.result_version,
            "facts": facts.model_dump(mode="json"),
        }

    @app.get("/v1/triage/{request_id}/evidence")
    def get_evidence(
        request_id: str,
        result_version: str = Query(min_length=1),
        field_path: str = Query(min_length=1, max_length=256),
    ) -> dict[str, Any]:
        try:
            evidence = service.lookup_submitted_facts_evidence(
                request_id, result_version, field_path
            )
        except SubmittedFactNotFound as exc:
            raise _ApiFailure(404, "evidence_not_found") from exc
        return {
            "request_id": evidence.request_id,
            "result_version": evidence.result_version,
            "field_path": evidence.field_path,
            "value": evidence.value,
            "content_sha256": evidence.content_sha256,
            "capture_scope": evidence.capture_scope,
            "kind": "submitted_fact",
            "root_cause_verified": False,
        }

    @app.get("/v1/triage/{request_id}/reviews")
    def get_review_history(
        request_id: str,
        result_version: str | None = Query(default=None, min_length=1),
    ) -> dict[str, Any]:
        envelope = service.get_result(request_id, result_version)
        if envelope is None:
            raise _ApiFailure(404, "result_not_found")
        current = service.get_result(request_id)
        return {
            "request_id": request_id,
            "result_version": result_version or envelope.result_version,
            "current_result_version": current.result_version if current is not None else None,
            "reviews": [
                item.model_dump(mode="json")
                for item in service.store.list_reviews(request_id, result_version)
            ],
            "actor_identity_verified": False,
        }

    @app.post("/v1/triage/{request_id}/reviews")
    def post_review(
        request_id: str,
        body: _ReviewBody,
        response: Response,
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias=_IDEMPOTENCY_HEADER),
    ) -> dict[str, Any]:
        key = _checked_idempotency_key(idempotency_key)
        result_version = _checked_if_match(if_match)
        expected_confirmation = f"{body.decision} {request_id[:8]}"
        if body.confirmation != expected_confirmation:
            raise _ApiFailure(422, "confirmation_mismatch")

        envelope = service.get_result(request_id, result_version)
        if envelope is None or envelope.result is None:
            current = service.get_result(request_id)
            if current is not None:
                raise _ApiFailure(409, "result_version_conflict")
            raise _ApiFailure(404, "result_not_found")

        stable_review_id = _review_id(service.idempotency_scope, key)
        existing = next(
            (
                item
                for item in service.store.list_reviews(request_id, result_version)
                if item.review_id == stable_review_id
            ),
            None,
        )
        if existing is not None:
            if not _same_review_intent(
                existing,
                decision=body.decision,
                operator_label=body.operator_label.strip(),
                reason=body.reason,
                after_result=body.after_result,
            ):
                raise _ApiFailure(409, "idempotency_conflict")
            current = service.get_result(request_id)
            response.headers["ETag"] = (
                f'"{current.result_version}"' if current is not None else f'"{result_version}"'
            )
            return {
                "review": existing.model_dump(mode="json"),
                "envelope": _envelope_response(current or envelope),
                "replayed": True,
                "actor_identity_verified": False,
            }

        current = service.get_result(request_id)
        if current is None or current.result_version != result_version:
            raise _ApiFailure(409, "result_version_conflict")
        if envelope.review_status not in {"pending", "deferred"}:
            raise _ApiFailure(409, "review_not_pending")
        if (
            body.decision == "correct"
            and body.after_result is not None
            and body.after_result.confidence != envelope.result.confidence
        ):
            raise _ApiFailure(422, "confidence_reassessment_not_supported")

        event = ReviewRecordV1(
            schema_version="review-record-v1",
            review_id=stable_review_id,
            request_id=request_id,
            result_version=result_version,
            actor_type="local_user",
            actor_origin=body.operator_label.strip(),
            assistance_mode="model_assisted",
            decision=body.decision,
            before_result=envelope.result,
            after_result=body.after_result,
            confidence_origin=(
                "inherited_unreassessed" if body.decision == "correct" else None
            ),
            reason=body.reason,
            reviewed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        replayed = False
        try:
            saved = service.store.append_review(event)
        except LocalStoreConflictError as exc:
            # A racing retry with the same stable review id may already have committed.
            replay = next(
                (
                    item
                    for item in service.store.list_reviews(request_id, result_version)
                    if item.review_id == stable_review_id
                ),
                None,
            )
            if replay is None or not _same_review_intent(
                replay,
                decision=body.decision,
                operator_label=body.operator_label.strip(),
                reason=body.reason,
                after_result=body.after_result,
            ):
                raise _ApiFailure(409, "review_conflict") from exc
            saved = replay
            replayed = True

        updated = service.get_result(request_id)
        response.headers["ETag"] = (
            f'"{updated.result_version}"' if updated is not None else f'"{result_version}"'
        )
        return {
            "review": saved.model_dump(mode="json"),
            "envelope": _envelope_response(updated or envelope),
            "replayed": replayed,
            "actor_identity_verified": False,
        }

    return app


def _checked_idempotency_key(value: str | None) -> str:
    if value is None or not value.strip() or len(value) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise _ApiFailure(422, "idempotency_key_required")
    if any(ord(char) < 0x21 or ord(char) > 0x7E for char in value):
        raise _ApiFailure(422, "idempotency_key_invalid")
    return value


def _checked_if_match(value: str | None) -> str:
    if value is None or len(value) < 3 or value[0] != '"' or value[-1] != '"':
        raise _ApiFailure(409, "result_version_required")
    version = value[1:-1]
    if not version or '"' in version or version.startswith("W/"):
        raise _ApiFailure(409, "result_version_required")
    return version


@dataclass(frozen=True, slots=True)
class _ServerOptions:
    host: str
    port: int
    expected_host: str
    store_path: Path
    profile: str
    allow_live_api: bool
    live_deadline_ms: int


def _parse_server_options(argv: list[str] | None = None) -> _ServerOptions:
    """Parse loopback binding and the explicit live-advisory startup authorization."""

    parser = argparse.ArgumentParser(description="Run JevTest's local HTTP API")
    parser.add_argument("--host", default="127.0.0.1", help="loopback IP literal only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--store", default=None, metavar="PATH", help="SQLite store path")
    parser.add_argument("--profile", choices=["offline", "live-advisory"], default="offline")
    parser.add_argument("--allow-live-api", action="store_true", help="explicitly authorize real Jev/Ark API calls for this process")
    parser.add_argument("--live-deadline-ms", type=int, default=30_000, help="live request deadline (1..60000 ms)")
    args = parser.parse_args(argv)

    if args.profile == "live-advisory" and not args.allow_live_api:
        parser.error("--profile live-advisory requires the explicit --allow-live-api flag")
    if args.profile == "offline" and args.allow_live_api:
        parser.error("--allow-live-api requires --profile live-advisory")
    if not 1 <= args.live_deadline_ms <= 60_000:
        parser.error("--live-deadline-ms must be between 1 and 60000")

    try:
        address = ipaddress.ip_address(args.host)
    except ValueError:
        parser.error("--host must be a loopback IP literal (127.0.0.1 or ::1)")
    if not address.is_loopback:
        parser.error("--host must be loopback; remote binding is not supported")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    host = address.compressed
    expected_host = f"[{host}]:{args.port}" if address.version == 6 else f"{host}:{args.port}"
    store_path = (
        Path(args.store).expanduser()
        if args.store
        else Path.home() / ".jevtest" / "triage.sqlite3"
    )
    return _ServerOptions(
        host=host,
        port=args.port,
        expected_host=expected_host,
        store_path=store_path,
        profile=args.profile,
        allow_live_api=args.allow_live_api,
        live_deadline_ms=args.live_deadline_ms,
    )


def _parse_server_args(argv: list[str] | None = None) -> tuple[str, int, str, Path]:
    """Compatibility helper for callers that only need bind/store arguments."""

    options = _parse_server_options(argv)
    return options.host, options.port, options.expected_host, options.store_path


def main(argv: list[str] | None = None) -> int:
    """Run the optional single-user loopback server (`python -m jevtest.http_api`)."""

    options = _parse_server_options(argv)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise SystemExit("HTTP server requires the optional extra: pip install jevtest[web]") from exc

    from jevtest.local_store import LocalStore

    from jevtest.runtime_authorization import authorize_live_advisory

    live_authorization = (
        authorize_live_advisory(
            explicit_opt_in=options.allow_live_api,
            deadline_ms=options.live_deadline_ms,
        )
        if options.profile == "live-advisory"
        else None
    )
    with LocalStore(options.store_path) as store:
        try:
            service = TriageApplicationService(
                store,
                idempotency_scope="local-http-v1",
                execution_profile=options.profile,  # type: ignore[arg-type]
                live_authorization=live_authorization,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"JevTest startup rejected: {exc}") from None
        try:
            uvicorn.run(
                create_app(service, expected_host=options.expected_host),
                host=options.host,
                port=options.port,
                log_level="warning",
            )
        finally:
            service.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - no tests start a listener
    raise SystemExit(main())
