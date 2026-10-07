"""Single-user SQLite persistence for envelopes, local handoffs, and review events.

This module persists local state only. A ``local_queued`` receipt proves that the
handoff card was written to this database; it does not prove that a person or
external system received it. ``actor_origin`` is stored as a caller-provided
label and is not authenticated. In particular, this local store cannot
distinguish a real person from another process running as the same OS user.

The stored envelope is the only source for handoff-card rendering. The v1
envelope has evidence references and summaries, not the original failure log,
so rendered cards do not invent missing source facts.
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from jevtest.execution_contracts import (
    CommonFactsV1,
    HandoffRecordV1,
    ProviderExecutionMetadataV1,
    ReviewRecordV1,
    RunReportV2,
    TriageEnvelopeV1,
)


class LocalStoreConflictError(ValueError):
    """A persisted identity or state conflicts with the requested operation."""


class IdempotencyConflictError(LocalStoreConflictError):
    """A scoped idempotency key was reused for a different input hash."""


class EnvelopeConflictError(LocalStoreConflictError):
    """An existing request/result version was submitted with different content."""


class HandoffConflictError(LocalStoreConflictError):
    """The result already has a handoff or cannot be queued in its current state."""


class StaleResultError(LocalStoreConflictError):
    """An operation targets a result version older than the stored current version."""


class ReviewConflictError(LocalStoreConflictError):
    """A review identity or decision conflicts with the append-only review history."""


_ENVELOPE_MUTABLE_FIELDS = {
    "review_status",
    "delivery_status",
    "resolution_status",
    "local_handoff_id",
}


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _as_envelope(value: TriageEnvelopeV1 | dict[str, object]) -> TriageEnvelopeV1:
    if isinstance(value, TriageEnvelopeV1):
        return value
    return TriageEnvelopeV1.model_validate(value)


def _as_review(value: ReviewRecordV1 | dict[str, object]) -> ReviewRecordV1:
    if isinstance(value, ReviewRecordV1):
        return value
    return ReviewRecordV1.model_validate(value)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _md_inline(value: str) -> str:
    return html.escape(value, quote=False).replace("`", "&#96;").replace("\n", " ")


def _code_fence(value: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    return "`" * max(3, longest + 1)


class LocalStore:
    """Transactional, single-user persistence backed only by stdlib SQLite.

    ``queue_handoff`` returns ``(envelope, handoff)``. Its receipt is always
    ``local_queued``; this API intentionally has no external delivery operation.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 10000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._create_schema()

    def __enter__(self) -> LocalStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS envelopes (
                    request_id TEXT NOT NULL,
                    result_version TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    envelope_json TEXT NOT NULL,
                    PRIMARY KEY (request_id, result_version)
                );

                CREATE TABLE IF NOT EXISTS request_heads (
                    request_id TEXT PRIMARY KEY,
                    result_version TEXT NOT NULL,
                    FOREIGN KEY (request_id, result_version)
                        REFERENCES envelopes(request_id, result_version)
                );

                CREATE TABLE IF NOT EXISTS handoffs (
                    handoff_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    result_version TEXT NOT NULL,
                    idempotency_scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    handoff_json TEXT NOT NULL,
                    UNIQUE (idempotency_scope, idempotency_key),
                    UNIQUE (request_id, result_version),
                    FOREIGN KEY (request_id, result_version)
                        REFERENCES envelopes(request_id, result_version)
                );

                CREATE TABLE IF NOT EXISTS common_facts (
                    request_id TEXT NOT NULL,
                    result_version TEXT NOT NULL,
                    facts_json TEXT NOT NULL,
                    PRIMARY KEY (request_id, result_version),
                    FOREIGN KEY (request_id, result_version)
                        REFERENCES envelopes(request_id, result_version)
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    result_version TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    FOREIGN KEY (request_id, result_version)
                        REFERENCES envelopes(request_id, result_version)
                );

                CREATE INDEX IF NOT EXISTS reviews_by_request
                    ON reviews(request_id, result_version);

                CREATE TABLE IF NOT EXISTS run_reports (
                    request_id TEXT NOT NULL,
                    result_version TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    PRIMARY KEY (request_id, result_version),
                    FOREIGN KEY (request_id, result_version)
                        REFERENCES envelopes(request_id, result_version)
                );

                CREATE TABLE IF NOT EXISTS provider_execution_metadata (
                    request_id TEXT NOT NULL,
                    result_version TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (request_id, result_version),
                    FOREIGN KEY (request_id, result_version)
                        REFERENCES envelopes(request_id, result_version)
                );

                CREATE TRIGGER IF NOT EXISTS reviews_no_update
                BEFORE UPDATE ON reviews
                BEGIN
                    SELECT RAISE(ABORT, 'review events are append-only');
                END;

                CREATE TRIGGER IF NOT EXISTS reviews_no_delete
                BEFORE DELETE ON reviews
                BEGIN
                    SELECT RAISE(ABORT, 'review events are append-only');
                END;

                CREATE TRIGGER IF NOT EXISTS run_reports_no_update
                BEFORE UPDATE ON run_reports
                BEGIN
                    SELECT RAISE(ABORT, 'run reports are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS run_reports_no_delete
                BEFORE DELETE ON run_reports
                BEGIN
                    SELECT RAISE(ABORT, 'run reports are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS provider_execution_no_update
                BEFORE UPDATE ON provider_execution_metadata
                BEGIN
                    SELECT RAISE(ABORT, 'provider execution metadata is immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS provider_execution_no_delete
                BEFORE DELETE ON provider_execution_metadata
                BEGIN
                    SELECT RAISE(ABORT, 'provider execution metadata is immutable');
                END;
                """
            )

    @staticmethod
    def _as_run_report(value: RunReportV2 | dict[str, object]) -> RunReportV2:
        if isinstance(value, RunReportV2):
            return value
        return RunReportV2.model_validate(value)

    @staticmethod
    def _as_provider_metadata(
        values: list[ProviderExecutionMetadataV1 | dict[str, object]],
    ) -> list[ProviderExecutionMetadataV1]:
        return [
            item
            if isinstance(item, ProviderExecutionMetadataV1)
            else ProviderExecutionMetadataV1.model_validate(item)
            for item in values
        ]

    def _persist_run_report_unlocked(
        self, request_id: str, result_version: str, report: RunReportV2
    ) -> RunReportV2:
        if not self._conn.execute(
            "SELECT 1 FROM envelopes WHERE request_id = ? AND result_version = ?",
            (request_id, result_version),
        ).fetchone():
            raise LocalStoreConflictError("run report requires a stored request/result version")
        row = self._conn.execute(
            "SELECT report_json FROM run_reports WHERE request_id = ? AND result_version = ?",
            (request_id, result_version),
        ).fetchone()
        encoded = _canonical_json(report)
        if row is not None:
            if row["report_json"] != encoded:
                raise EnvelopeConflictError("run report is immutable for a request/result version")
            return RunReportV2.model_validate_json(row["report_json"])
        self._conn.execute(
            "INSERT INTO run_reports(request_id, result_version, report_json) VALUES (?, ?, ?)",
            (request_id, result_version, encoded),
        )
        return report

    def _persist_provider_metadata_unlocked(
        self,
        request_id: str,
        result_version: str,
        values: list[ProviderExecutionMetadataV1],
    ) -> list[ProviderExecutionMetadataV1]:
        if any(
            item.request_id != request_id or item.result_version != result_version
            for item in values
        ):
            raise EnvelopeConflictError(
                "provider execution metadata must match its stored request/result version"
            )
        encoded = json.dumps(
            [item.model_dump(mode="json") for item in values],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        row = self._conn.execute(
            "SELECT metadata_json FROM provider_execution_metadata "
            "WHERE request_id = ? AND result_version = ?",
            (request_id, result_version),
        ).fetchone()
        if row is not None:
            if row["metadata_json"] != encoded:
                raise EnvelopeConflictError(
                    "provider execution metadata is immutable for a request/result version"
                )
            return [
                ProviderExecutionMetadataV1.model_validate(item)
                for item in json.loads(row["metadata_json"])
            ]
        self._conn.execute(
            "INSERT INTO provider_execution_metadata(request_id, result_version, metadata_json) "
            "VALUES (?, ?, ?)",
            (request_id, result_version, encoded),
        )
        return values

    def get_run_report(
        self, request_id: str, result_version: str | None = None
    ) -> tuple[RunReportV2, str] | None:
        """Read the immutable report and its source result version."""
        with self._lock:
            if result_version is None:
                row = self._conn.execute(
                    "SELECT result_version, report_json FROM run_reports "
                    "WHERE request_id = ? ORDER BY rowid DESC LIMIT 1",
                    (request_id,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT result_version, report_json FROM run_reports "
                    "WHERE request_id = ? AND result_version = ?",
                    (request_id, result_version),
                ).fetchone()
            if row is None:
                return None
            return (
                RunReportV2.model_validate_json(row["report_json"]),
                str(row["result_version"]),
            )

    def get_provider_execution_metadata(
        self, request_id: str, result_version: str | None = None
    ) -> tuple[list[ProviderExecutionMetadataV1], str] | None:
        """Read immutable provider metadata and the result version it describes."""
        with self._lock:
            if result_version is None:
                row = self._conn.execute(
                    "SELECT result_version, metadata_json FROM provider_execution_metadata "
                    "WHERE request_id = ? ORDER BY rowid DESC LIMIT 1",
                    (request_id,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT result_version, metadata_json FROM provider_execution_metadata "
                    "WHERE request_id = ? AND result_version = ?",
                    (request_id, result_version),
                ).fetchone()
            if row is None:
                return None
            values = [
                ProviderExecutionMetadataV1.model_validate(item)
                for item in json.loads(row["metadata_json"])
            ]
            return values, str(row["result_version"])

    @staticmethod
    def _immutable_envelope_data(envelope: TriageEnvelopeV1) -> dict[str, object]:
        data = envelope.model_dump(mode="json")
        for field in _ENVELOPE_MUTABLE_FIELDS:
            data.pop(field, None)
        return data

    def _load_envelope_unlocked(
        self, request_id: str, result_version: str | None = None
    ) -> TriageEnvelopeV1 | None:
        if result_version is None:
            head = self._conn.execute(
                "SELECT result_version FROM request_heads WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if head is None:
                return None
            result_version = str(head["result_version"])
        row = self._conn.execute(
            "SELECT envelope_json FROM envelopes WHERE request_id = ? AND result_version = ?",
            (request_id, result_version),
        ).fetchone()
        if row is None:
            return None
        return TriageEnvelopeV1.model_validate_json(row["envelope_json"])

    def _persist_envelope_unlocked(self, envelope: TriageEnvelopeV1) -> TriageEnvelopeV1:
        row = self._conn.execute(
            "SELECT envelope_json FROM envelopes WHERE request_id = ? AND result_version = ?",
            (envelope.request_id, envelope.result_version),
        ).fetchone()
        if row is not None:
            stored = TriageEnvelopeV1.model_validate_json(row["envelope_json"])
            if self._immutable_envelope_data(stored) != self._immutable_envelope_data(envelope):
                raise EnvelopeConflictError(
                    "request_id/result_version already exists with different result content"
                )
            return stored

        self._conn.execute(
            "INSERT INTO envelopes(request_id, result_version, input_hash, envelope_json) "
            "VALUES (?, ?, ?, ?)",
            (
                envelope.request_id,
                envelope.result_version,
                envelope.input_hash,
                _canonical_json(envelope),
            ),
        )
        self._conn.execute(
            "INSERT INTO request_heads(request_id, result_version) VALUES (?, ?) "
            "ON CONFLICT(request_id) DO UPDATE SET result_version = excluded.result_version",
            (envelope.request_id, envelope.result_version),
        )
        return envelope

    def put_envelope(
        self,
        envelope: TriageEnvelopeV1 | dict[str, object],
        *,
        common_facts: CommonFactsV1 | dict[str, object] | None = None,
        run_report: RunReportV2 | dict[str, object] | None = None,
        provider_execution_metadata: list[
            ProviderExecutionMetadataV1 | dict[str, object]
        ] | None = None,
    ) -> TriageEnvelopeV1:
        """Persist a result version as the request's current version.

        Result content is immutable for a given ``request_id`` and
        ``result_version``. Only ``queue_handoff`` may set local queue state.
        Saving a new result version makes that version current, so reviews for
        an older version are rejected as stale.
        """

        value = _as_envelope(envelope)
        if value.delivery_status == "local_queued" or value.local_handoff_id is not None:
            raise HandoffConflictError("local_queued state can only be created by queue_handoff")
        expected_review_status = "pending" if value.requires_human_review else "not_requested"
        if (
            value.review_status != expected_review_status
            or value.delivery_status not in {"pending", "not_requested"}
            or value.resolution_status != "unknown"
        ):
            raise ReviewConflictError(
                "put_envelope accepts only an unreviewed result with unknown resolution state"
            )
        with self._lock, self._transaction():
            row = self._conn.execute(
                "SELECT envelope_json FROM envelopes WHERE request_id = ? AND result_version = ?",
                (value.request_id, value.result_version),
            ).fetchone()
            encoded = _canonical_json(value)
            if row is not None:
                if row["envelope_json"] != encoded:
                    raise EnvelopeConflictError(
                        "request_id/result_version is immutable and already has different content"
                    )
                stored = TriageEnvelopeV1.model_validate_json(row["envelope_json"])
            else:
                stored = self._persist_envelope_unlocked(value)
            if common_facts is not None:
                self._persist_common_facts_unlocked(
                    value.request_id, value.result_version, self._as_common_facts(common_facts)
                )
            if run_report is not None:
                self._persist_run_report_unlocked(
                    value.request_id, value.result_version, self._as_run_report(run_report)
                )
            if provider_execution_metadata is not None:
                self._persist_provider_metadata_unlocked(
                    value.request_id,
                    value.result_version,
                    self._as_provider_metadata(provider_execution_metadata),
                )
        return stored

    @staticmethod
    def _as_common_facts(value: CommonFactsV1 | dict[str, object]) -> CommonFactsV1:
        if isinstance(value, CommonFactsV1):
            return value
        return CommonFactsV1.model_validate(value)

    def _persist_common_facts_unlocked(
        self, request_id: str, result_version: str, facts: CommonFactsV1
    ) -> CommonFactsV1:
        exists = self._conn.execute(
            "SELECT 1 FROM envelopes WHERE request_id = ? AND result_version = ?",
            (request_id, result_version),
        ).fetchone()
        if exists is None:
            raise LocalStoreConflictError("common facts require a stored request/result version")
        encoded = _canonical_json(facts)
        row = self._conn.execute(
            "SELECT facts_json FROM common_facts WHERE request_id = ? AND result_version = ?",
            (request_id, result_version),
        ).fetchone()
        if row is not None:
            if row["facts_json"] != encoded:
                raise EnvelopeConflictError(
                    "common facts are immutable for a request/result version"
                )
            return CommonFactsV1.model_validate_json(row["facts_json"])
        self._conn.execute(
            "INSERT INTO common_facts(request_id, result_version, facts_json) VALUES (?, ?, ?)",
            (request_id, result_version, encoded),
        )
        return facts

    def put_common_facts(
        self,
        request_id: str,
        result_version: str,
        facts: CommonFactsV1 | dict[str, object],
    ) -> CommonFactsV1:
        """Persist the versioned inference projection, without source labels."""

        value = self._as_common_facts(facts)
        with self._lock, self._transaction():
            return self._persist_common_facts_unlocked(request_id, result_version, value)

    def get_common_facts(
        self, request_id: str, result_version: str
    ) -> CommonFactsV1 | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT facts_json FROM common_facts WHERE request_id = ? AND result_version = ?",
                (request_id, result_version),
            ).fetchone()
            if row is None:
                return None
            return CommonFactsV1.model_validate_json(row["facts_json"])

    def get_envelope(
        self, request_id: str, result_version: str | None = None
    ) -> TriageEnvelopeV1 | None:
        with self._lock:
            return self._load_envelope_unlocked(request_id, result_version)

    def queue_handoff(
        self,
        envelope: TriageEnvelopeV1 | dict[str, object],
        *,
        idempotency_scope: str,
        idempotency_key: str,
        target_role: str,
        common_facts: CommonFactsV1 | dict[str, object] | None = None,
        run_report: RunReportV2 | dict[str, object] | None = None,
        provider_execution_metadata: list[
            ProviderExecutionMetadataV1 | dict[str, object]
        ] | None = None,
    ) -> tuple[TriageEnvelopeV1, HandoffRecordV1]:
        """Persist an envelope and local handoff atomically.

        A repeated scoped key with the same input hash returns its original
        stored envelope and handoff. Reusing the key for another input is a
        conflict. No external dispatch is attempted.
        """

        value = _as_envelope(envelope)
        if not value.requires_human_review:
            raise HandoffConflictError("local triage handoff requires human review")
        if value.review_status != "pending" or value.resolution_status != "unknown":
            raise ReviewConflictError("handoff input must be unreviewed with unknown resolution")
        if value.delivery_status == "local_queued" or value.local_handoff_id is not None:
            raise HandoffConflictError(
                "a new handoff must start without a local receipt; retry by idempotency key"
            )
        with self._lock, self._transaction():
            existing = self._conn.execute(
                "SELECT handoff_json, request_id, result_version, input_hash "
                "FROM handoffs WHERE idempotency_scope = ? AND idempotency_key = ?",
                (idempotency_scope, idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["input_hash"] != value.input_hash:
                    raise IdempotencyConflictError(
                        "idempotency scope/key already belongs to a different input_hash"
                    )
                prior_handoff = HandoffRecordV1.model_validate_json(existing["handoff_json"])
                prior_envelope = self._load_envelope_unlocked(
                    str(existing["request_id"]), str(existing["result_version"])
                )
                if prior_envelope is None:
                    raise LocalStoreConflictError("idempotency record has no stored envelope")
                return prior_envelope, prior_handoff

            head = self._conn.execute(
                "SELECT result_version FROM request_heads WHERE request_id = ?",
                (value.request_id,),
            ).fetchone()
            if head is not None and head["result_version"] != value.result_version:
                raise StaleResultError(
                    "cannot queue a handoff for a result version that is no longer current"
                )

            stored = self._persist_envelope_unlocked(value)
            if common_facts is not None:
                self._persist_common_facts_unlocked(
                    value.request_id,
                    value.result_version,
                    self._as_common_facts(common_facts),
                )
            if run_report is not None:
                self._persist_run_report_unlocked(
                    value.request_id, value.result_version, self._as_run_report(run_report)
                )
            if provider_execution_metadata is not None:
                self._persist_provider_metadata_unlocked(
                    value.request_id,
                    value.result_version,
                    self._as_provider_metadata(provider_execution_metadata),
                )
            existing_handoff = self._conn.execute(
                "SELECT handoff_id FROM handoffs WHERE request_id = ? AND result_version = ?",
                (value.request_id, value.result_version),
            ).fetchone()
            if existing_handoff is not None:
                raise HandoffConflictError("this result version already has a local handoff")

            handoff_token = uuid.uuid4().hex
            handoff = HandoffRecordV1(
                schema_version="handoff-record-v1",
                handoff_id=f"handoff-{handoff_token}",
                request_id=value.request_id,
                result_version=value.result_version,
                target_role=target_role,
                delivery_status="local_queued",
                local_receipt_id=f"local-receipt-{handoff_token}",
                idempotency_scope=idempotency_scope,
                idempotency_key=idempotency_key,
                input_hash=value.input_hash,
                created_at=_now_utc(),
            )
            queued_envelope = stored.model_copy(
                update={
                    "delivery_status": "local_queued",
                    "local_handoff_id": handoff.handoff_id,
                }
            )
            self._conn.execute(
                "UPDATE envelopes SET envelope_json = ? WHERE request_id = ? AND result_version = ?",
                (_canonical_json(queued_envelope), value.request_id, value.result_version),
            )
            self._conn.execute(
                "INSERT INTO handoffs(handoff_id, request_id, result_version, "
                "idempotency_scope, idempotency_key, input_hash, handoff_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    handoff.handoff_id,
                    handoff.request_id,
                    handoff.result_version,
                    handoff.idempotency_scope,
                    handoff.idempotency_key,
                    handoff.input_hash,
                    _canonical_json(handoff),
                ),
            )
            return queued_envelope, handoff

    def get_handoff(self, handoff_id: str) -> HandoffRecordV1 | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT handoff_json FROM handoffs WHERE handoff_id = ?", (handoff_id,)
            ).fetchone()
            if row is None:
                return None
            return HandoffRecordV1.model_validate_json(row["handoff_json"])

    def get_handoff_bundle(
        self, handoff_id: str
    ) -> tuple[TriageEnvelopeV1, HandoffRecordV1] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT handoff_json, request_id, result_version FROM handoffs WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
            if row is None:
                return None
            envelope = self._load_envelope_unlocked(
                str(row["request_id"]), str(row["result_version"])
            )
            if envelope is None:
                raise LocalStoreConflictError("handoff record has no stored envelope")
            return envelope, HandoffRecordV1.model_validate_json(row["handoff_json"])

    def list_queued_handoffs(
        self,
    ) -> list[tuple[TriageEnvelopeV1, HandoffRecordV1]]:
        """Return persisted local queue entries in creation order."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT handoff_id FROM handoffs ORDER BY rowid"
            ).fetchall()
            bundles = [self.get_handoff_bundle(str(row["handoff_id"])) for row in rows]
            return [bundle for bundle in bundles if bundle is not None]

    def list_current_review_queue(
        self,
    ) -> list[tuple[TriageEnvelopeV1, HandoffRecordV1 | None]]:
        """Return pending human-review request heads, including corrected heads.

        The handoff table is a receipt history. A corrected result creates a
        new current request head without rewriting or fabricating a handoff,
        so a reviewer view must be based on ``request_heads`` and only attach
        a handoff when it belongs to that exact current result version.
        """

        with self._lock:
            rows = self._conn.execute(
                "SELECT e.envelope_json, h.handoff_json "
                "FROM request_heads AS head "
                "JOIN envelopes AS e ON e.request_id = head.request_id "
                "AND e.result_version = head.result_version "
                "LEFT JOIN handoffs AS h ON h.request_id = head.request_id "
                "AND h.result_version = head.result_version "
                "ORDER BY e.rowid"
            ).fetchall()
            items: list[tuple[TriageEnvelopeV1, HandoffRecordV1 | None]] = []
            for row in rows:
                envelope = TriageEnvelopeV1.model_validate_json(row["envelope_json"])
                if not envelope.requires_human_review or envelope.review_status not in {
                    "pending",
                    "deferred",
                }:
                    continue
                handoff_json = row["handoff_json"]
                handoff = (
                    HandoffRecordV1.model_validate_json(handoff_json)
                    if handoff_json is not None
                    else None
                )
                items.append((envelope, handoff))
            return items

    @staticmethod
    def _review_payload(review: ReviewRecordV1) -> str:
        return _canonical_json(review)

    def append_review(
        self, review: ReviewRecordV1 | dict[str, object]
    ) -> ReviewRecordV1:
        """Append one review event and update only the envelope's review status.

        ``actor_origin`` is retained exactly as supplied; it is not proof of
        identity. Defer events may be followed by a new decision for the same
        current result. Accepted/corrected results are terminal for that version.
        """

        value = _as_review(review)
        if value.decision == "correct" and not (value.reason and value.reason.strip()):
            raise ReviewConflictError("correct decision requires a non-empty reason")
        payload = self._review_payload(value)
        with self._lock, self._transaction():
            existing = self._conn.execute(
                "SELECT review_json FROM reviews WHERE review_id = ?", (value.review_id,)
            ).fetchone()
            if existing is not None:
                if existing["review_json"] != payload:
                    raise ReviewConflictError("review_id already exists with different event content")
                return ReviewRecordV1.model_validate_json(existing["review_json"])

            head = self._conn.execute(
                "SELECT result_version FROM request_heads WHERE request_id = ?",
                (value.request_id,),
            ).fetchone()
            if head is None:
                raise StaleResultError("review target request/result version is not stored")
            if head["result_version"] != value.result_version:
                raise StaleResultError(
                    "review result_version is stale; review the current stored result version"
                )
            envelope = self._load_envelope_unlocked(value.request_id, value.result_version)
            if envelope is None:
                raise StaleResultError("review target result version is not stored")
            if envelope.result is None:
                raise ReviewConflictError("a result without TriageResult cannot be reviewed")
            if envelope.result.model_dump(mode="json") != value.before_result.model_dump(mode="json"):
                raise ReviewConflictError("before_result does not match the stored result version")
            if envelope.review_status in {"accepted", "corrected"}:
                raise ReviewConflictError(
                    f"result version already has terminal review status {envelope.review_status}"
                )

            status = {
                "accept": "accepted",
                "correct": "corrected",
                "defer": "deferred",
            }[value.decision]
            self._conn.execute(
                "INSERT INTO reviews(review_id, request_id, result_version, reviewed_at, review_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    value.review_id,
                    value.request_id,
                    value.result_version,
                    value.reviewed_at,
                    payload,
                ),
            )
            # Model/test-origin events are retained as evidence but cannot advance
            # the human workflow state or create a corrected human result version.
            if value.actor_type != "local_user":
                return value

            updated = envelope.model_copy(update={"review_status": status})
            self._conn.execute(
                "UPDATE envelopes SET envelope_json = ? WHERE request_id = ? AND result_version = ?",
                (_canonical_json(updated), value.request_id, value.result_version),
            )
            if value.decision == "correct":
                assert value.after_result is not None
                corrected_version = f"{value.result_version}:correction:{value.review_id}"
                needs_review = (
                    envelope.requires_human_review or value.after_result.category == "UNKNOWN"
                )
                corrected = envelope.model_copy(
                    update={
                        "result_version": corrected_version,
                        "created_at": _now_utc(),
                        "result": value.after_result,
                        "requires_human_review": needs_review,
                        "review_status": "pending" if needs_review else "not_requested",
                        "delivery_status": "pending",
                        "local_handoff_id": None,
                    }
                )
                self._persist_envelope_unlocked(corrected)
                common_facts_row = self._conn.execute(
                    "SELECT facts_json FROM common_facts WHERE request_id = ? AND result_version = ?",
                    (value.request_id, value.result_version),
                ).fetchone()
                if common_facts_row is not None:
                    self._persist_common_facts_unlocked(
                        value.request_id,
                        corrected_version,
                        CommonFactsV1.model_validate_json(common_facts_row["facts_json"]),
                    )
        return value

    def list_reviews(
        self, request_id: str, result_version: str | None = None
    ) -> list[ReviewRecordV1]:
        with self._lock:
            if result_version is None:
                rows = self._conn.execute(
                    "SELECT review_json FROM reviews WHERE request_id = ? ORDER BY rowid",
                    (request_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT review_json FROM reviews WHERE request_id = ? AND result_version = ? "
                    "ORDER BY rowid",
                    (request_id, result_version),
                ).fetchall()
            return [ReviewRecordV1.model_validate_json(row["review_json"]) for row in rows]

    def render_handoff_json(self, handoff_id: str) -> str:
        bundle = self.get_handoff_bundle(handoff_id)
        if bundle is None:
            raise KeyError(f"unknown handoff_id: {handoff_id}")
        envelope, handoff = bundle
        common_facts = self.get_common_facts(envelope.request_id, envelope.result_version)
        return json.dumps(
            {
                "handoff": handoff.model_dump(mode="json"),
                "envelope": envelope.model_dump(mode="json"),
                "common_facts": (
                    common_facts.model_dump(mode="json") if common_facts is not None else None
                ),
                "trust_boundary": (
                    "Persisted locally; local_queued does not mean an external person or system received it."
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )

    def render_handoff_markdown(self, handoff_id: str) -> str:
        bundle = self.get_handoff_bundle(handoff_id)
        if bundle is None:
            raise KeyError(f"unknown handoff_id: {handoff_id}")
        envelope, handoff = bundle
        result = envelope.result
        facts = self.get_common_facts(envelope.request_id, envelope.result_version)
        lines = [
            f"# Triage handoff {handoff.handoff_id}",
            "",
            f"- Delivery: `local_queued` (stored locally; not externally delivered)",
            f"- Local receipt: `{handoff.local_receipt_id}`",
            f"- Target role: `{handoff.target_role}` (role suggestion, not a named recipient)",
            f"- Request/result: `{envelope.request_id}` / `{envelope.result_version}`",
            f"- Input hash: `{envelope.input_hash}`",
            "",
            "## Submitted problem facts",
        ]
        if facts is None:
            lines.append("- No CommonFactsV1 projection is stored for this result version.")
        else:
            lines.extend(
                [
                    f"- Device/software: `{_md_inline(facts.device)}` / `{_md_inline(facts.sw_version)}`",
                    f"- Failure: {_md_inline(facts.assertion)}",
                    f"- Result: `{facts.result}`; retry: `{facts.retry}`",
                    f"- Previous runs: {', '.join(facts.previous_runs) or 'none recorded'}",
                    "- Environment:",
                ]
            )
            if facts.env:
                lines.extend(
                    f"  - `{_md_inline(key)}`: `{_md_inline(str(value))}`"
                    for key, value in sorted(facts.env.items())
                )
            else:
                lines.append("  - none recorded")
            fence = _code_fence(facts.syslog_excerpt)
            lines.extend(
                [
                    "- Syslog excerpt:",
                    f"  {fence}text",
                    facts.syslog_excerpt,
                    f"  {fence}",
                ]
            )
        lines.extend(
            [
                "",
                "## Suggested result",
            ]
        )
        if result is None:
            lines.append("- No completed triage result is stored.")
        else:
            lines.extend(
                [
                    f"- Category: `{result.category}`",
                    f"- Action: `{result.action}`",
                    f"- Priority: `{result.priority}`",
                    f"- Confidence: `{result.confidence:.3f}`",
                    f"- Reason: {result.reason}",
                ]
            )
        lines.extend(["", "## Attempted stages"])
        if envelope.route_steps:
            for step in envelope.route_steps:
                detail = f"{step.stage}: {step.status}; attempts={step.attempts}"
                if step.reason_code:
                    detail += f"; reason={step.reason_code}"
                lines.append(f"- {detail}")
        else:
            lines.append("- No route steps recorded.")
        lines.extend(["", "## Evidence references"])
        if envelope.evidence_refs:
            for ref in envelope.evidence_refs:
                lines.append(
                    f"- `{ref.evidence_id}` ({ref.evidence_kind}) at `{ref.field_path}`; "
                    f"sha256 `{ref.content_sha256}`"
                )
        else:
            lines.append("- No evidence references recorded.")
        lines.extend(["", "## Unresolved or missing evidence"])
        if envelope.missing_evidence:
            lines.extend(f"- {item}" for item in envelope.missing_evidence)
        else:
            lines.append("- No missing evidence items recorded by the triage run.")
        lines.extend(
            [
                "",
                "## Next step",
                "Review the stored suggestion and evidence references; confirm facts before acting.",
                "",
                "This card contains only fields present in the stored envelope. The local receipt is not an acknowledgement.",
                "",
            ]
        )
        return "\n".join(lines)

    def export_handoff_json(self, handoff_id: str, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render_handoff_json(handoff_id) + "\n", encoding="utf-8")
        return target

    def export_handoff_markdown(self, handoff_id: str, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render_handoff_markdown(handoff_id), encoding="utf-8")
        return target
