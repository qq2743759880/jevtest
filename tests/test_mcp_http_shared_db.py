"""Offline cross-entry proof that MCP and HTTP review one shared SQLite record."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("anyio")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import anyio
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from jevtest.app_service import TriageApplicationService
from jevtest.http_api import create_app
from jevtest.local_store import LocalStore

_ROOT = Path(__file__).resolve().parents[1]
_HOST = "127.0.0.1:8765"
_ORIGIN = f"http://{_HOST}"


def _failure_state() -> dict[str, object]:
    return {
        "case_id": "SYN-CROSS-ENTRY-0001",
        "result": "FAIL",
        "previous_runs": ["PASS", "PASS", "FAIL"],
        "device": "AP-660",
        "sw_version": "V6R1",
        "assertion": "association did not complete",
        "syslog_excerpt": "no matching cause",
        "env": {"AP_reachable": True, "AC_reachable": True},
        "retry": "failed",
        "provenance": {"dataset_version": "synthetic-v1", "synthetic": True},
    }


def _child_env(database: Path) -> dict[str, str]:
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        if key in os.environ
    }
    env.update(
        {
            "PYTHONPATH": str(_ROOT / "src"),
            "PYTHONIOENCODING": "utf-8",
            "JEVTEST_DB_PATH": str(database),
        }
    )
    assert not {
        "TYPESAFE_API_KEY",
        "ARK_API_KEY",
        "JEVTEST_PROVIDERS_CONFIG",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    } & env.keys()
    return env


def test_mcp_created_request_is_reviewed_through_http_on_same_db(tmp_path: Path) -> None:
    database = tmp_path / "shared-demo.sqlite3"

    async def create_through_mcp() -> dict[str, object]:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "jevtest.mcp_server"],
            cwd=str(_ROOT),
            env=_child_env(database),
        )
        async with stdio_client(parameters) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(
                    "triage_failure",
                    {
                        "failure_state": _failure_state(),
                        "idempotency_key": "cross-entry-once",
                    },
                )
                assert result.isError is False
                payload = getattr(result, "structuredContent", None)
                if payload is None:
                    payload = getattr(result, "structured_content", None)
                assert isinstance(payload, dict)
                return payload

    created = anyio.run(create_through_mcp)
    envelope = created["envelope"]
    assert isinstance(envelope, dict)
    request_id = str(envelope["request_id"])

    # The HTTP adapter opens the exact MCP-written database and reviews the
    # returned request id; this test never resubmits the failure state through HTTP.
    with LocalStore(database) as store:
        service = TriageApplicationService(store, idempotency_scope="http-app")
        with TestClient(
            create_app(service, expected_host=_HOST),
            base_url=_ORIGIN,
            client=("127.0.0.1", 54321),
        ) as client:
            token = client.get("/v1/session").json()["csrf_token"]
            queue = client.get("/v1/review-queue")
            assert queue.status_code == 200
            item = next(
                row for row in queue.json()["items"] if row["envelope"]["request_id"] == request_id
            )
            assert item["envelope"]["result_version"] == envelope["result_version"]
            assert item["envelope"]["execution_mode"] == "offline"
            assert item["handoff"]["result_version"] == envelope["result_version"]

            headers = {
                "Origin": _ORIGIN,
                "X-CSRF-Token": token,
                "Idempotency-Key": "cross-entry-review-once",
                "If-Match": f'"{envelope["result_version"]}"',
            }
            reviewed = client.post(
                f"/v1/triage/{request_id}/reviews",
                json={
                    "decision": "accept",
                    "operator_label": "test-reviewer",
                    "confirmation": f"accept {request_id[:8]}",
                },
                headers=headers,
            )
            assert reviewed.status_code == 200
            assert reviewed.json()["envelope"]["request_id"] == request_id
            assert reviewed.json()["envelope"]["review_status"] == "accepted"
            assert client.get("/v1/review-queue").json()["count"] == 0
