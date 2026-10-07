"""L2 real-process integration test for the local stdio MCP server.

This exercises the official MCP SDK client against ``python -m jevtest.mcp_server``.
It is not a Codex-host launch/configuration test and never calls a live provider.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("anyio")

import anyio
from mcp import ClientSession
import mcp.client.stdio as mcp_stdio
from mcp.client.stdio import StdioServerParameters, stdio_client


_ROOT = Path(__file__).resolve().parents[1]
_LIVE_ENV_KEYS = {
    "TYPESAFE_API_KEY",
    "ARK_API_KEY",
    "JEVTEST_PROVIDERS_CONFIG",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
}


def _failure_state() -> dict[str, object]:
    return {
        "case_id": "SYN-MCP-PROCESS-0001",
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


def _child_environment(database_path: Path) -> dict[str, str]:
    # A narrow allowlist preserves only what Python needs to start on the host.
    # MCP SDK 1.x also inherits its documented safe OS-variable allowlist; that
    # list does not contain provider keys, API endpoints, or proxy settings.
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        if key in os.environ
    }
    env.update(
        {
            "PYTHONPATH": str(_ROOT / "src"),
            "PYTHONIOENCODING": "utf-8",
            "JEVTEST_DB_PATH": str(database_path),
        }
    )
    assert not (_LIVE_ENV_KEYS & env.keys())
    return env


def _text(result: object) -> str:
    return " ".join(
        block.text
        for block in getattr(result, "content", [])
        if hasattr(block, "text")
    )


def _structured(result: object) -> dict[str, object]:
    assert getattr(result, "isError", False) is False, _text(result)
    value = getattr(result, "structuredContent", None)
    if value is None:
        value = getattr(result, "structured_content", None)
    assert isinstance(value, dict)
    return value


def test_local_mcp_stdio_subprocess_round_trip_and_clean_shutdown(tmp_path, monkeypatch):
    """L2: launch the actual stdio process; MCP framing proves stdout stays protocol-only."""

    async def scenario() -> None:
        database_path = tmp_path / "mcp-triage.sqlite3"
        server_env = _child_environment(database_path)
        server_stderr_path = tmp_path / "mcp-server.stderr.log"
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "jevtest.mcp_server"],
            cwd=str(_ROOT),
            env=server_env,
        )

        created_processes = []
        child_environments = []
        original_create_process = mcp_stdio._create_platform_compatible_process

        async def track_process(*args, **kwargs):
            child_environments.append(dict(kwargs.get("env") or {}))
            process = await original_create_process(*args, **kwargs)
            created_processes.append(process)
            return process

        monkeypatch.setattr(
            mcp_stdio, "_create_platform_compatible_process", track_process
        )

        with server_stderr_path.open("w+", encoding="utf-8") as server_stderr:
            async with stdio_client(server_params, errlog=server_stderr) as streams:
                async with ClientSession(*streams) as session:
                    initialized = await session.initialize()
                    assert initialized.serverInfo.name == "JevTest Offline Triage"

                    listed = await session.list_tools()
                    tools = {tool.name: tool for tool in listed.tools}
                    assert set(tools) == {
                        "triage_failure",
                        "get_triage_result",
                        "get_triage_evidence",
                    }
                    assert tools["triage_failure"].annotations.readOnlyHint is False
                    assert tools["triage_failure"].annotations.idempotentHint is True
                    assert tools["get_triage_result"].annotations.readOnlyHint is True
                    assert tools["get_triage_evidence"].annotations.readOnlyHint is True

                    first = await session.call_tool(
                        "triage_failure",
                        {
                            "failure_state": _failure_state(),
                            "idempotency_key": "mcp-process-key-1",
                        },
                    )
                    first_data = _structured(first)
                    envelope = first_data["envelope"]
                    assert isinstance(envelope, dict)
                    assert first_data["replayed"] is False
                    assert envelope["execution_mode"] == "offline"
                    assert envelope["requires_human_review"] is True
                    assert envelope["delivery_status"] == "local_queued"
                    assert envelope["resolution_status"] != "resolved"

                    replay = await session.call_tool(
                        "triage_failure",
                        {
                            "failure_state": _failure_state(),
                            "idempotency_key": "mcp-process-key-1",
                        },
                    )
                    replay_data = _structured(replay)
                    assert replay_data["replayed"] is True
                    assert replay_data["envelope"]["request_id"] == envelope["request_id"]
                    assert replay_data["envelope"]["result_version"] == envelope["result_version"]

                    fetched = await session.call_tool(
                        "get_triage_result",
                        {
                            "request_id": envelope["request_id"],
                            "result_version": envelope["result_version"],
                        },
                    )
                    assert _structured(fetched)["found"] is True

                    evidence = await session.call_tool(
                        "get_triage_evidence",
                        {
                            "request_id": envelope["request_id"],
                            "result_version": envelope["result_version"],
                            "field_path": "assertion",
                        },
                    )
                    evidence_data = _structured(evidence)
                    assert evidence_data["value"] == "association did not complete"
                    assert evidence_data["root_cause_verified"] is False

                    sentinel = "MCP-PROCESS-SECRET-SENTINEL"
                    invalid_state = _failure_state()
                    invalid_state["result"] = "PASS"
                    invalid_state["untrusted_path"] = sentinel
                    invalid = await session.call_tool(
                        "triage_failure",
                        {
                            "failure_state": invalid_state,
                            "idempotency_key": "mcp-process-invalid",
                        },
                    )
                    error_text = _text(invalid)
                    assert invalid.isError is True
                    assert "E_INPUT_INVALID" in error_text
                    assert sentinel not in error_text
                    assert "untrusted_path" not in error_text

        # stdio_client owns graceful stdin close and waits for the child before
        # returning. Retain the SDK-created process handle to verify clean exit.
        assert len(created_processes) == 1
        assert created_processes[0].returncode == 0
        assert len(child_environments) == 1
        assert not (_LIVE_ENV_KEYS & child_environments[0].keys())
        assert server_env["JEVTEST_DB_PATH"] == str(database_path)
        assert database_path.is_file()

    anyio.run(scenario)
