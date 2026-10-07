from __future__ import annotations

import anyio
import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session

from jevtest.app_service import TriageApplicationService
from jevtest.local_store import LocalStore
from jevtest.mcp_server import create_mcp_server


def _state_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_id": "SYN-MCP-0001",
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
    payload.update(overrides)
    return payload


def _run(async_fn):
    return anyio.run(async_fn)


def _structured(result):
    value = getattr(result, "structuredContent", None)
    if value is None:
        value = getattr(result, "structured_content", None)
    return value


def test_stdio_tool_surface_schema_annotations_and_local_round_trip():
    async def scenario():
        with LocalStore(":memory:") as store:
            server = create_mcp_server(TriageApplicationService(store))
            async with create_connected_server_and_client_session(server) as client:
                listed = await client.list_tools()
                tools = {tool.name: tool for tool in listed.tools}
                assert set(tools) == {
                    "triage_failure",
                    "get_triage_result",
                    "get_triage_evidence",
                }

                write_tool = tools["triage_failure"]
                assert write_tool.annotations.readOnlyHint is False
                assert write_tool.annotations.idempotentHint is True
                assert write_tool.annotations.openWorldHint is False
                schema = write_tool.inputSchema
                state_schema = schema["properties"]["failure_state"]
                assert set(state_schema["properties"]) == {
                    "case_id",
                    "result",
                    "previous_runs",
                    "device",
                    "sw_version",
                    "assertion",
                    "syslog_excerpt",
                    "env",
                    "retry",
                    "provenance",
                }
                assert "failure_state" in schema["required"]
                assert "idempotency_key" in schema["required"]
                assert set(state_schema["required"]) == set(state_schema["properties"])
                assert state_schema["additionalProperties"] is False
                result_schema = state_schema["properties"]["result"]
                assert result_schema.get("const") == "FAIL" or result_schema.get("enum") == ["FAIL"]
                provenance_schema = state_schema["properties"]["provenance"]
                assert set(provenance_schema["required"]) == {"dataset_version", "synthetic"}
                assert provenance_schema["additionalProperties"] is False
                assert tools["get_triage_result"].annotations.readOnlyHint is True
                assert tools["get_triage_evidence"].annotations.readOnlyHint is True

                first = await client.call_tool(
                    "triage_failure",
                    {"failure_state": _state_payload(), "idempotency_key": "mcp-idem-1"},
                )
                first_data = _structured(first)
                assert not first.isError
                assert first_data["storage_scope"] == "local_only"
                assert first_data["replayed"] is False
                envelope = first_data["envelope"]
                assert envelope["execution_mode"] == "offline"
                assert envelope["delivery_status"] == "local_queued"
                assert envelope["resolution_status"] != "resolved"
                assert envelope["requires_human_review"] is True

                replay = await client.call_tool(
                    "triage_failure",
                    {"failure_state": _state_payload(), "idempotency_key": "mcp-idem-1"},
                )
                replay_data = _structured(replay)
                assert not replay.isError
                assert replay_data["replayed"] is True
                assert replay_data["envelope"]["request_id"] == envelope["request_id"]
                assert replay_data["envelope"]["result_version"] == envelope["result_version"]

                fetched = await client.call_tool(
                    "get_triage_result",
                    {
                        "request_id": envelope["request_id"],
                        "result_version": envelope["result_version"],
                    },
                )
                assert not fetched.isError
                assert _structured(fetched)["found"] is True

                evidence = await client.call_tool(
                    "get_triage_evidence",
                    {
                        "request_id": envelope["request_id"],
                        "result_version": envelope["result_version"],
                        "field_path": "assertion",
                    },
                )
                evidence_data = _structured(evidence)
                assert not evidence.isError
                assert evidence_data["value"] == "association did not complete"
                assert evidence_data["root_cause_verified"] is False

                missing_result = await client.call_tool(
                    "get_triage_result", {"request_id": "absent"}
                )
                assert _structured(missing_result) == {"found": False, "envelope": None}

    _run(scenario)


def test_invalid_payload_and_conflict_errors_are_stable_and_do_not_echo_content():
    async def scenario():
        with LocalStore(":memory:") as store:
            server = create_mcp_server(TriageApplicationService(store))
            async with create_connected_server_and_client_session(server) as client:
                sentinel = "PRIVATE-SYSLOG-DO-NOT-ECHO"
                invalid = _state_payload(
                    result="PASS",
                    untrusted_path={"value": sentinel},
                )
                result = await client.call_tool(
                    "triage_failure",
                    {"failure_state": invalid, "idempotency_key": "mcp-invalid"},
                )
                error_text = " ".join(block.text for block in result.content if hasattr(block, "text"))
                assert result.isError is True
                assert "E_INPUT_INVALID" in error_text
                assert "failure_state or idempotency_key is invalid" in error_text
                assert sentinel not in error_text
                assert "untrusted_path" not in error_text

                # FastMCP's own argument-model errors are redacted too, including
                # missing required arguments whose error would otherwise include
                # the entire supplied failure_state object.
                missing_key = await client.call_tool(
                    "triage_failure",
                    {"failure_state": _state_payload(assertion=sentinel)},
                )
                missing_key_text = " ".join(
                    block.text for block in missing_key.content if hasattr(block, "text")
                )
                assert missing_key.isError is True
                assert "Field required" in missing_key_text
                assert sentinel not in missing_key_text
                assert "input_value" not in missing_key_text

                wrong_key_type = await client.call_tool(
                    "triage_failure",
                    {"failure_state": _state_payload(), "idempotency_key": 123456},
                )
                wrong_key_text = " ".join(
                    block.text for block in wrong_key_type.content if hasattr(block, "text")
                )
                assert wrong_key_type.isError is True
                assert "valid string" in wrong_key_text
                assert "input_value" not in wrong_key_text

                await client.call_tool(
                    "triage_failure",
                    {"failure_state": _state_payload(), "idempotency_key": "mcp-conflict"},
                )
                conflict = await client.call_tool(
                    "triage_failure",
                    {
                        "failure_state": _state_payload(assertion="different input"),
                        "idempotency_key": "mcp-conflict",
                    },
                )
                conflict_text = " ".join(
                    block.text for block in conflict.content if hasattr(block, "text")
                )
                assert conflict.isError is True
                assert "E_IDEMPOTENCY_CONFLICT" in conflict_text
                assert "different input" not in conflict_text

                missing_evidence = await client.call_tool(
                    "get_triage_evidence",
                    {
                        "request_id": "absent",
                        "result_version": "missing-version",
                        "field_path": "assertion",
                    },
                )
                missing_text = " ".join(
                    block.text for block in missing_evidence.content if hasattr(block, "text")
                )
                assert missing_evidence.isError is True
                assert "E_EVIDENCE_NOT_FOUND" in missing_text
                assert "absent" not in missing_text

    _run(scenario)
