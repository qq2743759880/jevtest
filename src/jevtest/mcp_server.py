"""Local stdio MCP tools for the offline triage application service.

Targets the official ``mcp`` Python SDK 1.x API currently installed in the workspace.
The MCP SDK is optional for the core CLI package; the local MCP launcher fails closed if
the SDK is unavailable or is on an unsupported major version.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict
from pydantic_core import core_schema

from jevtest.app_service import (
    ApplicationIdempotencyConflict,
    SubmittedFactNotFound,
    TriageApplicationService,
    UnsupportedApplicationPolicy,
)
from jevtest.execution_contracts import RunReportV2, TriageEnvelopeV1
from jevtest.local_store import LocalStore, LocalStoreConflictError
from jevtest.models import FailureState


class TriageFailureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    envelope: TriageEnvelopeV1
    report: RunReportV2
    replayed: bool
    storage_scope: Literal["local_only"] = "local_only"


class TriageResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    envelope: TriageEnvelopeV1 | None


class TriageEvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    result_version: str
    field_path: str
    value: Any
    content_sha256: str
    capture_scope: str
    root_cause_verified: Literal[False] = False


class _FailureStateInput:
    """Preserve FailureState's public schema while deferring validation to the handler.

    FastMCP validates function arguments before invoking a tool and its generic
    error wrapper includes Pydantic validation text. Accept the raw JSON value at
    that boundary, advertise the actual FailureState JSON Schema, then validate
    inside the handler where errors can be mapped to a fixed safe code/message.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> dict[str, Any]:
        generated = FailureState.model_json_schema()
        definitions = generated.pop("$defs", {})

        def inline(node: Any) -> Any:
            if isinstance(node, dict):
                ref = node.get("$ref")
                if isinstance(ref, str) and ref.startswith("#/$defs/"):
                    definition = definitions.get(ref.removeprefix("#/$defs/"))
                    if definition is not None:
                        return inline(definition)
                return {key: inline(value) for key, value in node.items()}
            if isinstance(node, list):
                return [inline(value) for value in node]
            return node

        return inline(generated)


def _require_supported_sdk() -> None:
    try:
        installed_version = version("mcp")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Local MCP requires the official 'mcp' Python SDK (currently targets mcp<2)."
        ) from exc
    major = installed_version.split(".", 1)[0]
    if major != "1":
        raise RuntimeError(
            f"Local MCP targets the official mcp 1.x API; found mcp {installed_version}."
        )


def _raise_tool_error(code: str, message: str) -> NoReturn:
    from mcp.server.fastmcp.exceptions import ToolError

    # Only fixed, reviewed text crosses the MCP error boundary. Never include request data,
    # provider output, exception strings, local paths, or SQLite diagnostics.
    raise ToolError(f"{code}: {message}") from None


def _redact_sdk_argument_errors(server: Any, tool_names: tuple[str, ...]) -> None:
    """Hide user inputs in FastMCP's generated per-tool argument validation errors.

    Official MCP SDK 1.x exposes these through its FastMCP tool manager. Keep the
    change scoped to this server's three tools; do not mutate the SDK's global
    ArgModelBase configuration.
    """
    manager = getattr(server, "_tool_manager", None)
    get_tool = getattr(manager, "get_tool", None)
    if not callable(get_tool):
        raise RuntimeError("Installed MCP SDK does not expose the supported FastMCP tool manager API")
    for name in tool_names:
        tool = get_tool(name)
        argument_model = getattr(getattr(tool, "fn_metadata", None), "arg_model", None)
        if argument_model is None or not hasattr(argument_model, "model_rebuild"):
            raise RuntimeError("Installed MCP SDK does not expose per-tool argument validation models")
        argument_model.model_config = {
            **argument_model.model_config,
            "hide_input_in_errors": True,
        }
        argument_model.model_rebuild(force=True)


def create_mcp_server(application: TriageApplicationService):
    """Create a stdio-capable MCP server around an injected offline application service."""
    _require_supported_sdk()
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP(
        "JevTest Offline Triage",
        instructions=(
            "Offline-only local JevTest tools. triage_failure writes a result and any required "
            "handoff to local SQLite. Read tools never modify state. No tool contacts external "
            "services or approves a result. A local handoff receipt is not external delivery."
        ),
    )

    @server.tool(
        name="triage_failure",
        title="Run offline triage",
        description=(
            "WRITE: validate a complete FailureState v1 object, run the configured offline "
            "Rule/Jev-mock/LLM-stub cascade, and persist the result plus a local handoff when "
            "human review is required. Requires a non-empty idempotency_key; same key and same "
            "input replays the stored result, while changed input is rejected. This never calls "
            "a live provider, sends an external ticket/message, resolves the fault, or records "
            "a human approval."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        structured_output=True,
    )
    def triage_failure(
        failure_state: _FailureStateInput, idempotency_key: str
    ) -> TriageFailureResponse:
        """Persist one offline result, mapping invalid content to a safe tool error."""
        try:
            state = FailureState.model_validate(failure_state)
            outcome = application.submit(state, idempotency_key=idempotency_key)
            return TriageFailureResponse(
                envelope=outcome.envelope,
                report=outcome.report,
                replayed=outcome.replayed,
            )
        except (ApplicationIdempotencyConflict, LocalStoreConflictError):
            _raise_tool_error(
                "E_IDEMPOTENCY_CONFLICT",
                "the idempotency key conflicts with an existing local request",
            )
        except UnsupportedApplicationPolicy:
            _raise_tool_error("E_POLICY_UNSUPPORTED", "only offline triage policies are allowed")
        except ValueError:
            _raise_tool_error("E_INPUT_INVALID", "failure_state or idempotency_key is invalid")
        except Exception:
            _raise_tool_error("E_INTERNAL", "the local triage request could not be completed")

    @server.tool(
        name="get_triage_result",
        title="Read triage result",
        description=(
            "READ ONLY: fetch a stored result by request_id and optional result_version. "
            "Returns found=false when no such local result exists."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
        structured_output=True,
    )
    def get_triage_result(
        request_id: str, result_version: str | None = None
    ) -> TriageResultResponse:
        """Read a locally stored result without changing it."""
        if not request_id.strip():
            _raise_tool_error("E_INPUT_INVALID", "request_id must be non-empty")
        try:
            envelope = application.get_result(request_id, result_version)
            return TriageResultResponse(found=envelope is not None, envelope=envelope)
        except Exception:
            _raise_tool_error("E_INTERNAL", "the local result could not be read")

    @server.tool(
        name="get_triage_evidence",
        title="Read submitted triage evidence",
        description=(
            "READ ONLY: fetch one field from the stored CommonFacts projection by request_id, "
            "result_version, and field_path (for example 'assertion' or 'env.AP_reachable'). "
            "The returned field is user-submitted evidence, not independent root-cause proof."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
        structured_output=True,
    )
    def get_triage_evidence(
        request_id: str, result_version: str, field_path: str
    ) -> TriageEvidenceResponse:
        """Read one locally stored submitted fact with its hash and capture scope."""
        if (
            not request_id.strip()
            or not result_version.strip()
            or not field_path.strip()
        ):
            _raise_tool_error("E_INPUT_INVALID", "request_id, result_version, and field_path are required")
        try:
            evidence = application.lookup_submitted_facts_evidence(
                request_id, result_version, field_path
            )
            return TriageEvidenceResponse(
                request_id=evidence.request_id,
                result_version=evidence.result_version,
                field_path=evidence.field_path,
                value=evidence.value,
                content_sha256=evidence.content_sha256,
                capture_scope=evidence.capture_scope,
                root_cause_verified=False,
            )
        except SubmittedFactNotFound:
            _raise_tool_error("E_EVIDENCE_NOT_FOUND", "the submitted field was not found")
        except Exception:
            _raise_tool_error("E_INTERNAL", "the submitted field could not be read")

    _redact_sdk_argument_errors(
        server,
        ("triage_failure", "get_triage_result", "get_triage_evidence"),
    )
    return server


def main() -> None:
    """Run the one-host local stdio server; the optional DB override is process-local config."""
    database_path = os.environ.get("JEVTEST_DB_PATH") or str(
        Path.home() / ".jevtest" / "triage.sqlite3"
    )
    store = LocalStore(database_path)
    try:
        application = TriageApplicationService(store)
        create_mcp_server(application).run(transport="stdio")
    finally:
        store.close()


if __name__ == "__main__":
    main()
