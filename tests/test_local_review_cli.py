from __future__ import annotations

import json
import io
from pathlib import Path

import pytest

from jevtest.cli import main as cli_main
from jevtest.local_store import LocalStore
from jevtest.models import TriageResult

_REPO_ROOT = Path(__file__).resolve().parents[1]
_POLICY = _REPO_ROOT / "config" / "cascade-demo-v1.yaml"


def _failure(*, direct_rule: bool = False) -> dict:
    return {
        "case_id": "LOCAL-CLI-R2" if direct_rule else "LOCAL-CLI-R4",
        "result": "FAIL",
        "previous_runs": ["PASS", "PASS", "PASS"],
        "device": "WA6638",
        "sw_version": "V600R024C00",
        "assertion": "association state differs from expected",
        "syslog_excerpt": "association state mismatch",
        "env": {"AP_reachable": False} if direct_rule else {"AP_reachable": True},
        "retry": "passed",
        "provenance": {
            "dataset_version": "synthetic-v0",
            "synthetic": True,
            "source": "local CLI fixture",
        },
    }


def _write_failure(tmp_path: Path, *, direct_rule: bool = False) -> Path:
    path = tmp_path / "failure.json"
    path.write_text(json.dumps(_failure(direct_rule=direct_rule)), encoding="utf-8")
    return path


def _run_queued(
    tmp_path: Path,
    *,
    direct_rule: bool = False,
    input_payload: dict | None = None,
    output_name: str = "result.json",
):
    failure = tmp_path / f"failure-{len(list(tmp_path.glob('failure-*.json')))}.json"
    failure.write_text(json.dumps(input_payload or _failure(direct_rule=direct_rule)), encoding="utf-8")
    output = tmp_path / output_name
    db = tmp_path / "local.sqlite3"
    args = [
        "run",
        str(failure),
        "--arm",
        "cascade",
        "--policy",
        str(_POLICY),
        "--out",
        str(output),
        "--local-store",
        str(db),
        "--idempotency-key",
        "handoff-key-1",
    ]
    return cli_main(args), output, db


def test_cascade_cli_queues_local_handoff_and_exports_fact_projection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    rc, output, db = _run_queued(tmp_path)

    assert rc == 0
    envelope = json.loads(output.read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "result.run-report-v2.json").read_text(encoding="utf-8"))
    handoff_json_path = tmp_path / "result.handoff.json"
    handoff_markdown_path = tmp_path / "result.handoff.md"
    card = json.loads(handoff_json_path.read_text(encoding="utf-8"))
    markdown = handoff_markdown_path.read_text(encoding="utf-8")

    assert envelope["delivery_status"] == "local_queued"
    assert envelope["local_handoff_id"] == card["handoff"]["handoff_id"]
    assert card["handoff"]["delivery_status"] == "local_queued"
    assert "not externally delivered" in markdown
    assert card["common_facts"]["assertion"] == _failure()["assertion"]
    assert "case_id" not in card["common_facts"]
    assert "provenance" not in card["common_facts"]
    assert report["accuracy"] is None
    assert report["accuracy_null_reason"]
    assert "delivery_status" in capsys.readouterr().out

    with LocalStore(db) as store:
        assert store.get_envelope(envelope["request_id"]) == store.list_queued_handoffs()[0][0]
        assert len(store.list_queued_handoffs()) == 1


def test_same_cli_idempotency_key_reuses_existing_handoff(tmp_path: Path):
    first_rc, first_output, db = _run_queued(tmp_path)
    second_rc, second_output, second_db = _run_queued(tmp_path)

    assert first_rc == second_rc == 0
    assert second_db == db
    first = json.loads(first_output.read_text(encoding="utf-8"))
    second = json.loads(second_output.read_text(encoding="utf-8"))
    assert second["request_id"] == first["request_id"]
    assert second["local_handoff_id"] == first["local_handoff_id"]
    with LocalStore(db) as store:
        assert len(store.list_queued_handoffs()) == 1


def test_cli_idempotency_key_with_changed_input_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _run_queued(tmp_path)
    changed = _failure()
    changed["assertion"] = "different failure"
    rc, output, _ = _run_queued(
        tmp_path, input_payload=changed, output_name="conflicting-result.json"
    )

    assert rc == 2
    assert "E_LOCAL_STORE_CONFLICT" in capsys.readouterr().err
    assert not output.exists()
    with LocalStore(tmp_path / "local.sqlite3") as store:
        assert len(store.list_queued_handoffs()) == 1


def test_direct_rule_result_persists_without_creating_handoff(tmp_path: Path):
    rc, output, db = _run_queued(tmp_path, direct_rule=True)

    assert rc == 0
    envelope = json.loads(output.read_text(encoding="utf-8"))
    assert envelope["delivery_status"] == "not_requested"
    assert envelope["local_handoff_id"] is None
    assert not (tmp_path / "result.handoff.json").exists()
    with LocalStore(db) as store:
        assert store.get_envelope(envelope["request_id"]).result.category == "ENVIRONMENT"
        assert store.list_queued_handoffs() == []


def test_review_cli_rejects_non_interactive_actor_assertion_without_opening_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    db = tmp_path / "should-not-be-created.sqlite3"
    monkeypatch.setattr("sys.stdin", io.StringIO("fake owner\naccept request-1\n"))

    rc = cli_main(
        ["review", "request-1", "--store", str(db), "--decision", "accept"]
    )

    assert rc == 2
    assert "E_REVIEW_INTERACTIVE_REQUIRED" in capsys.readouterr().err
    assert not db.exists()


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_review_cli_requires_explicit_typed_confirmation_and_records_local_event(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    rc, output, db = _run_queued(tmp_path)
    assert rc == 0
    envelope = json.loads(output.read_text(encoding="utf-8"))
    monkeypatch.setattr("sys.stdin", _FakeTTY("Operator A\naccept " + envelope["request_id"][:8] + "\n"))

    reviewed = cli_main(
        [
            "review",
            envelope["request_id"],
            "--store",
            str(db),
            "--result-version",
            envelope["result_version"],
            "--decision",
            "accept",
        ]
    )

    out = capsys.readouterr().out
    assert reviewed == 0
    assert "does not authenticate identity" in out
    with LocalStore(db) as store:
        current = store.get_envelope(envelope["request_id"])
        events = store.list_reviews(envelope["request_id"])
    assert current.review_status == "accepted"
    assert len(events) == 1
    assert events[0].actor_type == "local_user"
    assert events[0].actor_origin == "Operator A"


def test_review_cli_correction_creates_a_new_result_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    rc, output, db = _run_queued(tmp_path)
    assert rc == 0
    envelope = json.loads(output.read_text(encoding="utf-8"))
    corrected_result = {
        "category": "PRODUCT_DEFECT",
        "action": "ASSIGN_PRODUCT_DEVELOPER",
        "priority": 4,
        "confidence": 0.91,
        "reason": "Confirmed using the submitted assertion and syslog excerpt.",
    }
    after_path = tmp_path / "corrected-result.json"
    after_path.write_text(json.dumps(corrected_result), encoding="utf-8")
    monkeypatch.setattr(
        "sys.stdin",
        _FakeTTY("Operator B\ncorrect " + envelope["request_id"][:8] + "\n"),
    )

    reviewed = cli_main(
        [
            "review",
            envelope["request_id"],
            "--store",
            str(db),
            "--result-version",
            envelope["result_version"],
            "--decision",
            "correct",
            "--reason",
            "The provided evidence supports PRODUCT_DEFECT.",
            "--after-result",
            str(after_path),
        ]
    )

    assert reviewed == 0
    assert "correct" in capsys.readouterr().out
    with LocalStore(db) as store:
        parent = store.get_envelope(envelope["request_id"], envelope["result_version"])
        current = store.get_envelope(envelope["request_id"])
        events = store.list_reviews(envelope["request_id"])
    assert parent.review_status == "corrected"
    assert current.result_version != parent.result_version
    assert current.result == TriageResult.model_validate(corrected_result)
    assert events[0].result_version == parent.result_version
