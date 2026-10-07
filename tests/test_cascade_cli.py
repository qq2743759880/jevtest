from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevtest.cli import main as cli_main

_REPO_ROOT = Path(__file__).resolve().parents[1]
_POLICY = _REPO_ROOT / "config" / "cascade-demo-v1.yaml"


def _failure(*, rule_r2: bool = False) -> dict:
    return {
        "case_id": "CLI-CASCADE-R4" if not rule_r2 else "CLI-CASCADE-R2",
        "result": "FAIL",
        "previous_runs": ["PASS", "PASS", "PASS"],
        "device": "WA6638",
        "sw_version": "V600R024C00",
        "assertion": "association state differs from expected",
        "syslog_excerpt": "association state mismatch",
        "env": {"AP_reachable": False} if rule_r2 else {"AP_reachable": True},
        "retry": "passed",
        "provenance": {
            "dataset_version": "synthetic-v0",
            "synthetic": True,
            "source": "offline Cascade CLI test",
        },
    }


def _write_failure(tmp_path: Path, *, rule_r2: bool = False) -> Path:
    path = tmp_path / "failure.json"
    path.write_text(json.dumps(_failure(rule_r2=rule_r2)), encoding="utf-8")
    return path


def test_cascade_requires_explicit_policy_before_providers_or_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    failure = _write_failure(tmp_path)
    output = tmp_path / "envelope.json"

    rc = cli_main(["run", str(failure), "--arm", "cascade", "--out", str(output)])

    assert rc == 2
    assert "E_POLICY_REQUIRED" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("replacement", "error_code"),
    [
        ("execution_mode: live", "E_POLICY_UNAUTHORIZED"),
        ("run_purpose: formal_eval", "E_LABELS_REQUIRED"),
    ],
)
def test_cascade_rejects_live_or_evaluation_claims_before_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    replacement: str,
    error_code: str,
):
    policy = tmp_path / "policy.yaml"
    text = _POLICY.read_text(encoding="utf-8")
    source_key = "execution_mode: offline" if replacement.startswith("execution_mode:") else "run_purpose: demo"
    policy.write_text(text.replace(source_key, replacement), encoding="utf-8")
    failure = _write_failure(tmp_path)
    output = tmp_path / "envelope.json"

    rc = cli_main(
        ["run", str(failure), "--arm", "cascade", "--policy", str(policy), "--out", str(output)]
    )

    assert rc == 2
    assert error_code in capsys.readouterr().err
    assert not output.exists()


def test_cascade_bad_yaml_returns_policy_error_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    policy = tmp_path / "bad-policy.yaml"
    policy.write_text("schema_version: [unterminated", encoding="utf-8")
    failure = _write_failure(tmp_path)

    rc = cli_main(
        ["run", str(failure), "--arm", "cascade", "--policy", str(policy), "--out", str(tmp_path / "out.json")]
    )

    err = capsys.readouterr().err
    assert rc == 2
    assert "E_POLICY_INVALID" in err
    assert "Traceback" not in err


def test_cascade_cli_writes_r4_escalation_envelope_and_unlabeled_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    failure = _write_failure(tmp_path)
    output = tmp_path / "r4-envelope.json"

    rc = cli_main(
        ["run", str(failure), "--arm", "cascade", "--policy", str(_POLICY), "--out", str(output)]
    )

    assert rc == 0
    envelope = json.loads(output.read_text(encoding="utf-8"))
    report_path = tmp_path / "r4-envelope.run-report-v2.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [step["stage"] for step in envelope["route_steps"]] == ["rule", "jev", "llm", "human"]
    assert envelope["result"]["category"] == "UNKNOWN"
    assert envelope["requires_human_review"] is True
    assert [item["stage"] for item in envelope["attempts"]] == ["rule", "jev", "llm"]
    assert [item["provider_mock"] for item in envelope["attempts"]] == [False, True, True]
    assert report["accuracy"] is None
    assert report["accuracy_null_reason"]
    assert report["eligibility"]["status"] == "not_requested"
    assert "route" in capsys.readouterr().out


def test_cascade_cli_rule_direct_output_skips_model_stages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    failure = _write_failure(tmp_path, rule_r2=True)
    output = tmp_path / "r2-envelope.json"

    rc = cli_main(
        ["run", str(failure), "--arm", "cascade", "--policy", str(_POLICY), "--out", str(output)]
    )

    assert rc == 0
    envelope = json.loads(output.read_text(encoding="utf-8"))
    assert [step["stage"] for step in envelope["route_steps"]] == ["rule"]
    assert [item["stage"] for item in envelope["attempts"]] == ["rule"]
    assert envelope["result"]["category"] == "ENVIRONMENT"
    assert "result_ready" in capsys.readouterr().out
