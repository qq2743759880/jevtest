import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jevtest.contracts import ContractDriftError, export_schemas, verify_contracts
from jevtest.execution_contracts import VERSIONED_CONTRACTS
from jevtest.models import BenchmarkExperiment, FailureState, TriageResult

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"


def test_export_covers_frozen_and_versioned_contracts():
    schemas = export_schemas()
    expected = {"FailureState", "TriageResult", "BenchmarkExperiment"}
    expected.update(name for name, _, _ in VERSIONED_CONTRACTS)
    assert set(schemas) == expected


def test_contracts_zero_drift():
    # 冻结契约与版本化契约和模型导出零漂移：不抛 ContractDriftError 即通过
    verify_contracts(CONTRACTS_DIR)


def test_drift_is_detected_on_tampered_contract(tmp_path: Path):
    # 反向验证比较器有效：篡改枚举后必须抓到漂移
    contract = json.loads((CONTRACTS_DIR / "triage-result.schema.json").read_text(encoding="utf-8"))
    contract["properties"]["category"]["enum"] = contract["properties"]["category"]["enum"][:-1]
    (tmp_path / "triage-result.schema.json").write_text(
        json.dumps(contract, ensure_ascii=False), encoding="utf-8"
    )
    for name in ("failure-state.schema.json", "benchmark.schema.json"):
        (tmp_path / name).write_text(
            (CONTRACTS_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    for _, _, name in VERSIONED_CONTRACTS:
        (tmp_path / name).write_text(
            (CONTRACTS_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    with pytest.raises(ContractDriftError) as excinfo:
        verify_contracts(tmp_path)
    assert "category" in str(excinfo.value)


def test_invalid_category_enum_raises_with_field_name():
    # (b) 非法枚举构造 TriageResult 抛 ValidationError 且错误信息含字段名
    with pytest.raises(ValidationError) as excinfo:
        TriageResult(
            category="NOT_A_CATEGORY",
            action="RERUN",
            priority=3,
            confidence=0.9,
            reason="rule R-001 triggered",
        )
    assert "category" in str(excinfo.value)


def test_invalid_action_and_priority_rejected():
    with pytest.raises(ValidationError) as action_error:
        TriageResult(
            category="FLAKY", action="REBOOT", priority=3, confidence=0.5, reason="x"
        )
    assert "action" in str(action_error.value)
    with pytest.raises(ValidationError) as priority_error:
        TriageResult(
            category="FLAKY", action="RERUN", priority=6, confidence=0.5, reason="x"
        )
    assert "priority" in str(priority_error.value)


def test_wa6638_roaming_failure_constructs():
    # (c) 合法 WA6638 漫游失败样例可构造 FailureState
    failure = FailureState(
        case_id="wireless_roaming_case_017",
        result="FAIL",
        previous_runs=["PASS", "PASS", "FAIL", "FAIL"],
        device="WA6638",
        sw_version="V200R022C01",
        assertion="association timeout",
        syslog_excerpt="%DOT11-4-ASSOC_TIMEOUT: STA aa:bb:cc:dd:ee:ff association timeout on WLAN 1",
        env={"AP_reachable": True, "AC_reachable": True},
        retry="not_executed",
        provenance={"dataset_version": "synthetic-v0", "synthetic": True},
    )
    assert failure.result == "FAIL"
    assert failure.provenance.synthetic is True
    assert failure.env["AP_reachable"] is True


def test_failure_state_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        FailureState(
            case_id="c1",
            result="FAIL",
            previous_runs=["FAIL"],
            device="WA6638",
            sw_version="V1",
            assertion="timeout",
            syslog_excerpt="",
            env={},
            retry="failed",
            provenance={"dataset_version": "synthetic-v0", "synthetic": False},
            extra_field="not allowed",
        )
    assert "extra_field" in str(excinfo.value)


def test_benchmark_experiment_rejects_unknown_metrics_key():
    with pytest.raises(ValidationError) as excinfo:
        BenchmarkExperiment(
            experiment_id="exp-001",
            dataset_version="synthetic-v0",
            dataset_kind="synthetic",
            rule_version="0",
            arms=["rule"],
            metrics={
                "rule": {
                    "accuracy": 1.0,
                    "cost": 0.0,
                    "latency_p50_ms": 1.0,
                    "latency_p95_ms": 2.0,
                    "routing_ratio": {"rule": 1.0},
                },
                "unknown_arm": {
                    "accuracy": 1.0,
                    "cost": 0.0,
                    "latency_p50_ms": 1.0,
                    "latency_p95_ms": 2.0,
                    "routing_ratio": {"rule": 1.0},
                },
            },
            declaration="SYNTHETIC — 不代表企业效果",
        )
    assert "metrics" in str(excinfo.value)
