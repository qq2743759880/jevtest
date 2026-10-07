"""T03 Rule baseline v0 引擎单测：每规则正反例 + GWT 场景 + 顺序语义 + 输入防御。"""

from typing import Any

import pytest
from pydantic import ValidationError

from jevtest.models import FailureState, TriageResult
from jevtest.rule_engine import _load_error_code_table, triage_by_rules


def _state(**overrides: Any) -> FailureState:
    base: dict[str, Any] = {
        "case_id": "case_001",
        "result": "FAIL",
        "previous_runs": ["FAIL"],
        "device": "WA6638",
        "sw_version": "V200R022C01",
        "assertion": "unexpected CLI error",
        "syslog_excerpt": "%TEST-3-FAIL: sample excerpt",
        "env": {"AP_reachable": True, "AC_reachable": True},
        "retry": "passed",
        "provenance": {"dataset_version": "synthetic-v0", "synthetic": True},
    }
    base.update(overrides)
    return FailureState(**base)


# ---- R1_FLAKY ----


def test_r1_positive_task03_gwt_scenario():
    # GWT: previous_runs=[PASS,FAIL,PASS,FAIL] + retry=not_executed → R1 FLAKY/RERUN/priority∈{1,2}
    result = triage_by_rules(
        _state(previous_runs=["PASS", "FAIL", "PASS", "FAIL"], retry="not_executed")
    )
    assert isinstance(result, TriageResult)
    assert result.category == "FLAKY"
    assert result.action == "RERUN"
    assert result.priority in {1, 2}
    assert result.priority == 1  # 引擎取 YAML 区间下限


def test_r1_reason_contains_rule_id_and_flip_count():
    # GWT: reason 含规则 ID + 触发字段值，如 `R1: flip_count=3, retry=not_executed`
    result = triage_by_rules(
        _state(previous_runs=["PASS", "FAIL", "PASS", "FAIL"], retry="not_executed")
    )
    assert "R1:" in result.reason
    assert "flip_count=3" in result.reason
    assert "retry=not_executed" in result.reason


def test_r1_negative_retry_executed_falls_through():
    # 反例：翻转足够但 retry 已执行 → R1 不命中（assertion/env 均不触发 → R4）
    result = triage_by_rules(
        _state(previous_runs=["PASS", "FAIL", "PASS", "FAIL"], retry="failed")
    )
    assert result.category == "UNKNOWN"


def test_r1_negative_flip_below_threshold():
    # 反例：retry=not_executed 但翻转仅 1 次 → R1 不命中
    result = triage_by_rules(_state(previous_runs=["PASS", "FAIL"], retry="not_executed"))
    assert result.category == "UNKNOWN"
    assert "R1" not in result.reason


# ---- R2_ENVIRONMENT ----


def test_r2_positive_env_probe_false():
    # GWT: env 任一 reachable=false → R2 ENVIRONMENT/CHECK_ENVIRONMENT/3
    result = triage_by_rules(_state(env={"AP_reachable": False, "AC_reachable": True}))
    assert result.category == "ENVIRONMENT"
    assert result.action == "CHECK_ENVIRONMENT"
    assert result.priority == 3
    assert "R2:" in result.reason
    assert "AP_reachable" in result.reason  # reason 含失败键名


def test_r2_negative_all_probes_true():
    # 反例：全部探针为 true → R2 不命中
    result = triage_by_rules(_state())
    assert "R2:" not in result.reason


def test_r2_reason_reports_first_failing_key():
    # 多个探针为 false 时报告遍历序首个失败键，且引擎不做任何重排
    result = triage_by_rules(
        _state(env={"AC_reachable": False, "AP_reachable": False})
    )
    assert "env[AC_reachable]=false" in result.reason


# ---- R3_ERROR_CODE ----


def test_r3_positive_matches_table_entry():
    result = triage_by_rules(_state(assertion="sta association timeout on wlan 1"))
    entry = next(
        e for e in _load_error_code_table() if e["match_keyword"] == "association timeout"
    )
    assert result.category == entry["category"]
    assert result.action == entry["action"]
    assert result.priority == entry["priority"]
    assert "R3:" in result.reason
    assert f"error_code={entry['code']} matched" in result.reason


def test_r3_match_is_case_insensitive():
    result = triage_by_rules(_state(assertion="STA ASSOCIATION TIMEOUT ON WLAN 1"))
    assert "R3:" in result.reason
    assert result.category == "ENVIRONMENT"


def test_r3_negative_no_keyword_match_falls_to_r4():
    # 反例：assertion 不含任何表关键词 → R3 不命中
    result = triage_by_rules(_state(assertion="roaming handshake stuck"))
    assert "R3:" not in result.reason


# ---- R4_FALLBACK ----


def test_r4_fallback_unknown_human_triage():
    # GWT: 无规则命中 → R4 UNKNOWN/HUMAN_TRIAGE/3，reason 含规则 ID 与 assertion
    result = triage_by_rules(_state(assertion="mystery failure xy"))
    assert result.category == "UNKNOWN"
    assert result.action == "HUMAN_TRIAGE"
    assert result.priority == 3
    assert "R4:" in result.reason
    assert "mystery failure xy" in result.reason


def test_r4_long_assertion_keeps_reason_within_limit():
    result = triage_by_rules(_state(assertion="x" * 500))
    assert result.category == "UNKNOWN"
    assert len(result.reason) <= 200  # TriageResult.reason 上限


# ---- 版本化 ----


def test_rule_version_appended_to_reason():
    # GWT: 输出附带 rule_version="0"（TriageResult 无此字段，按约定放 reason 尾部）
    for state in (
        _state(previous_runs=["PASS", "FAIL", "PASS", "FAIL"], retry="not_executed"),
        _state(env={"AP_reachable": False}),
        _state(assertion="dhcp offer missing"),
        _state(),
    ):
        result = triage_by_rules(state)
        assert result.reason.endswith("| rule_version=0")


# ---- 顺序语义与输入防御 ----


def test_rule_order_r1_wins_over_r2_when_both_match():
    # GWT: 同时满足 R1 与 R2 条件 → YAML 声明序在先的 R1 输出
    result = triage_by_rules(
        _state(
            previous_runs=["PASS", "FAIL", "PASS", "FAIL"],
            retry="not_executed",
            env={"AP_reachable": False},
        )
    )
    assert result.category == "FLAKY"
    assert result.reason.startswith("R1:")


def test_rule_order_r2_wins_over_r3_when_both_match():
    result = triage_by_rules(
        _state(assertion="association timeout", env={"AP_reachable": False})
    )
    assert result.category == "ENVIRONMENT"
    assert result.reason.startswith("R2:")


def test_missing_field_dict_raises_validation_error_not_silent_output():
    # (c) 缺字段 dict 必须被 pydantic ValidationError 拦截，绝不静默产出 TriageResult
    incomplete: dict[str, Any] = {
        "case_id": "case_bad",
        "result": "FAIL",
        # 缺 previous_runs / env / retry / provenance 等
    }
    with pytest.raises(ValidationError):
        triage_by_rules(incomplete)


def test_valid_dict_input_is_accepted():
    state: dict[str, Any] = {
        "case_id": "case_dict",
        "result": "FAIL",
        "previous_runs": ["PASS", "FAIL", "PASS", "FAIL"],
        "device": "WA6638",
        "sw_version": "V200R022C01",
        "assertion": "unexpected CLI error",
        "syslog_excerpt": "",
        "env": {"AP_reachable": True},
        "retry": "not_executed",
        "provenance": {"dataset_version": "synthetic-v0", "synthetic": True},
    }
    result = triage_by_rules(state)
    assert result.category == "FLAKY"


# ---- 错误码表契约自检（来源纪律：8-12 条、来源可验证）----


def test_error_code_table_shape_and_sources():
    entries = _load_error_code_table()
    assert 8 <= len(entries) <= 12
    for entry in entries:
        assert entry["priority"] in range(1, 6)
        TriageResult(
            category=entry["category"],
            action=entry["action"],
            priority=entry["priority"],
            confidence=1.0,
            reason="table entry validation",
        )
        assert entry["source_url"].startswith("https://")
        assert str(entry["match_keyword"]) == entry["match_keyword"].lower()
