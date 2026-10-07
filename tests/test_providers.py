"""T05 Provider 抽象 + Jev mock 单测：接口同构、确定性、契约校验兜底、三字段分离、回归保护。"""

import json
from pathlib import Path
from typing import Any

import pytest

from jevtest.models import FailureState, TriageResult
from jevtest.providers import (
    InvalidProviderOutput,
    JevMockProvider,
    Provider,
    RuleProvider,
    sanitize,
)
from jevtest.rule_engine import triage_by_rules

_DATA_V0 = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "v0.jsonl"


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


# ---- GWT: Given 任一 provider 实现，Then 通过同一接口类型检查（输出同构，PRD §5.1 条款 2）----


def test_providers_share_the_same_interface():
    mock = JevMockProvider()
    rule = RuleProvider()
    assert isinstance(mock, Provider)
    assert isinstance(rule, Provider)
    assert mock.arm_name == "jev" and mock.is_mock is True
    assert rule.arm_name == "rule" and rule.is_mock is False
    for provider in (mock, rule):
        output = provider.predict_validated(_state())
        assert isinstance(output, TriageResult)


# ---- GWT: Given mock 同输入跑两次，Then 逐字节相同（确定性：禁止 random/时间）----


def test_mock_is_deterministic():
    state = _state(env={"AP_reachable": False}, assertion="association timeout")
    provider = JevMockProvider(simulated_latency_ms=12.5, simulated_cost=0.01)
    first = provider.predict_validated(state)
    second = provider.predict_validated(state)
    # 逐字段相同（pydantic 模型 __eq__ 即逐字段比较）
    assert first == second
    # 序列化后逐字节相同
    assert first.model_dump_json() == second.model_dump_json()
    # 模拟参数只做纯记录，不改变输出
    assert provider.calls_made == 2
    assert provider.accumulated_latency_ms == 25.0
    assert provider.accumulated_cost == pytest.approx(0.02)


# ---- GWT: Given mock 被配置为返回违规结果，Then 输出被契约校验拒绝并走兜底（输出结构稳定性）----


def test_fail_mode_missing_reason_rejected_then_fallback():
    provider = JevMockProvider(fail_mode="missing_reason")
    state = _state(env={"AP_reachable": False})
    with pytest.raises(InvalidProviderOutput):
        provider.predict_validated(state)
    # 兜底：违规输出经 sanitize 后结构稳定为 UNKNOWN + HUMAN_TRIAGE
    raw = provider.predict(state)
    fallback = sanitize(raw)
    assert fallback.category == "UNKNOWN"
    assert fallback.action == "HUMAN_TRIAGE"
    assert fallback.priority == 3
    assert fallback.reason == "invalid provider output, fell back"


def test_fail_mode_bad_enum_rejected_then_fallback():
    provider = JevMockProvider(fail_mode="bad_enum")
    with pytest.raises(InvalidProviderOutput):
        provider.predict_validated(_state())
    fallback = sanitize(provider.predict(_state()))
    assert fallback.category == "UNKNOWN"
    assert fallback.action == "HUMAN_TRIAGE"


def test_sanitize_treats_whitespace_only_reason_as_invalid():
    ok = TriageResult(
        category="FLAKY", action="RERUN", priority=1, confidence=0.9, reason="  "
    )
    fallback = sanitize(ok)
    assert fallback.category == "UNKNOWN"
    assert fallback.reason == "invalid provider output, fell back"


def test_sanitize_passes_through_valid_result():
    result = triage_by_rules(_state())
    assert sanitize(result) == result


# ---- (c) RuleProvider 过统一接口后与直接调 triage_by_rules 一致（WA6638 回归保护）----


def test_rule_provider_matches_direct_engine_call_on_wa6638():
    # GWT 场景（T03）：WA6638 + previous_runs 翻转 >=2 + retry=not_executed → R1 FLAKY
    state = _state(previous_runs=["PASS", "FAIL", "PASS", "FAIL"], retry="not_executed")
    via_provider = RuleProvider().predict_validated(state)
    direct = triage_by_rules(state)
    assert via_provider == direct
    assert via_provider.category == "FLAKY"
    assert via_provider.action == "RERUN"
    assert via_provider.priority in {1, 2}

    # 第二个 WA6638 场景：env 探针 false → 环境规则命中
    env_state = _state(env={"AP_reachable": False, "AC_reachable": True})
    assert RuleProvider().predict_validated(env_state) == triage_by_rules(env_state)


# ---- (d) mock 的三字段分离：confidence / confidence_sharpness / probs 各自承载（PRD §10.2）----


@pytest.mark.parametrize(
    "state_kwargs",
    [
        {"env": {"AP_reachable": False}},                     # ENVIRONMENT 命中
        {"previous_runs": ["PASS", "FAIL", "PASS", "FAIL"],
         "retry": "not_executed"},                            # FLAKY 命中
        {"assertion": "association timeout"},                 # PRODUCT_DEFECT 命中
        {},                                                   # UNKNOWN
    ],
    ids=["environment", "flaky", "product_defect", "unknown"],
)
def test_mock_three_field_separation(state_kwargs):
    result = JevMockProvider().predict_validated(_state(**state_kwargs))
    assert result.confidence is not None
    assert result.confidence_sharpness is not None
    assert isinstance(result.probs, dict) and result.probs
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.confidence_sharpness <= 1.0
    if result.category == "UNKNOWN":
        # 任务单 mock 数值规格：UNKNOWN 时 confidence=0.3 且 confidence_sharpness=0.3，
        # 与验收项 (d)「值不同」在 UNKNOWN 情形存在内在冲突（已上报完工报告契约问题）；
        # 此处按更具体的数值规格断言，三字段分离的"值不同"性质在三臂命中情形验证。
        assert result.confidence == 0.3
        assert result.confidence_sharpness == 0.3
    else:
        # confidence=正确概率(0.9)，confidence_sharpness=分布尖锐度(0.8)，同时存在且值不同
        assert result.confidence == 0.9
        assert result.confidence_sharpness == 0.8
        assert result.confidence != result.confidence_sharpness


def test_mock_heuristic_dispatch():
    mock = JevMockProvider()
    assert mock.predict(_state(env={"AC_reachable": False})).category == "ENVIRONMENT"
    assert (
        mock.predict(_state(previous_runs=["PASS", "FAIL", "PASS"], retry="not_executed")).category
        == "FLAKY"
    )
    assert mock.predict(_state(assertion="association timeout")).category == "PRODUCT_DEFECT"
    unknown = mock.predict(_state(assertion="unexpected CLI error"))
    assert unknown.category == "UNKNOWN"
    assert "jev-mock heuristic" in unknown.reason


# ---- (e) mock 对 200 行合成数据集全量跑一遍，逐行 predict_validated 无异常 ----


def test_mock_over_full_synthetic_dataset():
    provider = JevMockProvider()
    counts: dict[str, int] = {}
    total = 0
    with _DATA_V0.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            state = FailureState.model_validate(row["failure_state"])
            result = provider.predict_validated(state)
            assert isinstance(result, TriageResult)
            counts[result.category] = counts.get(result.category, 0) + 1
            total += 1
    assert total == 200
