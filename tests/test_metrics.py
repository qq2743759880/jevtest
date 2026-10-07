"""T08 计量层单测：SampleRecord 契约、mock 记账、E_PARSE_FAILURE 批次不中断、聚合 p50/p95、UTC 时间戳。"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from jevtest.metrics import ErrorLog, MeasureRunner, SampleRecord, aggregate
from jevtest.models import FailureState, TriageResult
from jevtest.providers import JevMockProvider, RuleProvider

_ERROR_KEYS = {"arm", "case_id", "raw_output", "timestamp"}


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


def _rec(arm: str, latency_ms: float, **overrides: Any) -> SampleRecord:
    values: dict[str, Any] = {
        "case_id": f"case_{arm}_{latency_ms}",
        "arm": arm,
        "category": "UNKNOWN",
        "action": "HUMAN_TRIAGE",
        "priority": 3,
        "confidence": 0.0,
        "latency_ms": latency_ms,
        "cost": 0.0,
        "is_mock": False,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    values.update(overrides)
    return SampleRecord(**values)


# ---- (a) run_one 返回的 SampleRecord 过自己模型校验且字段齐 ----


def test_sample_record_self_validates_and_has_full_fields():
    runner = MeasureRunner(JevMockProvider())
    result, record = runner.run_one(_state())
    assert isinstance(result, TriageResult)
    # 过自己的模型校验（含 JSON 序列化往返）
    assert SampleRecord.model_validate(record.model_dump()) == record
    assert SampleRecord.model_validate_json(record.to_json_line()) == record
    # 字段齐：与 benchmark.schema per_sample 描述一致的字段集
    assert set(record.model_dump()) == {
        "case_id",
        "arm",
        "category",
        "action",
        "priority",
        "confidence",
        "latency_ms",
        "cost",
        "is_mock",
        "parse_failure",
        "timestamp",
    }
    assert record.arm == "jev"
    assert record.case_id == "case_001"
    assert record.is_mock is True
    assert record.parse_failure is False
    assert record.category == result.category
    assert record.action == result.action
    # sort_keys 序列化：键序固定
    line = record.to_json_line()
    assert list(json.loads(line)) == sorted(json.loads(line))


# ---- (b) mock provider 的 simulated_cost 累进记账正确、is_mock=True ----


def test_mock_cost_and_latency_bookkeeping_accumulates():
    provider = JevMockProvider(simulated_latency_ms=2.5, simulated_cost=0.01)
    runner = MeasureRunner(provider)
    records = [runner.run_one(_state(case_id=f"c{i}"))[1] for i in range(3)]
    assert provider.is_mock is True
    assert all(r.is_mock for r in records)
    # 逐样本：每次调用的记账差 = 单次模拟值
    assert all(r.cost == 0.01 for r in records)
    assert all(r.latency_ms == 2.5 for r in records)
    # provider 侧累计：3 次调用累进
    assert provider.calls_made == 3
    assert provider.accumulated_latency_ms == 7.5
    assert provider.accumulated_cost == 0.03
    # 非 mock 臂对照：RuleProvider 无记账 → cost 恒 0.0，latency 为真实耗时（>=0）
    rule_runner = MeasureRunner(RuleProvider())
    _, rule_record = rule_runner.run_one(_state())
    assert rule_record.cost == 0.0
    assert rule_record.is_mock is False
    assert rule_record.latency_ms >= 0.0


# ---- (c) fail_mode 触发 parse_failure 时批次不中断：3 条含 1 条坏输出 ----


def test_parse_failure_does_not_break_batch_and_writes_error_log(tmp_path: Path):
    error_log = ErrorLog(tmp_path / "error_samples.jsonl")
    provider = JevMockProvider(simulated_latency_ms=1.0, simulated_cost=0.01, fail_mode="bad_enum")
    runner = MeasureRunner(provider, error_log)

    # 第 1 条：坏输出 → 兜底，不抛异常、不中断
    result1, record1 = runner.run_one(_state(case_id="bad_case"))
    first_attempts = list(runner.last_attempts)
    assert record1.parse_failure is True
    assert result1.category == "UNKNOWN"
    assert result1.action == "HUMAN_TRIAGE"
    assert result1.reason == "invalid provider output, fell back"
    assert record1.category == "UNKNOWN"
    assert len(first_attempts) == 1
    assert first_attempts[0].status == "failed"

    # 第 2、3 条：恢复正常（mock 计量管道联调，直接关掉注入的 fail_mode）
    provider.fail_mode = None
    _, record2 = runner.run_one(_state(case_id="good_case_2"))
    _, record3 = runner.run_one(_state(case_id="good_case_3"))
    assert record2.parse_failure is False
    assert record3.parse_failure is False

    # 批次完整：3 条记录，恰 1 条 parse_failure
    records = [record1, record2, record3]
    assert len(records) == 3
    assert sum(1 for r in records if r.parse_failure) == 1
    # 坏输出那次调用只计一次记账（重放 predict 取原始输出的开销不计入）
    assert record1.cost == 0.01 and record1.latency_ms == 1.0
    assert record2.cost == 0.01 and record3.cost == 0.01

    # error_samples.jsonl 同步落盘 1 行，行字段齐
    lines = (tmp_path / "error_samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert set(row) == _ERROR_KEYS
    assert row["case_id"] == "bad_case"
    assert row["arm"] == "jev"
    # raw_output 保留的是原始违规输出（可追溯、可重放），不是兜底结果
    assert row["raw_output"]["category"] == "NOT_A_CATEGORY"
    # 坏输出的记账差与重放调用隔离：累计仍是 3 次有效调用（1 次失败 + 1 次重放 + 2 次正常）
    assert provider.calls_made == 3
    # 行内 timestamp 合法且与对应 SampleRecord 一致
    assert record1.timestamp == row["timestamp"]


# ---- (d) aggregate 的 p50/p95 对已知输入手算断言（nearest-rank：ceil(q*n) 位）----


def test_aggregate_percentiles_hand_computed():
    records = [
        # arm=jev：10 条，latency 10..100 → p50=第5位=50，p95=第10位(ceil(9.5))=100
        _rec("jev", float(10 * (i + 1))) for i in range(10)
    ]
    records.append(_rec("jev", 0.0, case_id="bad", parse_failure=True))
    # arm=rule：3 条 [100, 200, 300] → p50=第2位=200，p95=第3位(ceil(2.85))=300
    records += [_rec("rule", 100.0, case_id="r1"), _rec("rule", 200.0, case_id="r2"), _rec("rule", 300.0, case_id="r3")]

    report = aggregate(records)
    assert set(report) == {"jev", "rule"}

    jev = report["jev"]
    assert jev["count"] == 11
    assert jev["latency_p50_ms"] == 50.0
    assert jev["latency_p95_ms"] == 100.0
    assert jev["parse_failures"] == 1
    assert jev["cost"] == 0.0

    rule = report["rule"]
    assert rule["count"] == 3
    assert rule["latency_p50_ms"] == 200.0
    assert rule["latency_p95_ms"] == 300.0
    assert rule["parse_failures"] == 0

    # cost 总和：0.01 x 5 = 0.05
    paid = [ _rec("llm", 1.0, case_id=f"p{i}", cost=0.01) for i in range(5) ]
    assert aggregate(paid)["llm"]["cost"] == 0.05


def test_aggregate_empty_input():
    assert aggregate([]) == {}


# ---- (e) timestamp 是合法 ISO8601 UTC ----


def test_timestamp_is_valid_iso8601_utc():
    _, record = MeasureRunner(JevMockProvider()).run_one(_state())
    parsed = datetime.fromisoformat(record.timestamp)
    assert parsed.utcoffset() == timedelta(0)
    # 过模型校验的时间戳（含非法值拒绝）
    assert SampleRecord.model_validate(record.model_dump()).timestamp == record.timestamp


def test_error_log_default_timestamp_is_iso8601_utc(tmp_path: Path):
    log = ErrorLog(tmp_path / "errors.jsonl")
    log.append("case_x", "rule", {"oops": True})
    row = json.loads((tmp_path / "errors.jsonl").read_text(encoding="utf-8"))
    assert datetime.fromisoformat(row["timestamp"]).utcoffset() == timedelta(0)


def test_sample_record_rejects_non_utc_timestamp():
    import pytest

    with pytest.raises(Exception):
        _rec("jev", 1.0, timestamp="2026-01-01T00:00:00+08:00")
    with pytest.raises(Exception):
        _rec("jev", 1.0, timestamp="not-a-timestamp")
