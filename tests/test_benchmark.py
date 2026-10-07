"""T09 benchmark runner 单测（非网络）：假 Provider 三臂 + cascade 缺席契约。

覆盖任务 GWT：
- (a) 输入对等断言：同一份 state 列表对象逐臂传递；臂内篡改 state → InputParityError 中止；
  正常路径四臂收到的输入逐字节相同。
- (b) fail_mode 坏输出不中断批次、error_samples.jsonl 落盘、absent_arms 正确。
- (c) experiment-report.json 过 BenchmarkExperiment 模型校验（契约单一事实源）。
- (d) declaration：synthetic → "SYNTHETIC — 不代表企业效果"；legacy v1 refuses non-synthetic new runs.
- (e) accuracy 手算核对（5 条小数据集）。
- 双跑同配置（mock 臂）报告指标一致（GWT 第 3 条）。
- CLI benchmark 子命令接线。

live 测试（marker live，真实 API，预算 jev<=12 / ark<=12）在文件尾部。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jevtest.benchmark import (
    CASCADE_ABSENT_REASON,
    COST_CURRENCY_NOTE,
    EvaluationNotEligibleError,
    InputParityError,
    UnknownArmError,
    render_report,
    run_benchmark,
)
from jevtest.cli import main as cli_main
from jevtest.models import BenchmarkExperiment, FailureState, TriageResult
from jevtest.providers.base import Provider

_DECLARATION_SYNTHETIC = "SYNTHETIC — 不代表企业效果"


# ---------------------------------------------------------------------------
# 假 Provider（可控三臂）与数据集工厂
# ---------------------------------------------------------------------------


class FakeArmProvider(Provider):
    """确定性假臂：is_mock=True（latency/cost 走记账口径，可双跑一致）。

    - category_by_case：按 case_id 指定输出类别（accuracy 手算用）；
    - fail_mode：输出非法枚举（E_PARSE_FAILURE 路径：批次不中断 + sanitize 兜底）；
    - tamper：predict 内篡改共享 state（输入对等断言的攻击者模拟）。
    """

    def __init__(
        self,
        arm_name: str,
        *,
        category_by_case: dict[str, str] | None = None,
        default_category: str = "UNKNOWN",
        fail_mode: bool = False,
        tamper: bool = False,
        simulated_latency_ms: float = 1.0,
        simulated_cost: float = 0.0,
    ) -> None:
        self.arm_name = arm_name
        self.is_mock = True
        self.category_by_case = category_by_case or {}
        self.default_category = default_category
        self.fail_mode = fail_mode
        self.tamper = tamper
        self.simulated_latency_ms = simulated_latency_ms
        self.simulated_cost = simulated_cost
        self.accumulated_latency_ms = 0.0
        self.accumulated_cost = 0.0
        self.calls_made = 0
        self.seen_inputs: list[str] = []

    def predict(self, state: FailureState) -> TriageResult:
        self.calls_made += 1
        self.accumulated_latency_ms += self.simulated_latency_ms
        self.accumulated_cost += self.simulated_cost
        self.seen_inputs.append(state.model_dump_json())
        if self.tamper:
            state.assertion = f"TAMPERED-BY-{self.arm_name}"
        category = self.category_by_case.get(state.case_id, self.default_category)
        result = TriageResult(
            category=category,  # type: ignore[arg-type]
            action="HUMAN_TRIAGE",
            priority=3,
            confidence=0.9,
            reason=f"fake arm {self.arm_name} for {state.case_id}",
        )
        if self.fail_mode:
            return result.model_copy(update={"category": "NOT_A_CATEGORY"})
        return result


def _state(case_id: str, *, synthetic: bool = True) -> FailureState:
    return FailureState(
        case_id=case_id,
        result="FAIL",
        previous_runs=["FAIL"],
        device="WA6638",
        sw_version="V200R022C01",
        assertion="unexpected CLI error",
        syslog_excerpt="%TEST-3-FAIL: sample excerpt",
        env={"AP_reachable": True},
        retry="passed",
        provenance={"dataset_version": "synthetic-v0", "synthetic": synthetic},
    )


def _write_dataset(path: Path, expected_labels: list[str], *, synthetic: bool = True) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for idx, expected in enumerate(expected_labels, start=1):
            row = {
                "expected": expected,
                "failure_state": _state(f"case_{idx:03d}", synthetic=synthetic).model_dump(),
            }
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _fake_factory(overrides: dict[str, FakeArmProvider] | None = None):
    """provider_factory：三臂假 Provider（rule/jev/llm），可按臂覆盖实例。"""
    defaults = {
        "rule": FakeArmProvider("rule"),
        "jev": FakeArmProvider("jev"),
        "llm": FakeArmProvider("llm"),
    }
    if overrides:
        defaults.update(overrides)

    def factory(arm: str) -> Provider:
        return defaults[arm]

    factory.providers = defaults  # type: ignore[attr-defined]
    return factory


# ---------------------------------------------------------------------------
# (a) 输入对等：同一份对象逐字节相同；篡改即中止
# ---------------------------------------------------------------------------


def test_parity_all_arms_receive_byte_identical_inputs(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["PRODUCT_DEFECT"] * 3)
    factory = _fake_factory()
    run_benchmark(dataset, ["rule", "jev", "llm"], out_dir=tmp_path / "out", provider_factory=factory)
    providers = factory.providers  # type: ignore[attr-defined]
    rule_inputs = providers["rule"].seen_inputs
    assert len(rule_inputs) == 3
    # 条款 1 的可测形态：四臂（此处三假臂）收到的输入集合逐字节相同
    assert providers["jev"].seen_inputs == rule_inputs
    assert providers["llm"].seen_inputs == rule_inputs


def test_parity_tampered_state_aborts_run(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["PRODUCT_DEFECT"] * 3)
    factory = _fake_factory({"llm": FakeArmProvider("llm", tamper=True)})
    with pytest.raises(InputParityError, match="parity violation"):
        run_benchmark(
            dataset, ["rule", "jev", "llm"], out_dir=tmp_path / "out", provider_factory=factory
        )
    # 中止即中止：不产出假装完整的报告
    assert not (tmp_path / "out" / "experiment-report.json").exists()


# ---------------------------------------------------------------------------
# (b) fail_mode：批次不中断 + error_samples + absent_arms
# ---------------------------------------------------------------------------


def test_fail_mode_batch_continues_error_log_and_absent_arms(tmp_path: Path):
    labels = ["PRODUCT_DEFECT", "TEST_SCRIPT", "ENVIRONMENT"]
    dataset = _write_dataset(tmp_path / "ds.jsonl", labels)
    factory = _fake_factory({"jev": FakeArmProvider("jev", fail_mode=True)})
    report_path = run_benchmark(
        dataset,
        ["rule", "jev", "llm", "cascade"],
        out_dir=tmp_path / "out",
        provider_factory=factory,
    )
    out = tmp_path / "out"

    # cascade 被请求但未实现：缺席说明，不报错、不假装
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["arms"] == ["rule", "jev", "llm"]
    assert report["absent_arms"] == [{"arm": "cascade", "reason": CASCADE_ABSENT_REASON}]

    # 坏输出不中断批次：jev 3 条样本全落盘，全部 parse_failure（fail_mode 对该臂全局生效）
    jev_lines = (out / "per_sample_jev.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jev_lines) == 3
    failures = [json.loads(line) for line in jev_lines if json.loads(line)["parse_failure"]]
    assert len(failures) == 3
    assert all(f["category"] == "UNKNOWN" and f["action"] == "HUMAN_TRIAGE" for f in failures)
    # 兜底 UNKNOWN 如实计入 accuracy（expected 全非 UNKNOWN → 0.0）
    assert report["metrics"]["jev"]["accuracy"] == 0.0

    # error_samples.jsonl 同步落盘：保留原始违规输出（可追溯、可重放）
    error_rows = [
        json.loads(line)
        for line in (out / "error_samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(error_rows) == 3
    assert all(row["arm"] == "jev" for row in error_rows)
    assert all(row["raw_output"]["category"] == "NOT_A_CATEGORY" for row in error_rows)
    assert all(set(row) == {"case_id", "arm", "raw_output", "timestamp"} for row in error_rows)


def test_absent_arms_empty_when_cascade_not_requested(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY"])
    report_path = run_benchmark(
        dataset, ["rule"], out_dir=tmp_path / "out", provider_factory=_fake_factory()
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["absent_arms"] is None
    assert report["arms"] == ["rule"]


# ---------------------------------------------------------------------------
# (c) 报告过 BenchmarkExperiment 模型校验（契约单一事实源）
# ---------------------------------------------------------------------------


def test_report_json_validates_against_benchmark_experiment(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY", "UNKNOWN"])
    report_path = run_benchmark(
        dataset, ["rule", "jev", "llm"], out_dir=tmp_path / "out", provider_factory=_fake_factory()
    )
    experiment = BenchmarkExperiment.model_validate(
        json.loads(report_path.read_text(encoding="utf-8"))
    )
    assert experiment.dataset_version == "synthetic-v0"
    assert experiment.rule_version == "0"  # contracts/rule-baseline-v0.yaml
    assert experiment.per_sample_path == "per_sample_rule.jsonl;per_sample_jev.jsonl;per_sample_llm.jsonl"
    assert experiment.error_samples_path == "error_samples.jsonl"
    for arm, metrics in experiment.metrics.items():
        assert metrics.routing_ratio == {arm: 1.0}  # 无 cascade：100% 自身，如实填写
        assert metrics.sample_count == 2
        assert metrics.cost >= 0.0
    assert experiment.metrics["llm"].cost_currency == COST_CURRENCY_NOTE
    # 每臂 per-sample JSONL 行数 = 样本数
    for arm in ("rule", "jev", "llm"):
        lines = (tmp_path / "out" / f"per_sample_{arm}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# (d) declaration：合成/真实强制分开（裁定 2）
# ---------------------------------------------------------------------------


def test_declaration_synthetic_forced_in_report_and_markdown(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY"])
    report_path = run_benchmark(
        dataset, ["rule"], out_dir=tmp_path / "out", provider_factory=_fake_factory()
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["declaration"] == _DECLARATION_SYNTHETIC
    md = render_report(report_path)
    assert _DECLARATION_SYNTHETIC in md
    assert "不代表企业真实日志效果" in md  # 合成数据集专属局限声明


def test_legacy_v1_refuses_non_synthetic_before_writing_artifacts(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY"], synthetic=False)
    out_dir = tmp_path / "out"
    factory_called = False

    def factory(arm: str) -> Provider:
        nonlocal factory_called
        factory_called = True
        return FakeArmProvider(arm)

    with pytest.raises(EvaluationNotEligibleError, match="RunReport v2"):
        run_benchmark(dataset, ["rule"], out_dir=out_dir, provider_factory=factory)

    assert factory_called is False
    assert not out_dir.exists()


def test_cli_benchmark_reports_non_synthetic_gate_without_traceback(tmp_path: Path, capsys):
    dataset = _write_dataset(tmp_path / "real.jsonl", ["FLAKY"], synthetic=False)
    out_dir = tmp_path / "out"

    exit_code = cli_main(
        ["benchmark", str(dataset), "--arms", "rule", "--out", str(out_dir)]
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "evaluation not eligible" in output
    assert "RunReport v2" in output
    assert "Traceback" not in output
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# (e) accuracy 手算核对（5 条）
# ---------------------------------------------------------------------------


def test_accuracy_hand_computed_five_samples(tmp_path: Path):
    # expected:           PD           TS           ENV          FLAKY        UNKNOWN
    labels = [
        "PRODUCT_DEFECT",
        "TEST_SCRIPT",
        "ENVIRONMENT",
        "FLAKY",
        "UNKNOWN",
    ]
    dataset = _write_dataset(tmp_path / "ds.jsonl", labels)
    factory = _fake_factory(
        {
            # rule：前 3 命中，后 2 错 → 3/5 = 0.6
            "rule": FakeArmProvider(
                "rule",
                category_by_case={
                    "case_001": "PRODUCT_DEFECT",
                    "case_002": "TEST_SCRIPT",
                    "case_003": "ENVIRONMENT",
                    "case_004": "PRODUCT_DEFECT",  # 错（expected FLAKY）
                },
                default_category="TEST_SCRIPT",  # case_005 错（expected UNKNOWN）
            ),
            # jev：前 4 命中，case_005 错 → 4/5 = 0.8
            "jev": FakeArmProvider(
                "jev",
                category_by_case={
                    "case_001": "PRODUCT_DEFECT",
                    "case_002": "TEST_SCRIPT",
                    "case_003": "ENVIRONMENT",
                    "case_004": "FLAKY",
                },
                default_category="PRODUCT_DEFECT",  # 错
            ),
            # llm：5/5 = 1.0
            "llm": FakeArmProvider(
                "llm",
                category_by_case=dict(zip((f"case_{i:03d}" for i in range(1, 6)), labels)),
            ),
        }
    )
    report_path = run_benchmark(
        dataset, ["rule", "jev", "llm"], out_dir=tmp_path / "out", provider_factory=factory
    )
    metrics = json.loads(report_path.read_text(encoding="utf-8"))["metrics"]
    assert metrics["rule"]["accuracy"] == pytest.approx(0.6)
    assert metrics["jev"]["accuracy"] == pytest.approx(0.8)
    assert metrics["llm"]["accuracy"] == pytest.approx(1.0)
    assert all(metrics[arm]["sample_count"] == 5 for arm in ("rule", "jev", "llm"))


def test_accuracy_counts_parse_failure_fallback_honestly(tmp_path: Path):
    # jev 全部坏输出 → 兜底 UNKNOWN；expected 全非 UNKNOWN → accuracy 0.0，不修饰
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY", "FLAKY"])
    factory = _fake_factory({"jev": FakeArmProvider("jev", fail_mode=True)})
    report_path = run_benchmark(
        dataset, ["jev"], out_dir=tmp_path / "out", provider_factory=factory
    )
    metrics = json.loads(report_path.read_text(encoding="utf-8"))["metrics"]["jev"]
    assert metrics["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# 双跑同配置（mock 臂）指标一致（GWT 第 3 条）
# ---------------------------------------------------------------------------


def test_double_run_same_mock_config_gives_identical_metrics(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY", "UNKNOWN", "FLAKY"])

    def one_run(out_name: str) -> dict[str, Any]:
        factory = _fake_factory(
            {
                "rule": FakeArmProvider("rule", simulated_latency_ms=2.0, simulated_cost=0.0),
                "jev": FakeArmProvider("jev", simulated_latency_ms=5.0, simulated_cost=0.01),
                "llm": FakeArmProvider("llm", simulated_latency_ms=9.0, simulated_cost=0.0),
            }
        )
        report_path = run_benchmark(
            dataset, ["rule", "jev", "llm"], out_dir=tmp_path / out_name, provider_factory=factory
        )
        return json.loads(report_path.read_text(encoding="utf-8"))["metrics"]

    first, second = one_run("run_a"), one_run("run_b")
    assert first == second  # mock 臂无 API 温度：双跑指标逐字段一致


# ---------------------------------------------------------------------------
# 臂名校验
# ---------------------------------------------------------------------------


def test_unknown_arm_name_rejected(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY"])
    with pytest.raises(UnknownArmError, match="unknown arm"):
        run_benchmark(
            dataset, ["rule", "gpt4"], out_dir=tmp_path / "out", provider_factory=_fake_factory()
        )
    with pytest.raises(UnknownArmError, match="unimplemented"):
        run_benchmark(
            dataset, ["cascade"], out_dir=tmp_path / "out", provider_factory=_fake_factory()
        )


def test_duplicate_arms_deduplicated(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY"])
    report_path = run_benchmark(
        dataset, ["rule", "rule", "llm"], out_dir=tmp_path / "out", provider_factory=_fake_factory()
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["arms"] == ["rule", "llm"]


# ---------------------------------------------------------------------------
# CLI 接线
# ---------------------------------------------------------------------------


def test_cli_benchmark_subcommand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY", "UNKNOWN"])
    factory = _fake_factory()
    monkeypatch.setattr(
        "jevtest.benchmark.default_provider_factory", lambda arm: factory(arm)
    )
    out_dir = tmp_path / "cli-out"
    exit_code = cli_main(
        ["benchmark", str(dataset), "--arms", "rule,jev", "--limit", "2", "--out", str(out_dir)]
    )
    assert exit_code == 0
    report = json.loads((out_dir / "experiment-report.json").read_text(encoding="utf-8"))
    assert report["arms"] == ["rule", "jev"]
    assert report["metrics"]["rule"]["sample_count"] == 2
    assert (out_dir / "experiment-report.md").exists()


def test_cli_benchmark_invalid_arm_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY"])
    monkeypatch.setattr(
        "jevtest.benchmark.default_provider_factory", lambda arm: FakeArmProvider(arm)
    )
    exit_code = cli_main(
        ["benchmark", str(dataset), "--arms", "rule,nope", "--out", str(tmp_path / "out")]
    )
    assert exit_code == 2
    assert "unknown arm" in capsys.readouterr().out


def test_cli_benchmark_default_factory_uses_real_rule_arm(tmp_path: Path):
    """不加 monkeypatch、只跑 rule 臂：CLI 默认工厂走真实 RuleProvider（无网络）。"""
    dataset = _write_dataset(tmp_path / "ds.jsonl", ["FLAKY"])
    out_dir = tmp_path / "out"
    exit_code = cli_main(["benchmark", str(dataset), "--arms", "rule", "--out", str(out_dir)])
    assert exit_code == 0
    report = json.loads((out_dir / "experiment-report.json").read_text(encoding="utf-8"))
    assert report["metrics"]["rule"]["sample_count"] == 1


# ---------------------------------------------------------------------------
# live：真实 API；每供应商上限为 10 个样本 × 2 次适配器尝试。
# 必须显式 opt-in、允许 typesafe/ark，并提供逐供应商调用预算。
# 数据集前 10 行 = data/synthetic/v0.jsonl 头部（PRODUCT_DEFECT 区段）。
# ---------------------------------------------------------------------------


@pytest.mark.live(providers={"typesafe": 20, "ark": 20})
def test_live_benchmark_limit10_full_report() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "runs" / "live-limit10"
    report_path = run_benchmark(
        project_root / "data" / "synthetic" / "v0.jsonl",
        ["rule", "jev", "llm", "cascade"],
        limit=10,
        out_dir=out_dir,
    )
    experiment = BenchmarkExperiment.model_validate(json.loads(report_path.read_text(encoding="utf-8")))
    assert experiment.arms == ["rule", "jev", "llm"]
    assert [entry.model_dump() for entry in (experiment.absent_arms or [])] == [
        {"arm": "cascade", "reason": CASCADE_ABSENT_REASON}
    ]
    assert all(m.sample_count == 10 for m in experiment.metrics.values())
    md_path = report_path.parent / "experiment-report.md"
    md_text = md_path.read_text(encoding="utf-8")
    assert _DECLARATION_SYNTHETIC in md_text
    print(md_text)
