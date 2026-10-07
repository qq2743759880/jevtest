"""Benchmark runner（T09）：四臂竞争实验的执行核心 + 报告生成（PRD §5.1 条款 1/2 的机制化）。

条款 1（输入对等）机制化：
- 数据集加载**一次**，得到同一份 FailureState 列表对象，逐臂传递同一对象（不是逐臂重拷）；
- 每臂开始前与结束后各 dump 一次当前 state 列表 JSON（sort_keys 固定键序）与基准快照比对，
  不一致立即抛 `InputParityError` 中止实验——防臂间污染不靠自觉，靠断言。

条款 2（输出同构）机制化：
- 逐臂逐样本走 MeasureRunner（provider.predict_validated 统一入口），计量口径全部来自
  metrics 层（cost 逐次记账差 / latency / E_PARSE_FAILURE 不中断批次 + error_samples.jsonl）。

cost 口径（冻结）：
- rule 臂：本地规则，无 API 费用，cost 恒 0.0；
- jev 臂：S1 探测单价（docs/jev-s1-probe.md）——$42/Btok 输入 token（输出免费），
  换算 USD 进 cost（经 _JevCostedProvider 包装逐次记账）；
- llm 臂：Ark 无公开单价 → cost=0.0，token 数已由 LLMJudgeProvider 记账
  （total_prompt_tokens / total_completion_tokens）；cost_currency 注明
  "USD (jev); llm: tokens only, price not frozen"。
- cascade 臂：T11 未实现——被请求时记入 absent_arms（不报错、不假装跑过）。

accuracy：expected 标签在 dataset 行顶层，category 命中即对；parse_failure 兜底出的
UNKNOWN 如实参与 accuracy（不修饰）。routing_ratio：本实验无 cascade，每臂 {self: 1.0}。
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jevtest.metrics import ErrorLog, MeasureRunner, aggregate
from jevtest.models import BenchmarkExperiment, FailureState
from jevtest.resource_paths import resource_path
from jevtest.pricing import JEV_INPUT_USD_PER_TOKEN
from jevtest.providers.base import Provider

#: 允许的臂名（与 benchmark.schema / models._ArmName 一致）。
VALID_ARMS: frozenset[str] = frozenset({"rule", "jev", "llm", "cascade"})

#: 已实现臂（cascade 属 T11，未实现——请求即入 absent_arms，不抛错）。
IMPLEMENTED_ARMS: frozenset[str] = frozenset({"rule", "jev", "llm"})

CASCADE_ABSENT_REASON = "cascade not implemented (T11 pending)"

#: 报告 cost_currency 注记（任务冻结口径：llm 无冻结单价，只报 token）。
COST_CURRENCY_NOTE = "USD (jev); llm: tokens only, price not frozen"

_COST_CURRENCY_BY_ARM = {
    "rule": "USD (no API cost)",
    "jev": "USD (jev: $42/Btok input per S1 probe, output free)",
    "llm": COST_CURRENCY_NOTE,
    "cascade": "USD",
}

_SYNTHETIC_DECLARATION = "SYNTHETIC — 不代表企业效果"
_REAL_DECLARATION = "REAL — 正式基准"

_CONTRACTS_DIR = resource_path("contracts")


class UnknownArmError(ValueError):
    """请求了 rule/jev/llm/cascade 之外的臂名（E_SCHEMA_INVALID_INPUT 家族，拒绝而非猜测）。"""


class InputParityError(RuntimeError):
    """条款 1 输入对等断言失败：某臂运行前后 state 列表与基准快照不一致（臂间污染）。"""


class EvaluationNotEligibleError(ValueError):
    """Legacy benchmark v1 cannot faithfully represent non-synthetic evaluation output."""


# ---------------------------------------------------------------------------
# provider 构造
# ---------------------------------------------------------------------------


class _JevCostedProvider(Provider):
    """JevLiveProvider 的计量包装：按 S1 单价把 input_tokens 折算 USD 累进 accumulated_cost。

    MeasureRunner 的 cost 口径是 accumulated_cost 逐次差，所以包装层只负责在每次
    predict 后（含解析失败路径——last_usage 在适配器内已更新）推进记账，判定逻辑
    零改动地委托内层 provider（单一事实源仍是 JevLiveProvider）。
    """

    arm_name = "jev"
    is_mock = False

    def __init__(self, inner: Provider) -> None:
        self._inner = inner
        self.accumulated_cost = 0.0
        self.total_input_tokens = 0

    def predict(self, state: FailureState):  # noqa: ANN201 - 与 Provider.predict 同形
        usage_before = getattr(self._inner, "last_usage", None)
        try:
            return self._inner.predict(state)
        finally:
            usage = getattr(self._inner, "last_usage", None)
            if usage is not None and usage is not usage_before:
                input_tokens = int(usage.get("input_tokens") or 0)
                self.total_input_tokens += input_tokens
                self.accumulated_cost = round(
                    self.accumulated_cost + input_tokens * JEV_INPUT_USD_PER_TOKEN, 12
                )

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()


def default_provider_factory(arm: str) -> Provider:
    """默认臂构造：rule/jev/llm 真实臂；cascade 未实现（T11）在此之前已被过滤。"""
    if arm == "rule":
        from jevtest.providers.rule import RuleProvider

        return RuleProvider()
    if arm == "jev":
        from jevtest.providers.jev_live import JevLiveProvider

        return _JevCostedProvider(JevLiveProvider())
    if arm == "llm":
        from jevtest.providers.llm_judge import LLMJudgeProvider

        return LLMJudgeProvider()
    raise UnknownArmError(f"arm {arm!r} has no provider implementation")


# ---------------------------------------------------------------------------
# 数据集加载与输入对等快照
# ---------------------------------------------------------------------------


def load_states(
    dataset_path: str | Path,
    limit: int | None = None,
    sampling: str = "first",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """加载 JSONL 数据集（行 = {"expected": ..., "failure_state": {...}}），返回
    [{"expected": str, "state": FailureState}, ...]。expected 在行顶层（不在 FailureState 内，
    契约 additionalProperties=false）。

    sampling：first=按文件顺序取前 N 行（默认，smoke 用）；random=整体随机抽样；
    stratified=按 expected 类别配额分层抽样（校准/正式基准必须用此选项——
    本合成数据集按类别分块排列，first 抽到的是单一类别块，不具代表性）。
    random/stratified 使用 random.Random(seed)，同 seed 可复现。"""
    if sampling not in ("first", "random", "stratified"):
        raise ValueError(f"unknown sampling strategy: {sampling}")
    rows: list[dict[str, Any]] = []
    with open(dataset_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "expected": row["expected"],
                    "state": FailureState.model_validate(row["failure_state"]),
                }
            )
    if not rows:
        raise ValueError(f"dataset {dataset_path} has no records")
    if limit is None or sampling == "first":
        return rows if limit is None else rows[:limit]
    rng = random.Random(seed)
    if sampling == "random":
        return rng.sample(rows, min(limit, len(rows)))
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row["expected"], []).append(row)
    labels = sorted(by_label)  # 排序保证同 seed 可复现（不依赖 dict 插入序）
    per_label = limit // len(labels)
    remainder = limit % len(labels)
    sampled: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        take = per_label + (1 if i < remainder else 0)
        pool = by_label[label]
        sampled.extend(rng.sample(pool, min(take, len(pool))))
    rng.shuffle(sampled)  # 混洗避免臂内顺序与类别块相关
    return sampled


def _snapshot(states: Sequence[FailureState]) -> str:
    """当前 state 列表的逐字节快照（sort_keys 固定键序，与 SampleRecord.to_json_line 同纪律）。"""
    return json.dumps(
        [state.model_dump() for state in states], sort_keys=True, ensure_ascii=False
    )


def _assert_parity(states: Sequence[FailureState], baseline: str, phase: str) -> None:
    current = _snapshot(states)
    if current != baseline:
        # 定位第一条分歧，便于归因到具体样本。
        old = json.loads(baseline)
        new = json.loads(current)
        first_index = next(
            (i for i, (a, b) in enumerate(zip(old, new)) if a != b),
            min(len(old), len(new)),
        )
        raise InputParityError(
            f"input parity violation ({phase}): state list diverges from baseline "
            f"at record index {first_index} — an arm mutated or replaced shared inputs"
        )


# ---------------------------------------------------------------------------
# 实验执行
# ---------------------------------------------------------------------------


def _rule_version() -> str:
    with (_CONTRACTS_DIR / "rule-baseline-v0.yaml").open(encoding="utf-8") as f:
        contract = yaml.safe_load(f)
    return str(contract["rule_version"])


def _dataset_kind(states: list[dict[str, Any]]) -> str:
    synthetic_flags = [rec["state"].provenance.synthetic for rec in states]
    if all(synthetic_flags):
        return "synthetic"
    if not any(synthetic_flags):
        return "real"
    return "mixed"


def _normalize_arms(arms: Sequence[str]) -> tuple[list[str], list[dict[str, str]]]:
    """校验并去重臂名；返回 (参与臂, absent_arms)。未知臂名拒绝；cascade 请求即缺席。"""
    seen: list[str] = []
    for arm in arms:
        if arm not in VALID_ARMS:
            raise UnknownArmError(
                f"unknown arm {arm!r}; valid arms: {sorted(VALID_ARMS)}"
            )
        if arm not in seen:
            seen.append(arm)
    if not seen:
        raise UnknownArmError("no arms requested")
    participating = [arm for arm in seen if arm in IMPLEMENTED_ARMS]
    if not participating:
        raise UnknownArmError(
            "all requested arms are unimplemented: "
            + ", ".join(f"{arm} ({CASCADE_ABSENT_REASON})" for arm in seen)
        )
    absent = [
        {"arm": arm, "reason": CASCADE_ABSENT_REASON}
        for arm in seen
        if arm not in IMPLEMENTED_ARMS
    ]
    return participating, absent


def run_benchmark(
    dataset_path: str | Path,
    arms: Sequence[str],
    *,
    limit: int | None = None,
    sampling: str = "first",
    seed: int = 0,
    out_dir: str | Path = "runs/benchmark",
    provider_factory: Callable[[str], Provider] | None = None,
) -> Path:
    """跑一次四臂竞争实验，返回 experiment-report.json 路径。

    - 参与臂共用同一份 state 列表对象；每臂前后做输入对等快照断言，违约立即中止。
    - 逐臂逐样本 MeasureRunner.run_one；per-sample JSONL（每臂一个）+ error_samples.jsonl
      同步落盘；experiment-report.json 过 BenchmarkExperiment 模型校验后写盘，
      并由 render_report 生成 experiment-report.md。
    """
    factory = provider_factory if provider_factory is not None else default_provider_factory
    participating, absent_arms = _normalize_arms(arms)

    records = load_states(dataset_path, limit, sampling=sampling, seed=seed)
    dataset_kind = _dataset_kind(records)
    if dataset_kind != "synthetic":
        raise EvaluationNotEligibleError(
            "legacy benchmark v1 labels every non-synthetic run as a formal benchmark and carries "
            "no eligibility evidence; use RunReport v2 for exploratory reporting"
        )

    # Eligibility must be decided before creating output files or invoking providers.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    states = [rec["state"] for rec in records]
    expected_labels = [rec["expected"] for rec in records]
    baseline = _snapshot(states)

    error_log = ErrorLog(out / "error_samples.jsonl")
    # 无失败也要让报告里的 error_samples_path 指向真实存在的文件（空 JSONL = 零解析失败）。
    error_log.path.touch(exist_ok=True)
    per_arm_records: dict[str, list[Any]] = {}
    providers: dict[str, Provider] = {}

    for arm in participating:
        _assert_parity(states, baseline, phase=f"before arm {arm!r} starts")
        provider = factory(arm)
        providers[arm] = provider
        runner = MeasureRunner(provider, error_log)
        sample_records: list[Any] = []
        per_sample_path = out / f"per_sample_{arm}.jsonl"
        with per_sample_path.open("w", encoding="utf-8", newline="\n") as fh:
            for state in states:
                _result, sample_record = runner.run_one(state)
                sample_records.append(sample_record)
                fh.write(sample_record.to_json_line() + "\n")
        per_arm_records[arm] = sample_records
        _assert_parity(states, baseline, phase=f"after arm {arm!r} finished")

    # ---- 指标聚合 ----
    metrics: dict[str, dict[str, Any]] = {}
    for arm in participating:
        rows = per_arm_records[arm]
        agg = aggregate(rows)[arm]
        correct = sum(
            1
            for row, expected in zip(rows, expected_labels)
            if row.category == expected
        )
        metrics[arm] = {
            "accuracy": correct / len(rows),
            "cost": agg["cost"],
            "cost_currency": _COST_CURRENCY_BY_ARM[arm],
            # cost_state（审计采纳项，臂级扩展字段 schema 合法）：known=单价已冻结可折算；
            # unknown=token 已记账但无公开单价（llm/Ark），读者不得把 cost=0 读成免费
            "cost_state": "known" if arm in ("rule", "jev") else "unknown",
            "latency_p50_ms": agg["latency_p50_ms"],
            "latency_p95_ms": agg["latency_p95_ms"],
            "routing_ratio": {arm: 1.0},  # 无 cascade 的单臂直通：100% 自身，如实填写
            "sample_count": agg["count"],
        }

    experiment_id = (
        "bench-"
        + records[0]["state"].provenance.dataset_version
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    experiment = BenchmarkExperiment(
        experiment_id=experiment_id,
        dataset_version=records[0]["state"].provenance.dataset_version,
        dataset_kind=dataset_kind,
        rule_version=_rule_version(),
        arms=participating,
        absent_arms=absent_arms or None,
        metrics=metrics,
        per_sample_path=";".join(f"per_sample_{arm}.jsonl" for arm in participating),
        error_samples_path="error_samples.jsonl",
        declaration=_SYNTHETIC_DECLARATION if dataset_kind == "synthetic" else _REAL_DECLARATION,
    )

    report_path = out / "experiment-report.json"
    report_path.write_text(
        json.dumps(experiment.model_dump(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out / "experiment-report.md").write_text(
        render_report(report_path, providers), encoding="utf-8", newline="\n"
    )
    return report_path


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def _token_notes(experiment: BenchmarkExperiment, providers: dict[str, Provider]) -> list[str]:
    notes: list[str] = []
    llm = providers.get("llm")
    if llm is not None and hasattr(llm, "total_prompt_tokens"):
        notes.append(
            f"llm arm tokens: prompt={llm.total_prompt_tokens}, "
            f"completion={llm.total_completion_tokens} (cost not frozen: no public Ark price)"
        )
    jev = providers.get("jev")
    if jev is not None and hasattr(jev, "total_input_tokens"):
        notes.append(f"jev arm input tokens: {jev.total_input_tokens} ($42/Btok, output free)")
    return notes


def render_report(experiment_path: str | Path, providers: dict[str, Provider] | None = None) -> str:
    """experiment-report.json → experiment-report.md（四指标表 + declaration 页眉 + absent_arms）。"""
    payload = json.loads(Path(experiment_path).read_text(encoding="utf-8"))
    experiment = BenchmarkExperiment.model_validate(payload)

    lines: list[str] = []
    lines.append(f"# JevTest Benchmark Report — {experiment.experiment_id}")
    lines.append("")
    lines.append(f"> **{experiment.declaration}**")
    lines.append("")
    # benchmark_scope（审计采纳项，仅 md 层——顶层 additionalProperties=false 禁止 JSON 加字段）：
    # 小样本 run 明确标注 smoke，防未来读者误当正式基准。按单臂最大样本数判定（多臂不叠加）
    max_arm_samples = max(m.sample_count for m in experiment.metrics.values())
    scope = (
        "smoke_validation_only（非正式基准，禁止作为效果结论）"
        if max_arm_samples < 30
        else "dataset_benchmark"
    )
    lines.append(f"- benchmark_scope: `{scope}`  |  samples_per_arm: {max_arm_samples}")
    lines.append(
        f"- dataset_version: `{experiment.dataset_version}`  |  dataset_kind: `{experiment.dataset_kind}`"
    )
    lines.append(f"- rule_version: `{experiment.rule_version}`")
    lines.append(f"- participating arms: {', '.join(experiment.arms)}")
    lines.append(f"- per-sample JSONL: {experiment.per_sample_path}")
    lines.append(f"- error samples: {experiment.error_samples_path} (E_PARSE_FAILURE, exit 0)")
    lines.append("")

    if experiment.absent_arms:
        lines.append("## Absent arms")
        lines.append("")
        for entry in experiment.absent_arms:
            lines.append(f"- **{entry.arm}**: {entry.reason}")
        lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| Arm | Accuracy | Cost | Latency p50 (ms) | Latency p95 (ms) | Routing Ratio | Samples |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in experiment.arms:
        m = experiment.metrics[arm]
        routing = ", ".join(f"{k}:{v:.2f}" for k, v in m.routing_ratio.items())
        cost_state = getattr(m, "cost_state", "known")
        cost_cell = f"{m.cost:.6f} {m.cost_currency}"
        if cost_state != "known":
            cost_cell += f" (**cost_state: {cost_state}**)"
        lines.append(
            f"| {arm} | {m.accuracy:.3f} | {cost_cell} "
            f"| {m.latency_p50_ms:.1f} | {m.latency_p95_ms:.1f} | {{{routing}}} "
            f"| {m.sample_count} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(f"- cost_currency 口径: {COST_CURRENCY_NOTE}。")
    lines.append(
        "- live 臂（jev/llm）的延迟与准确率波动仅可归因于 API 温度（非 mock 臂无确定性保证）；"
        "mock 臂双跑指标应逐字节一致。"
    )
    lines.append(
        "- accuracy 口径：dataset 行顶层 expected 标签，category 命中即对；"
        "parse_failure 兜底（UNKNOWN+HUMAN_TRIAGE）如实计入，不修饰。"
    )
    if experiment.dataset_kind == "synthetic":
        lines.append(
            "- 数据为合成数据集（裁定 2）：以上指标不代表企业真实日志效果，"
            "真实脱敏样本 >=100 条后另行汇报。"
        )
    if providers:
        for note in _token_notes(experiment, providers):
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)
