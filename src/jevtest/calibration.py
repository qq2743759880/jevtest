"""Confidence 校准预演（T10 合成预演模式，owner 裁定 C 授权）。

边界（tasks/task10-calibration.md「合成预演模式约束」）：
- 本模块产出只是管道预演 + 观测数据，**不是阈值决定**——只出 observed calibration
  curve 与 threshold candidates（候选），不产出任何 approved threshold。
- 采样强制 stratified（benchmark.load_states(sampling="stratified")）——本数据集按
  类别分块排列，first 抽到单一类别块，不具代表性。
- 语义分离（PRD §10.2）：confidence 承载 Probs[Value]（正确概率），confidence_sharpness
  承载 Jev 原生 Confidence（分布尖锐度）。两套分箱分别出，禁止混用。
- cost 口径冻结沿用 benchmark：只呈现 jev 的 USD 与 token 量，不做跨臂 cost 排名
  （llm 臂 cost_state=unknown，本模块也不跑 llm 臂）。

纯离线函数（bin_calibration）只吃 records（dict 列表），不依赖网络，单测友好。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jevtest.benchmark import _JevCostedProvider, load_states
from jevtest.metrics import ErrorLog, MeasureRunner
from jevtest.models import FailureState, TriageResult
from jevtest.providers.base import Provider

#: 默认每类配额（6 类 × 10 = 60 条）。
DEFAULT_PER_CATEGORY = 10

#: 预演产物默认落盘目录。
DEFAULT_OUT_DIR = Path("data") / "calibration"

#: 逐样本 JSONL 的字段名（confidence=Probs[Value]，confidence_sharpness=分布尖锐度，
#: 两字段分离是本预演的核心观测对象，禁止合并）。
RECORD_FIELDS = (
    "case_id",
    "expected",
    "predicted",
    "correct",
    "action",
    "priority",
    "confidence",
    "confidence_sharpness",
    "probs",
    "latency_ms",
    "cost_usd",
    "parse_failure",
)


def _count_labels(dataset_path: str | Path) -> int:
    """数据集顶层 expected 的去重类数（用于 limit = per_category × 类数，凑齐每类配额）。"""
    labels: set[str] = set()
    with open(dataset_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                labels.add(str(json.loads(line)["expected"]))
    if not labels:
        raise ValueError(f"dataset {dataset_path} has no records")
    return len(labels)


def _record_from_result(
    state: FailureState,
    expected: str,
    result: TriageResult,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """(state, expected, TriageResult, SampleRecord) → 逐样本校准记录（纯函数，可离线测）。"""
    return {
        "case_id": state.case_id,
        "expected": expected,
        "predicted": result.category,
        "correct": result.category == expected,
        "action": result.action,
        "priority": result.priority,
        # 语义分离（PRD §10.2）：confidence=Probs[Value]（正确概率），
        # confidence_sharpness=Jev 原生 Confidence（分布尖锐度）。
        "confidence": float(result.confidence),
        "confidence_sharpness": (
            float(result.confidence_sharpness)
            if result.confidence_sharpness is not None
            else None
        ),
        "probs": result.probs,
        "latency_ms": float(sample["latency_ms"]),
        "cost_usd": float(sample["cost"]),
        "parse_failure": bool(sample["parse_failure"]),
    }


def run_calibration_preview(
    dataset_path: str | Path,
    seed: int = 0,
    per_category: int = DEFAULT_PER_CATEGORY,
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    provider_factory: Callable[[], Provider] | None = None,
) -> Path:
    """合成校准预演：stratified 抽样 → 逐条 Jev（MeasureRunner 计量）→ 逐样本 JSONL。

    返回 JSONL 路径（`<out_dir>/preview-<seed>.jsonl`）。逐行字段见 RECORD_FIELDS。

    - 抽样：benchmark.load_states(sampling="stratified", seed=seed)，每类 per_category
      条（limit = per_category × 类数），全类别覆盖、同 seed 可复现。
    - provider：默认 JevLiveProvider（经 benchmark 的 costed 包装，USD 记账口径冻结）；
      测试可注入 provider_factory（离线 stub）。
    - 本函数不产出任何阈值——阈值分析属 docs/synthetic-calibration-preview.md 的
      threshold candidates（候选，未经 owner 批准），由 bin_calibration 的观测数据支撑。
    """
    n_labels = _count_labels(dataset_path)
    limit = per_category * n_labels
    records = load_states(dataset_path, limit, sampling="stratified", seed=seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / f"preview-{seed}.jsonl"

    provider = provider_factory() if provider_factory is not None else _JevCostedProvider(
        _default_jev_provider()
    )
    error_log = ErrorLog(out / "error_samples.jsonl")
    runner = MeasureRunner(provider, error_log)

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            result, sample = runner.run_one(rec["state"])
            row = _record_from_result(rec["state"], rec["expected"], result, sample.model_dump())
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    close = getattr(provider, "close", None)
    if callable(close):
        close()
    return jsonl_path


def _default_jev_provider() -> Provider:
    from jevtest.providers.jev_live import JevLiveProvider

    return JevLiveProvider()


def load_preview_records(path: str | Path) -> list[dict[str, Any]]:
    """逐样本 JSONL → records（bin_calibration 的输入形态；离线可复算）。"""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# 分箱（纯离线：给 records 就能算，不依赖网络）
# ---------------------------------------------------------------------------


def _bin_label(low: float, high: float) -> str:
    return f"{low:.1f}-{high:.1f}"


def _in_bin(value: float, low: float, high: float, *, is_last: bool) -> bool:
    """左闭右开；末箱右闭（confidence=1.0 落最后一箱，不被丢掉）。"""
    if is_last:
        return low <= value <= high
    return low <= value < high


def bin_calibration(
    records: Sequence[Mapping[str, Any]],
    n_bins: int = 5,
    value_field: str = "confidence",
) -> list[dict[str, Any]]:
    """按 value_field 分箱（默认 confidence=Probs[Value]；尖适度用 value_field=
    "confidence_sharpness" 再跑一套），等宽 n_bins 箱（0-0.2/.../0.8-1.0）。

    每箱输出：bin 标签、low/high、count、accuracy（该箱 correct 比例；空箱为 None）、
    mean_confidence 与 mean_sharpness（箱内均值，缺 None 值的记录跳过该均值）。
    value_field 缺失/为 None 的记录不计入任何箱（如实留空，不猜测）。
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        is_last = i == n_bins - 1
        rows = [
            r
            for r in records
            if r.get(value_field) is not None
            and _in_bin(float(r[value_field]), low, high, is_last=is_last)
        ]
        count = len(rows)
        confidences = [float(r["confidence"]) for r in rows if r.get("confidence") is not None]
        sharpnesses = [
            float(r["confidence_sharpness"]) for r in rows if r.get("confidence_sharpness") is not None
        ]
        bins.append(
            {
                "bin": _bin_label(low, high),
                "low": low,
                "high": high,
                "count": count,
                "accuracy": (sum(1 for r in rows if r.get("correct")) / count) if count else None,
                "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
                "mean_sharpness": (sum(sharpnesses) / len(sharpnesses)) if sharpnesses else None,
            }
        )
    return bins
