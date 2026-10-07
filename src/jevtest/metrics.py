"""计量层（T08）：逐样本 cost / latency / parse_failure 落盘，供 ROI 模型与四臂成本对比。

口径约定：
- latency：mock 臂取 provider 的 simulated_latency_ms 逐次记账差（确定性纪律：真实墙钟
  不允许进 mock 结果）；非 mock 臂取 time.perf_counter 真实耗时。
- cost：provider 累计记账（accumulated_cost）逐次差；provider 无记账字段则恒为 0.0。
- parse_failure（E_PARSE_FAILURE，exit 0）：臂输出过不了 TriageResult 契约校验时
  不中断批次，样本计 UNKNOWN + HUMAN_TRIAGE（复用 fallback.sanitize），并同步落
  error_samples.jsonl（行 = {case_id, arm, raw_output, timestamp}，契约 owner 审阅增强项）。
- accuracy 不在本层聚合（需要标注，属基准报告层职责）；本层只出 count / cost 总和 /
  latency p50 / p95 / parse_failure 计数。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jevtest.models import FailureState, TriageResult
from jevtest.execution_contracts import AttemptRecordV1
from jevtest.providers.base import InvalidProviderOutput, Provider
from jevtest.providers.fallback import sanitize

ArmName = Literal["rule", "jev", "llm", "cascade"]


def _utc_now_iso() -> str:
    """ISO8601 UTC 时间戳（带 +00:00 偏移，可被 datetime.fromisoformat 逆向解析）。"""
    return datetime.now(timezone.utc).isoformat()


class SampleRecord(BaseModel):
    """逐样本计量行——benchmark.schema 的 per_sample_path 行格式：TriageResult 关键字段 + arm + case_id + latency_ms + cost，附 mock / parse_failure 标记。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    arm: ArmName
    category: str
    action: str
    priority: int
    confidence: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    cost: float = Field(ge=0, description="本样本费用；mock 臂恒为 0")
    is_mock: bool = Field(description="mock 臂标记：费用恒 0，可区分于真实费用")
    parse_failure: bool = Field(
        default=False,
        description="E_PARSE_FAILURE 标记：True 表示该样本走了 sanitize 兜底（UNKNOWN）",
    )
    timestamp: str = Field(description="ISO8601 UTC")

    @field_validator("timestamp")
    @classmethod
    def _check_utc_iso8601(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)  # 非 ISO8601 直接抛 ValidationError
        offset = parsed.utcoffset()
        if offset is None or offset != timedelta(0):
            raise ValueError(f"timestamp must be UTC (offset 0), got {value!r}")
        return value

    def to_json_line(self) -> str:
        """sort_keys 单行 JSON 序列化（键序固定 → 逐字节可比、diff 友好）。换行由写入方统一用 \\n。"""
        return json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)


class _Bookkeeping(NamedTuple):
    """provider 计量记账快照（latency 仅 mock 臂有意义）。"""

    latency_ms: float
    cost: float


class ErrorLog:
    """E_PARSE_FAILURE 追踪落盘：每行 {case_id, arm, raw_output, timestamp}，固定 \\n 换行。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self, case_id: str, arm: str, raw_output: Any, timestamp: str | None = None
    ) -> None:
        def json_safe(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump(mode="json")
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            return value

        row = {
            "case_id": case_id,
            "arm": arm,
            "raw_output": json_safe(raw_output),
            "timestamp": timestamp if timestamp is not None else _utc_now_iso(),
        }
        line = json.dumps(row, sort_keys=True, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


class MeasureRunner:
    """包装任意 Provider：run_one 产出 (TriageResult, SampleRecord)，解析失败不中断批次。"""

    def __init__(self, provider: Provider, error_log: ErrorLog | None = None) -> None:
        self.provider = provider
        self.error_log = error_log
        self.last_attempts: list[AttemptRecordV1] = []

    def _attempt_ledger(self) -> list[AttemptRecordV1]:
        """Prefer a wrapped provider's innermost ledger when an existing decorator is used."""
        provider: Any = self.provider
        selected: list[AttemptRecordV1] = []
        visited: set[int] = set()
        while provider is not None and id(provider) not in visited:
            visited.add(id(provider))
            attempts = getattr(provider, "last_attempts", None)
            if attempts:
                selected = list(attempts)
            provider = getattr(provider, "_inner", None)
        return selected

    def run_one(
        self, state: FailureState, *, max_attempts: int | None = None
    ) -> tuple[TriageResult, SampleRecord]:
        start = perf_counter()
        self.last_attempts = []
        if hasattr(self.provider, "reset_attempts"):
            self.provider.reset_attempts()
        book0 = self._bookkeeping()
        raw_output: Any = None
        parse_failure = False
        provider_failure = False
        try:
            result = self.provider.predict_validated(state, max_attempts=max_attempts)
        except InvalidProviderOutput as exc:
            parse_failure = True
            raw_output = exc.raw_output
            if raw_output is None:
                raw_output = getattr(self.provider, "last_raw_output", None)
            result = sanitize(raw_output)
        except Exception as exc:
            if not self._is_provider_service_error(exc):
                self.last_attempts = self._attempt_ledger()
                raise
            provider_failure = True
            raw_output = getattr(self.provider, "last_raw_output", None)
            if raw_output is None:
                raw_output = {"error_type": type(exc).__name__}
            result = sanitize(None)
        except BaseException:
            # Cancellation/interrupts still expose the already completed wire attempts to the caller.
            self.last_attempts = self._attempt_ledger()
            raise
        book1 = self._bookkeeping()
        self.last_attempts = self._attempt_ledger()
        elapsed_ms = (perf_counter() - start) * 1000.0
        timestamp = _utc_now_iso()
        if (parse_failure or provider_failure) and self.error_log is not None:
            self.error_log.append(state.case_id, self.provider.arm_name, raw_output, timestamp)
        latency_ms = (
            book1.latency_ms - book0.latency_ms if self.provider.is_mock else elapsed_ms
        )
        cost = round(book1.cost - book0.cost, 12)
        record = SampleRecord(
            case_id=state.case_id,
            arm=self.provider.arm_name,  # type: ignore[arg-type]
            category=result.category,
            action=result.action,
            priority=result.priority,
            confidence=result.confidence,
            latency_ms=latency_ms,
            cost=cost,
            is_mock=self.provider.is_mock,
            parse_failure=parse_failure,
            timestamp=timestamp,
        )
        return result, record

    @staticmethod
    def _is_provider_service_error(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        try:
            import openai

            if isinstance(exc, openai.APIError):
                return True
        except ImportError:
            pass
        try:
            from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError

            return isinstance(exc, (TypeSafeAPIConnectionError, TypeSafeAPIError))
        except ImportError:
            return False

    def _bookkeeping(self) -> _Bookkeeping:
        provider = self.provider
        latency = (
            float(getattr(provider, "accumulated_latency_ms", 0.0)) if provider.is_mock else 0.0
        )
        cost = float(getattr(provider, "accumulated_cost", 0.0))
        return _Bookkeeping(latency_ms=latency, cost=cost)


def _nearest_rank(sorted_values: list[float], q: float) -> float:
    """nearest-rank 百分位：ceil(q*n) 位（1-based）。确定性、可手算复现。"""
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


def aggregate(records: Sequence[SampleRecord]) -> dict[str, dict[str, Any]]:
    """按 arm 聚合：count / cost 总和 / latency p50、p95（nearest-rank）/ parse_failure 计数。

    accuracy 不在此计算——需要 ground truth 标注，属基准报告层（T09/T12）职责。
    """
    buckets: dict[str, list[SampleRecord]] = {}
    for record in records:
        buckets.setdefault(record.arm, []).append(record)
    report: dict[str, dict[str, Any]] = {}
    for arm in sorted(buckets):
        rows = buckets[arm]
        latencies = sorted(row.latency_ms for row in rows)
        report[arm] = {
            "count": len(rows),
            "cost": round(sum(row.cost for row in rows), 12),
            "latency_p50_ms": _nearest_rank(latencies, 0.50),
            "latency_p95_ms": _nearest_rank(latencies, 0.95),
            "parse_failures": sum(1 for row in rows if row.parse_failure),
        }
    return report
