"""兜底层（PRD §10.2）：无依据的判定视为无效输出，路由器按 UNKNOWN+HUMAN_TRIAGE 兜底。

分工边界：
- 非法枚举 / 越界值等硬违规：TriageResult 模型层已拦截（predict_validated 抛
  InvalidProviderOutput），不会以"合法模型实例"的形态到达这里。
- reason 为空 / 纯空白：模型层 min_length=1 拦不住 `\" \"` 这类值，属于软失败，
  由 `sanitize` 统一替换为 UNKNOWN + HUMAN_TRIAGE。
- 为兼容上游用 model_construct 绕过校验产出的坏实例（模拟违规 provider），
  sanitize 会对输入强制重过一次模型校验：校验不过同样落入兜底。
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from jevtest.models import TriageResult

_FALLBACK_REASON = "invalid provider output, fell back"


def _fallback_result() -> TriageResult:
    # confidence=0.0：无任何依据，交由漏斗路由器按阈值策略处置（与 Rule 臂 R4 兜底同语义）。
    return TriageResult(
        category="UNKNOWN",
        action="HUMAN_TRIAGE",
        priority=3,
        confidence=0.0,
        reason=_FALLBACK_REASON,
    )


def sanitize(result: Any) -> TriageResult:
    """reason 为空（或纯空白）→ 视为无效，替换为 UNKNOWN/HUMAN_TRIAGE/priority 3。"""
    try:
        payload = result.model_dump() if isinstance(result, TriageResult) else result
        validated = TriageResult.model_validate(payload)
    except (ValidationError, TypeError, ValueError, AttributeError):
        return _fallback_result()
    if not (validated.reason or "").strip():
        return _fallback_result()
    return validated
