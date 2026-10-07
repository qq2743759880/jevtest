"""Rule baseline v0 引擎：规则即数据，顺序语义完全来自 contracts/rule-baseline-v0.yaml 的声明顺序，
引擎内禁止任何隐式排序。输出 reason = YAML reason_template 填充 + 尾部版本标记
（TriageResult 无 rule_version 字段且模型契约冻结，故按任务约定放 reason 尾部 `| rule_version=0`）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from jevtest.models import FailureState, TriageResult
from jevtest.resource_paths import resource_path

_CONTRACTS_DIR = resource_path("contracts")

_ERROR_CODE_TABLE_KEYS = {"code", "match_keyword", "category", "action", "priority", "source_url"}


@lru_cache(maxsize=1)
def _load_rule_contract() -> dict[str, Any]:
    with (_CONTRACTS_DIR / "rule-baseline-v0.yaml").open(encoding="utf-8") as f:
        contract: dict[str, Any] = yaml.safe_load(f)
    if contract.get("execution") != "sequential_first_match":
        raise ValueError("rule-baseline contract must declare execution=sequential_first_match")
    return contract


@lru_cache(maxsize=1)
def _load_error_code_table() -> tuple[dict[str, Any], ...]:
    # R3 逐条 priority 来自表本身；加载即校验，坏表 fail-fast 而非运行期静默错配。
    with (_CONTRACTS_DIR / "error-codes-v0.yaml").open(encoding="utf-8") as f:
        entries: list[dict[str, Any]] = yaml.safe_load(f)["entries"]
    for entry in entries:
        missing = _ERROR_CODE_TABLE_KEYS - set(entry)
        if missing:
            raise ValueError(f"error-codes-v0 entry missing keys {missing}: {entry}")
    return tuple(entries)


def _flip_count(previous_runs: list[str]) -> int:
    """PASS<->FAIL 翻转次数：相邻运行结果不同的位置数。"""
    return sum(1 for before, after in zip(previous_runs, previous_runs[1:]) if before != after)


def _match_error_codes(assertion: str) -> dict[str, Any] | None:
    lowered = assertion.lower()
    for entry in _load_error_code_table():
        if str(entry["match_keyword"]).lower() in lowered:
            return dict(entry)
    return None


def _evaluate(rule: dict[str, Any], state: FailureState) -> dict[str, Any] | None:
    """按 when 子句求值，命中返回填充 reason_template 所需的上下文，未命中返回 None。"""
    when = rule["when"]
    if when == "always":  # R4 兜底
        # 断言文本可能很长，截断防 reason 超 TriageResult 200 上限（前缀+版本标记约占 50）
        return {"assertion": state.assertion[:150]}
    if "any_env_false" in when:
        for key, value in state.env.items():
            if value is False:
                return {"failing_key": key}
        return None
    if "assertion_matches_table" in when:
        matched = _match_error_codes(state.assertion)
        return dict(matched) if matched else None
    # 剩余为 retry + flip_count_gte 组合条件（R1）
    if "retry" in when and state.retry != when["retry"]:
        return None
    flips = _flip_count(state.previous_runs)
    if "flip_count_gte" in when and flips < when["flip_count_gte"]:
        return None
    return {"flip_count": flips}


def _priority_output(rule: dict[str, Any], context: dict[str, Any]) -> int:
    spec = rule["priority_output"]
    if spec == "from_table":
        return int(context["priority"])
    if isinstance(spec, list):
        # YAML 声明区间（如 [1, 2]），引擎按契约注释取下限
        return min(spec)
    return int(spec)


def triage_by_rules(state: FailureState | dict[str, Any]) -> TriageResult:
    """Rule 臂入口：顺序执行 YAML 声明的规则，先命中先输出。

    接受 dict 是为在边界处显式校验——缺字段的 dict 在此抛 pydantic ValidationError，绝不静默输出。
    """
    if not isinstance(state, FailureState):
        state = FailureState.model_validate(state)
    contract = _load_rule_contract()
    for rule in contract["rules"]:
        context = _evaluate(rule, state)
        if context is None:
            continue
        if rule["priority_output"] == "from_table":
            category: Any = context["category"]
            action: Any = context["action"]
        else:
            category = rule["output"]["category"]
            action = rule["output"]["action"]
        reason = rule["reason_template"].format(**context)
        reason = f"{reason} | rule_version={contract['rule_version']}"
        # R1-R3 命中为确定性规则判定；R4 兜底代表无证据，confidence=0 交由漏斗路由
        confidence = 0.0 if rule["when"] == "always" else 1.0
        return TriageResult(
            category=category,
            action=action,
            priority=_priority_output(rule, context),
            confidence=confidence,
            reason=reason,
        )
    raise AssertionError("R4_FALLBACK (when: always) 未命中：规则契约缺少兜底规则")
