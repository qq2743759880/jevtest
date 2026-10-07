"""契约一致性：导出冻结和版本化 schema，并与 contracts/ 文本做零漂移校验。

比较策略（与任务 T02 约定一致）：逐节点比较字段名集合、required 集合、枚举值集合、
数值/长度约束（minimum/maximum/minLength/maxLength/minItems/maxItems）。
description 以及手写 JSON 的 $id/$schema 头部差异忽略。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from jevtest.execution_contracts import VERSIONED_CONTRACTS
from jevtest.models import BenchmarkExperiment, FailureState, TriageResult


class ContractDriftError(Exception):
    """pydantic 导出 schema 与 contracts/ 冻结文本不一致（E_CONTRACT_DRIFT）。"""


_MODEL_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("FailureState", FailureState, "failure-state.schema.json"),
    ("TriageResult", TriageResult, "triage-result.schema.json"),
    ("BenchmarkExperiment", BenchmarkExperiment, "benchmark.schema.json"),
) + VERSIONED_CONTRACTS

_VALUE_ATTRS = ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems")
_CHILD_KEYS = ("items", "properties", "required", "patternProperties", "additionalProperties")


def export_schemas() -> dict[str, dict[str, Any]]:
    """按契约文件同名导出所有 JSON Schema。"""
    return {name: model.model_json_schema() for name, model, _ in _MODEL_CONTRACTS}


def verify_contracts(contracts_dir: str | Path) -> None:
    """比较导出 schema 与 contracts/*.json；存在任何差异则抛出 ContractDriftError。"""
    directory = Path(contracts_dir)
    exported = export_schemas()
    diffs: list[str] = []
    for name, _, filename in _MODEL_CONTRACTS:
        contract = json.loads((directory / filename).read_text(encoding="utf-8"))
        _compare(contract, exported[name], name, contract, exported[name], diffs)
    if diffs:
        raise ContractDriftError("contract drift detected:\n" + "\n".join(diffs))


def _deref(node: Any, root: dict[str, Any]) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        target: Any = root
        for part in node["$ref"].lstrip("#/").split("/"):
            target = target[part]
        node = target
    return node


def _type_set(node: dict[str, Any]) -> set[str]:
    value = node.get("type")
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def _unwrap(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """把 anyOf（可选字段/联合类型）展开为合并视图：剔除 null 成员后合并约束。"""
    if "anyOf" not in node:
        return node
    subs = [s for s in (_deref(item, root) for item in node["anyOf"])
            if isinstance(s, dict) and s.get("type") != "null"]
    merged: dict[str, Any] = {}
    types: set[str] = set()
    enums: list[Any] = []
    consts: list[Any] = []
    for sub in subs:
        types |= _type_set(sub)
        if "enum" in sub:
            enums.extend(sub["enum"])
        if "const" in sub:
            consts.append(sub["const"])
        for attr in _VALUE_ATTRS:
            if attr in sub:
                merged.setdefault(attr, sub[attr])
        for key in _CHILD_KEYS:
            if key in sub:
                merged.setdefault(key, sub[key])
    if types:
        merged["type"] = sorted(types)
    if enums:
        merged["enum"] = enums
    if consts:
        merged["const"] = consts[0] if len(consts) == 1 else consts
    return merged


def _enum_values(node: dict[str, Any]) -> set[Any] | None:
    if "enum" in node:
        return set(node["enum"])
    if "const" in node:
        return {node["const"]}
    return None


def _compare(
    c_node: Any,
    m_node: Any,
    path: str,
    c_root: dict[str, Any],
    m_root: dict[str, Any],
    diffs: list[str],
) -> None:
    if not isinstance(c_node, dict) or not isinstance(m_node, dict):
        if c_node != m_node:
            diffs.append(f"{path}: 契约 {c_node!r} != 导出 {m_node!r}")
        return

    c = _unwrap(_deref(c_node, c_root), c_root)
    m = _unwrap(_deref(m_node, m_root), m_root)

    c_enum = _enum_values(c)
    m_enum = _enum_values(m)
    if c_enum is not None or m_enum is not None:
        if c_enum != m_enum:
            diffs.append(f"{path}: 枚举 {sorted(map(str, c_enum or []))} != {sorted(map(str, m_enum or []))}")
    elif (_type_set(c) or _type_set(m)) and _type_set(c) != _type_set(m):
        diffs.append(f"{path}: 类型 {_type_set(c)} != {_type_set(m)}")

    for attr in _VALUE_ATTRS:
        in_c, in_m = attr in c, attr in m
        if in_c != in_m:
            diffs.append(f"{path}: 约束 {attr} 契约{'有' if in_c else '无'}导出{'有' if in_m else '无'}")
        elif in_c and c[attr] != m[attr]:
            diffs.append(f"{path}: 约束 {attr} {c[attr]!r} != {m[attr]!r}")

    c_props = c.get("properties", {})
    m_props = m.get("properties", {})
    if c_props or m_props or "required" in c or "required" in m:
        if set(c_props) != set(m_props):
            only_c = sorted(set(c_props) - set(m_props))
            only_m = sorted(set(m_props) - set(c_props))
            diffs.append(f"{path}: 字段集不一致 契约独有={only_c} 导出独有={only_m}")
        c_req = set(c.get("required", []))
        m_req = set(m.get("required", []))
        if c_req != m_req:
            diffs.append(f"{path}: required 不一致 契约={sorted(c_req)} 导出={sorted(m_req)}")
        for prop in c_props:
            if prop in m_props:
                _compare(c_props[prop], m_props[prop], f"{path}.{prop}", c_root, m_root, diffs)

    c_pattern = c.get("patternProperties")
    c_extra = c.get("additionalProperties")
    m_extra = m.get("additionalProperties")
    if c_pattern:
        m_pattern = m.get("patternProperties")
        if isinstance(m_pattern, dict) and set(m_pattern) == set(c_pattern):
            for key, pattern_value in c_pattern.items():
                _compare(pattern_value, m_pattern[key], f"{path}[pattern:{key}]", c_root, m_root, diffs)
        elif isinstance(m_extra, dict):
            # 导出把 map 值约束放在 additionalProperties 上（如 pydantic dict 字段）
            for i, pattern_value in enumerate(c_pattern.values()):
                _compare(pattern_value, m_extra, f"{path}[pattern:{i}]", c_root, m_root, diffs)
        else:
            diffs.append(f"{path}: 契约 patternProperties 在导出中缺少对应约束")
        return
    if c_extra is False and m_extra is not False:
        diffs.append(f"{path}: 契约 additionalProperties=false，导出未禁止额外字段")
    if c_extra is None and isinstance(m_extra, dict):
        diffs.append(f"{path}: 导出存在契约没有的 additionalProperties 约束")
    if isinstance(c_extra, dict) and isinstance(m_extra, dict):
        _compare(c_extra, m_extra, f"{path}[extra]", c_root, m_root, diffs)

    if "items" in c:
        if "items" not in m:
            diffs.append(f"{path}: 契约有 items，导出缺失")
        else:
            _compare(c["items"], m["items"], f"{path}[]", c_root, m_root, diffs)
    elif "items" in m:
        diffs.append(f"{path}: 导出有 items，契约没有")
