"""Dataset Contract Layer 语义 validator（S-2）。

职责边界：JSON Schema（contracts/annotation-schema.json /
dataset-metadata.schema.json）管**结构**；本模块管**语义**——
JSON Schema 表达不了的跨字段/跨记录约束（DCL-003 A≠B、DCL-004
agreement 与原始标签一致、DCL-006 UNKNOWN→HUMAN_TRIAGE 之外的
重算类约束、metadata 计数一致性与 split 可行性）。

三层校验面：
  - validate_annotation_pair：单条标注记录（记录内自洽）；
  - validate_metadata：metadata 内部自洽（数字彼此一致、台账内集合一致）；
  - validate_dataset_bundle：bundle 级硬门（collector 前最后一道）——
    从原始 annotations/split_assignment/provenance **重算**派生统计并与
    metadata 缓存逐项比对（外部审计 P0 裁定：内部自洽 ≠ 与真实记录一致）。

全部为纯函数：输入 dict，输出违规列表（空列表 = 通过），不做 IO、
不维护状态，可直接被采集管道与 pytest 消费（tests/test_dataset_contracts.py）。

约定：validate_* 函数不替代 schema 校验——调用方应先用
Draft202012Validator 校验结构，再用本模块校验语义。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import FormatChecker

# sample_id 公式的字段分隔符（DCL-002；与 annotation-schema.json description 一致）
SAMPLE_ID_SEPARATOR = "\n"

# ---------------------------------------------------------------------------
# Dataset Contract 版本绑定（外部审计 P0 裁定：metadata 自称的版本号必须
# 等于当前冻结值，禁止生产者手填任何"自洽"的旧/错版本号通过语义门）。
# 未来升版只改本常量，全局（metadata 三字段 + annotation 记录两字段）生效。
# ---------------------------------------------------------------------------
DATASET_CONTRACT_VERSIONS: dict[str, str] = {
    "annotation_schema_version": "2",
    "guideline_version": "2",
    "split_policy_version": "1.1",
}

# ---------------------------------------------------------------------------
# source_type ↔ source_tier 绑定表（DCL-001 语义：source_type 决定 tier，
# 禁止 provenance 台账中出现 "public_ci 但 P0" 之类的错配）。未知
# source_type 一律报错——新来源类型必须先扩充本表再入库。
# 可按扩展流程新增条目（模块常量，改表即全局生效）。
# ---------------------------------------------------------------------------
SOURCE_TYPE_TIER: dict[str, str] = {
    "enterprise_log": "P0",
    "public_ci": "P1",
    "oss_issue": "P2",
    "jenkins_log": "P1",
    "github_actions_log": "P1",
    "github_issue": "P2",
}

# category 枚举中 UNKNOWN 的 action 硬约束（DCL-006；schema if/then 之外的语义兜底）
UNKNOWN = "UNKNOWN"
HUMAN_TRIAGE = "HUMAN_TRIAGE"

# ---------------------------------------------------------------------------
# Q2 bundle 硬门（P0-1 修复：Q2 从"采集时检查"升级为"bundle 硬门"）。
# 过密 artifact（关联样本数 >10）的**每个样本**，其 lineage latest
# disposition 必须是 approve 才放行（外审六件收口 #1：从 artifact 级
# 放行集合升级为 sample 级 latest disposition 核对）。
# ---------------------------------------------------------------------------
Q2_SAMPLE_COUNT_THRESHOLD = 10

# lineage disposition 的合法取值（quarantine_reviewed.detail.disposition 闭集）
LINEAGE_DISPOSITIONS = ("approve", "reject", "pending")

# ---------------------------------------------------------------------------
# Q2 review policy v2（contracts/q2-review-policy-v2.yaml）机器化：
#   - batch_purpose 二分类（缺省 = effectiveness_dataset，从严）；
#   - effectiveness_dataset 批次存在 repeat_signature=true 样本 → 硬报错；
#   - annotation_protocol_pilot 批次允许 controlled repeat，但
#     metadata.signature_stats 必须与逐行 repeat_signature 重算一致。
# ---------------------------------------------------------------------------
BATCH_PURPOSES = ("annotation_protocol_pilot", "effectiveness_dataset")
DEFAULT_BATCH_PURPOSE = "effectiveness_dataset"

_CATEGORIES = ("PRODUCT_DEFECT", "TEST_SCRIPT", "ENVIRONMENT", "CONFIGURATION", "FLAKY", "UNKNOWN")
_SPLITS = ("calibration", "holdout")
_TIERS = ("P0", "P1", "P2")
_RELEVANCES = ("in_domain", "adjacent_domain", "out_of_domain", "unknown")


def _canonicalize_event_time(event_time: str) -> str | None:
    """event_time 规范化（sample_id 公式的输入预处理）。

    流程：解析 ISO8601（容忍尾缀 Z/z）→ 无时区视为 UTC（显式化）→
    转 UTC → 输出 RFC3339 Z 后缀、毫秒固定 3 位。同一时刻无论以
    "+08:00" 还是 "Z" 记录，规范化结果相同 → sample_id 相同。

    返回 None 表示无法解析（调用方报错；结构层 format 校验先行）。
    """
    text = event_time.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    millis = parsed.microsecond // 1000
    base = parsed.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{millis:03d}Z"


def compute_sample_id(
    artifact_id: str, case_id: str, event_time: str, occurrence_index: int
) -> str:
    """DCL-002：一次具体 failure event 的身份。

    sample_id = SHA256(artifact_id + "\\n" + case_id + "\\n" +
                       canonicalize(event_time) + "\\n" + occurrence_index)

    event_time 先经 _canonicalize_event_time 规范化（统一转 UTC、
    RFC3339 Z 后缀、毫秒固定 3 位）再参与 hash——同一时刻的不同时区
    记录法必须产生同一 sample_id（timezone canonicalization）。无法
    解析的 event_time 原样参与拼接（结构层 format 校验会先拦截，
    此处不做隐式修正）。

    occurrence_index 语义：该 artifact 内同 case 第几次失败（从 0 起），
    **由 raw artifact 中的稳定顺序产生**（如日志出现顺序 + case 首次
    出现序），不由 parser 遍历顺序产生——parser 实现更换不得改变
    occurrence_index，否则 sample_id 漂移。以十进制字符串参与拼接。
    返回小写 hex。

    兼容性：与旧公式（原始 event_time 直接拼接）**不兼容**——DCL 尚未
    有真实数据，直接改公式；annotation-schema.json 的 description 已同步。
    """
    canonical_time = _canonicalize_event_time(event_time)
    if canonical_time is None:
        canonical_time = event_time
    payload = SAMPLE_ID_SEPARATOR.join(
        (artifact_id, case_id, canonical_time, str(occurrence_index))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_iso8601(value: str) -> bool:
    """date-time FormatChecker 后备实现（rfc3339-validator 缺席时启用）。

    接受 ISO8601（含尾缀 Z）；非字符串交回给 schema 的 type 约束处理。
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def build_format_checker() -> FormatChecker:
    """返回启用了 date-time 检查的 FormatChecker。

    rfc3339-validator 已安装时用其官方实现；否则注册本模块的 ISO8601
    后备实现——保证 "format": "date-time" 在本仓库内永远被实际校验
    （而不是 jsonschema 在无 checker 时静默跳过）。
    """
    checker = FormatChecker()
    if "date-time" not in checker.checkers:
        checker.checks("date-time")(_is_iso8601)
    return checker


FORMAT_CHECKER = build_format_checker()


def _final_category(record: dict[str, Any]) -> str | None:
    """样本的最终 category：有 adjudication 用 final_category，否则用 A 值
    （A/B 完全一致时二者相同；分歧必仲裁，因此不存在"分歧且无仲裁"的合法样本）。"""
    adjudication = record.get("adjudication")
    if isinstance(adjudication, dict) and "final_category" in adjudication:
        return adjudication["final_category"]
    pair = record.get("annotation_pair")
    if isinstance(pair, dict):
        annotator_a = pair.get("annotator_a")
        if isinstance(annotator_a, dict):
            return annotator_a.get("category")
    return None


def validate_annotation_pair(record: dict[str, Any]) -> list[str]:
    """单条标注记录的语义校验（结构由 annotation-schema.json 先行校验）。

    检查项：
      1. sample_id 与公式重算一致（DCL-002：禁止手填不一致的主键）；
      2. annotator_a.annotator_id != annotator_b.annotator_id（DCL-003）；
      3. 有 adjudication 时 adjudicator_id 不等于 A/B 任一 id（DCL-003）；
      4. agreement 派生缓存与 A/B 原始标签重算一致（DCL-004）；
      5. UNKNOWN → HUMAN_TRIAGE（A/B 与 adjudication 三处，DCL-006 语义兜底）；
      6. 记录级版本绑定：annotation_schema_version / guideline_version 必须等于
         当前冻结值（DATASET_CONTRACT_VERSIONS），错值报错。
    """
    errors: list[str] = []

    pair = record.get("annotation_pair")
    annotator_a = pair.get("annotator_a") if isinstance(pair, dict) else None
    annotator_b = pair.get("annotator_b") if isinstance(pair, dict) else None
    if not isinstance(annotator_a, dict) or not isinstance(annotator_b, dict):
        return ["annotation_pair.annotator_a/annotator_b 缺失或非对象，无法做语义校验"]

    # 6. 记录级版本绑定（与 metadata 三字段同表；bundle 校验同样核对）
    errors.extend(_check_record_version_binding(record))

    # 1. sample_id 公式重算（DCL-002）
    needed = ("sample_id", "artifact_id", "case_id", "event_time", "occurrence_index")
    missing = [k for k in needed if k not in record]
    if missing:
        errors.append(f"sample_id 重算缺字段: {missing}")
    else:
        expected = compute_sample_id(
            record["artifact_id"], record["case_id"],
            record["event_time"], record["occurrence_index"],
        )
        if record["sample_id"] != expected:
            errors.append(
                f"sample_id 与公式重算不一致: 记录={record['sample_id']} 期望={expected}"
            )

    # 2. A != B（DCL-003）
    id_a = annotator_a.get("annotator_id")
    id_b = annotator_b.get("annotator_id")
    if not id_a:
        errors.append("annotator_a.annotator_id 缺失（DCL-003 必填）")
    if not id_b:
        errors.append("annotator_b.annotator_id 缺失（DCL-003 必填）")
    if id_a and id_b and id_a == id_b:
        errors.append(
            f"annotator_a.annotator_id == annotator_b.annotator_id ({id_a!r})："
            "双标必须是不同的人（DCL-003）"
        )

    adjudication = record.get("adjudication")
    if isinstance(adjudication, dict):
        # 3. adjudicator 独立于 A/B（DCL-003）
        adjudicator_id = adjudication.get("adjudicator_id")
        if not adjudicator_id:
            errors.append("adjudication.adjudicator_id 缺失")
        else:
            for role, value in (("annotator_a", id_a), ("annotator_b", id_b)):
                if value and adjudicator_id == value:
                    errors.append(
                        f"adjudication.adjudicator_id ({adjudicator_id!r}) "
                        f"== {role}.annotator_id：仲裁人必须独立于双标双方（DCL-003）"
                    )

        # 5b. 仲裁结果的 UNKNOWN → HUMAN_TRIAGE（DCL-006）
        if adjudication.get("final_category") == UNKNOWN and \
                adjudication.get("final_action") != HUMAN_TRIAGE:
            errors.append(
                "adjudication: final_category == UNKNOWN 但 final_action != "
                "HUMAN_TRIAGE（DCL-006 硬约束）"
            )

    # 4. agreement 派生缓存重算核对（DCL-004）
    agreement = record.get("agreement")
    if not isinstance(agreement, dict):
        errors.append("agreement 缺失或非对象")
    else:
        expected_category_match = annotator_a.get("category") == annotator_b.get("category")
        expected_action_match = annotator_a.get("action") == annotator_b.get("action")
        pa, pb = annotator_a.get("priority"), annotator_b.get("priority")
        if isinstance(pa, int) and isinstance(pb, int):
            expected_delta = abs(pa - pb)
            if agreement.get("priority_delta") != expected_delta:
                errors.append(
                    f"agreement.priority_delta 与原始标签重算不一致: "
                    f"缓存={agreement.get('priority_delta')} 期望={expected_delta}（DCL-004）"
                )
        else:
            errors.append("annotator priority 缺失或非整数，priority_delta 无法重算")
        if agreement.get("category_match") != expected_category_match:
            errors.append(
                f"agreement.category_match 与原始标签重算不一致: "
                f"缓存={agreement.get('category_match')} 期望={expected_category_match}（DCL-004）"
            )
        if agreement.get("action_match") != expected_action_match:
            errors.append(
                f"agreement.action_match 与原始标签重算不一致: "
                f"缓存={agreement.get('action_match')} 期望={expected_action_match}（DCL-004）"
            )

    # 5a. 双标双方的 UNKNOWN → HUMAN_TRIAGE（DCL-006）
    for role, annotator in (("annotator_a", annotator_a), ("annotator_b", annotator_b)):
        if annotator.get("category") == UNKNOWN and annotator.get("action") != HUMAN_TRIAGE:
            errors.append(
                f"{role}: category == UNKNOWN 但 action != HUMAN_TRIAGE（DCL-006 硬约束）"
            )

    # 附带核对：counts 报告消费的最终 category 必须在冻结 6 枚举内
    final_category = _final_category(record)
    if final_category is not None and final_category not in _CATEGORIES:
        errors.append(f"最终 category {final_category!r} 不在冻结 6 枚举内")

    return errors


def _check_record_version_binding(record: dict[str, Any]) -> list[str]:
    """单条记录的 annotation_schema_version / guideline_version 冻结值绑定。"""
    errors: list[str] = []
    for field in ("annotation_schema_version", "guideline_version"):
        expected = DATASET_CONTRACT_VERSIONS[field]
        actual = record.get(field)
        if actual != expected:
            errors.append(
                f"{field} 版本绑定失败: 记录={actual!r} 当前冻结值={expected!r}"
                f"（版本由 DATASET_CONTRACT_VERSIONS 统一管理）"
            )
    return errors


def _sum_of(obj: Any, label: str, errors: list[str]) -> int | None:
    """dict[int] 求和；异常结构写入 errors 并返回 None。"""
    if not isinstance(obj, dict) or not obj:
        errors.append(f"{label} 缺失或为空，无法求和")
        return None
    values = [v for v in obj.values() if isinstance(v, int) and not isinstance(v, bool)]
    if len(values) != len(obj):
        errors.append(f"{label} 存在非整数值，无法求和")
        return None
    return sum(values)


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    """数据集级 metadata 的语义校验（结构由 dataset-metadata.schema.json 先行校验）。

    检查项（全部为"派生统计 vs 事实源重算"或可行性门）：
      1. sum(counts.per_class) == counts.total；
      2. sum(counts.domain_relevance) == counts.total；
      3. sum(split) == counts.total；
      4. sum(counts.groups_per_split) == counts.unique_source_groups
         （**仅 group 模式**；temporal 模式同 group 允许跨 split，允许重叠，
         sum == unique_source_groups 不成立——该核对由
         validate_dataset_bundle 以 union(groups) == unique_source_groups 替代）；
      5. split feasibility：holdout_basis == "group_level" 必须
         unique_source_groups >= 2（owner 指令 6）；
      6. counts.source_tier_x_domain_relevance（若提供）sum == total；
      7. 顶层 source_tier 数组 == provenance[].source_tier 去重集合；
      8. metadata 三字段版本绑定（annotation_schema_version /
         guideline_version / split_policy_version == DATASET_CONTRACT_VERSIONS）。

    注意：本函数只对 metadata **内部**自洽性做门（数字彼此一致、
    provenance 台账内的 tier 集合一致）；"派生统计 vs 真实
    annotations/split_assignment 重算"由 validate_dataset_bundle 承担
    （需要 annotations list 与 split_assignment 明细）。
    """
    errors: list[str] = []
    counts = metadata.get("counts")
    if not isinstance(counts, dict):
        return ["counts 缺失或非对象，无法做计数一致性校验"]

    total = counts.get("total")

    def _sum(obj: Any, label: str) -> int | None:
        return _sum_of(obj, label, errors)

    # 1. per_class（收口 #4：per_class=null 合法当且仅当 label_status=="pending"；
    #    禁止 UNKNOWN 占位伪装真实分布）
    per_class_raw = counts.get("per_class")
    if per_class_raw is None:
        if metadata.get("label_status") != "pending":
            errors.append(
                "counts.per_class=null 但 label_status != 'pending'：未完成标注时必须"
                "声明 label_status='pending'（禁止 UNKNOWN 占位伪装真实分布）"
            )
    else:
        per_class_sum = _sum(per_class_raw, "counts.per_class")
        if per_class_sum is not None and isinstance(total, int) and per_class_sum != total:
            errors.append(f"sum(counts.per_class)={per_class_sum} != counts.total={total}")

    # 2. domain_relevance
    dr_sum = _sum(counts.get("domain_relevance"), "counts.domain_relevance")
    if dr_sum is not None and isinstance(total, int) and dr_sum != total:
        errors.append(f"sum(counts.domain_relevance)={dr_sum} != counts.total={total}")

    # 3. split
    split_sum = _sum(metadata.get("split"), "split")
    if split_sum is not None and isinstance(total, int) and split_sum != total:
        errors.append(f"sum(split)={split_sum} != counts.total={total}")

    # 4. groups_per_split（仅 group 模式：temporal 允许同 group 跨 split，允许重叠）
    if metadata.get("split_mode") != "temporal":
        gps_sum = _sum(counts.get("groups_per_split"), "counts.groups_per_split")
        unique_groups = counts.get("unique_source_groups")
        if gps_sum is not None and isinstance(unique_groups, int) and gps_sum != unique_groups:
            errors.append(
                f"sum(counts.groups_per_split)={gps_sum} != "
                f"counts.unique_source_groups={unique_groups}"
            )

    # 5. split feasibility（owner 指令 6）
    unique_groups = counts.get("unique_source_groups")
    if metadata.get("holdout_basis") == "group_level" and \
            isinstance(unique_groups, int) and unique_groups < 2:
        errors.append(
            "holdout_basis == 'group_level' 但 counts.unique_source_groups < 2："
            "禁止声称 group-level holdout（split feasibility gate）"
        )

    # 6. 交叉分层（可选字段）
    cross = counts.get("source_tier_x_domain_relevance")
    if cross is not None:
        cross_sum = _sum(cross, "counts.source_tier_x_domain_relevance")
        if cross_sum is not None and isinstance(total, int) and cross_sum != total:
            errors.append(
                f"sum(counts.source_tier_x_domain_relevance)={cross_sum} != "
                f"counts.total={total}"
            )

    # 7. 顶层 source_tier == provenance[].source_tier 去重集合（DCL-001 事实源核对）
    provenance = metadata.get("provenance")
    if isinstance(provenance, list) and provenance:
        sample_tiers = {
            item.get("source_tier")
            for item in provenance
            if isinstance(item, dict) and item.get("source_tier") in _TIERS
        }
        declared = metadata.get("source_tier")
        if isinstance(declared, list) and sample_tiers != set(declared):
            errors.append(
                f"顶层 source_tier {sorted(declared)} != provenance[].source_tier "
                f"去重集合 {sorted(sample_tiers)}（事实源为 artifact 级属性，"
                "经 artifact_id join 投影到样本，DCL-001）"
            )

    # 8. 版本绑定（冻结值见 DATASET_CONTRACT_VERSIONS）
    for field, expected in DATASET_CONTRACT_VERSIONS.items():
        actual = metadata.get(field)
        if actual != expected:
            errors.append(
                f"{field} 版本绑定失败: metadata={actual!r} 当前冻结值={expected!r}"
                "（版本由 DATASET_CONTRACT_VERSIONS 统一管理）"
            )

    return errors


# ---------------------------------------------------------------------------
# bundle 级语义校验（外部审计 P0 裁定：metadata 内部自洽 ≠ 与真实记录一致）
# ---------------------------------------------------------------------------

def _recompute_per_class(annotations: list[dict[str, Any]]) -> dict[str, int]:
    """从标注记录重算最终 category 分布（仲裁值优先，无仲裁用一致双标值）。"""
    counts = dict.fromkeys(_CATEGORIES, 0)
    for record in annotations:
        category = _final_category(record)
        if category in counts:
            counts[category] += 1
    return counts


def _artifact_fact_index(
    provenance: list[dict[str, Any]] | None,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    """provenance 台账按 original_hash 建索引（artifact→tier/domain_relevance）。

    original_hash 重复视为台账缺陷（报错一次，取首条）。
    """
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(provenance, list):
        errors.append("provenance 缺失或非数组，artifact 级 fact join 无法进行")
        return index
    for i, item in enumerate(provenance):
        if not isinstance(item, dict):
            errors.append(f"provenance[{i}] 非对象，跳过")
            continue
        artifact_id = item.get("original_hash")
        if not artifact_id:
            errors.append(f"provenance[{i}] 缺 original_hash，无法与 annotations join")
            continue
        if artifact_id in index:
            errors.append(
                f"provenance 台账 original_hash 重复: {artifact_id[:16]}…"
                f"（provenance[{i}]）——台账去重以 original_hash 为准"
            )
            continue
        index[artifact_id] = item
    return index


def _recompute_sample_level_stats(
    annotations: list[dict[str, Any]],
    artifact_index: dict[str, dict[str, Any]],
    errors: list[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, int] | None]:
    """按 artifact 继承规则重算 sample 级 domain_relevance / tier 统计。

    继承规则：annotation.artifact_id → provenance.original_hash 找到台账
    条目，取其 source_tier / domain_relevance（artifact-level fact
    projected to samples）。artifact_id 在台账中找不到 → join 完整性错误。
    返回 (domain_relevance 计数, tier 计数, tier×relevance 交叉表)。
    """
    relevance: dict[str, int] = dict.fromkeys(_RELEVANCES, 0)
    tier_counts: dict[str, int] = dict.fromkeys(_TIERS, 0)
    cross: dict[str, int] = {}
    for i, record in enumerate(annotations):
        artifact_id = record.get("artifact_id") if isinstance(record, dict) else None
        entry = artifact_index.get(artifact_id)
        if entry is None:
            errors.append(
                f"annotations[{i}] 的 artifact_id={artifact_id!r} 在 provenance "
                "台账中找不到对应 original_hash：artifact join 完整性失败"
                "（无来源的样本不得进入正式集）"
            )
            continue
        rel = entry.get("domain_relevance")
        tier = entry.get("source_tier")
        if rel in relevance:
            relevance[rel] += 1
        if tier in tier_counts:
            tier_counts[tier] += 1
            key = f"{tier}|{rel}"
            cross[key] = cross.get(key, 0) + 1
    return relevance, tier_counts, cross


def _check_split_consistency(
    metadata: dict[str, Any],
    annotations: list[dict[str, Any]],
    errors: list[str],
) -> None:
    """split_assignment 与真实标注的核对（group / temporal 两模式分派）。"""
    split_mode = metadata.get("split_mode")
    assignments = metadata.get("split_assignment")
    if not isinstance(assignments, list) or not assignments:
        errors.append("split_assignment 缺失或为空，无法做分配明细核对")
        return

    split_counts: dict[str, int] = dict.fromkeys(_SPLITS, 0)
    groups_per_split: dict[str, set[str]] = {name: set() for name in _SPLITS}

    if split_mode == "temporal":
        # temporal：逐样本 assignment，样本级计数在 _check_temporal_split 内重算
        _check_temporal_split(
            metadata, annotations, assignments, split_counts, groups_per_split, errors
        )
    else:
        group_to_split = _check_group_split(assignments, groups_per_split, errors)
        # group 模式：组→split 投影回 annotations 得样本级 split 计数
        # （assignment 是组级条目，不能直接当样本数求和）
        for record in annotations:
            if not isinstance(record, dict):
                continue
            split = group_to_split.get(record.get("source_group_id"))
            if split is not None:
                split_counts[split] = split_counts.get(split, 0) + 1

    # split 样本数 vs metadata.split 声明
    declared_split = metadata.get("split")
    if isinstance(declared_split, dict):
        for name in _SPLITS:
            declared = declared_split.get(name)
            actual = split_counts.get(name, 0)
            if isinstance(declared, int) and declared != actual:
                errors.append(
                    f"metadata.split[{name}]={declared} != 按 split_assignment/"
                    f"annotations 重算 {actual}"
                )

    # groups_per_split vs metadata.counts.groups_per_split 声明
    counts = metadata.get("counts")
    declared_gps = counts.get("groups_per_split") if isinstance(counts, dict) else None
    if isinstance(declared_gps, dict):
        for name in _SPLITS:
            declared = declared_gps.get(name)
            actual = len(groups_per_split.get(name, ()))
            if isinstance(declared, int) and declared != actual:
                errors.append(
                    f"counts.groups_per_split[{name}]={declared} != 按 "
                    f"split_assignment 重算 {actual}"
                )


def _check_group_split(
    assignments: list[dict[str, Any]],
    groups_per_split: dict[str, set[str]],
    errors: list[str],
) -> dict[str, str]:
    """group 模式：同 source_group_id 跨 split = 硬违规；每组一条 assignment。

    返回 group→split 映射（供样本级 split 计数投影）。
    """
    group_to_split: dict[str, str] = {}
    for i, entry in enumerate(assignments):
        if not isinstance(entry, dict):
            errors.append(f"split_assignment[{i}] 非对象")
            continue
        group_id = entry.get("source_group_id")
        split = entry.get("split")
        if not group_id or split not in _SPLITS:
            errors.append(f"split_assignment[{i}] 缺 source_group_id 或 split 非法: {entry!r}")
            continue
        groups_per_split.setdefault(split, set()).add(group_id)
        previous = group_to_split.get(group_id)
        if previous is not None and previous != split:
            errors.append(
                f"group 模式下同一 source_group_id {group_id!r} 出现在两个 split"
                f"（{previous} 与 {split}）：同组禁跨 split（split_assignment[{i}]）"
            )
        elif previous is None:
            group_to_split[group_id] = split
    return group_to_split


def _check_temporal_split(
    metadata: dict[str, Any],
    annotations: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    split_counts: dict[str, int],
    groups_per_split: dict[str, set[str]],
    errors: list[str],
) -> None:
    """temporal 模式：逐样本 assignment，按 event_time/cutoff 重算核对。

    允许同 group 跨 split（不报错）；union(groups across splits) 必须等于
    unique_source_groups；cutoff 之后 → holdout。
    """
    cutoff_raw = metadata.get("temporal_cutoff")
    if not isinstance(cutoff_raw, str):
        errors.append("temporal 模式缺 temporal_cutoff，无法按 event_time 重算分配")
        return
    cutoff_canonical = _canonicalize_cutoff_text(cutoff_raw)
    if cutoff_canonical == cutoff_raw and not _is_iso8601(cutoff_raw):
        errors.append(f"temporal_cutoff 无法解析: {cutoff_raw!r}")
        return

    # assignment 按 sample_id 索引
    assignment_by_sample: dict[str, dict[str, Any]] = {}
    for i, entry in enumerate(assignments):
        if not isinstance(entry, dict):
            errors.append(f"split_assignment[{i}] 非对象")
            continue
        sample_id = entry.get("sample_id")
        split = entry.get("split")
        group_id = entry.get("source_group_id")
        if not sample_id:
            errors.append(
                f"temporal 模式 split_assignment[{i}] 缺 sample_id：逐样本分配"
                "必须逐样本可核对（schema 修订已将 sample_id 列为必填）"
            )
            continue
        if split not in _SPLITS:
            errors.append(f"split_assignment[{i}] split 非法: {split!r}")
            continue
        if not group_id:
            errors.append(
                f"temporal 模式 split_assignment[{i}] 缺 source_group_id："
                "union(groups across splits)==unique_source_groups 核对无法进行"
            )
            continue
        assignment_by_sample[sample_id] = entry

    for record in annotations:
        if not isinstance(record, dict):
            continue
        sample_id = record.get("sample_id")
        event_time_raw = record.get("event_time")
        group_id = record.get("source_group_id")
        # 期望 split：event_time <= cutoff → calibration，否则 holdout
        canonical = _canonicalize_event_time(event_time_raw) if isinstance(event_time_raw, str) else None
        if canonical is None:
            errors.append(f"sample {sample_id!r} 的 event_time 无法解析: {event_time_raw!r}")
            continue
        expected_split = "calibration" if canonical <= cutoff_canonical else "holdout"
        split_counts[expected_split] = split_counts.get(expected_split, 0) + 1
        if group_id:
            groups_per_split.setdefault(expected_split, set()).add(group_id)
        entry = assignment_by_sample.get(sample_id)
        if entry is None:
            errors.append(
                f"annotations 中的 sample {sample_id[:16]}… 在 split_assignment 中"
                "无分配条目：temporal 模式必须逐样本分配"
            )
        else:
            if entry.get("split") != expected_split:
                errors.append(
                    f"sample {sample_id[:16]}… split 与 event_time/cutoff 重算不一致: "
                    f"assignment={entry.get('split')!r} 期望={expected_split!r} "
                    f"(event_time={event_time_raw!r}, cutoff={cutoff_raw!r})"
                )
            # 逐样本组一致性：assignment 声明的组必须与标注记录的组相同
            # （防止凭空多组使 union(groups)==unique_source_groups 失真）
            if group_id and entry.get("source_group_id") != group_id:
                errors.append(
                    f"sample {sample_id[:16]}… 的 source_group_id 在 split_assignment"
                    f"（{entry.get('source_group_id')!r}）与 annotations"
                    f"（{group_id!r}）不一致"
                )

    # union(groups across splits) == unique_source_groups
    counts = metadata.get("counts")
    unique_declared = counts.get("unique_source_groups") if isinstance(counts, dict) else None
    union_groups: set[str] = set()
    for members in groups_per_split.values():
        union_groups |= members
    if isinstance(unique_declared, int) and (union_groups or unique_declared > 0) and \
            len(union_groups) != unique_declared:
        errors.append(
            f"temporal 模式 union(groups across splits)={len(union_groups)} "
            f"!= counts.unique_source_groups={unique_declared}"
            f"（允许同组跨 split 重叠，但并集必须等于去重总数）"
        )


def _canonicalize_cutoff_text(cutoff_raw: str) -> str:
    """temporal_cutoff 规范化为与 _canonicalize_event_time 相同的格式，供字符串比较。"""
    canonical = _canonicalize_event_time(cutoff_raw)
    return canonical if canonical is not None else cutoff_raw


def _check_source_type_tier_binding(
    provenance: list[dict[str, Any]] | None,
    errors: list[str],
) -> None:
    """provenance 每条 source_type 必须在 SOURCE_TYPE_TIER 中且 tier 一致。"""
    if not isinstance(provenance, list):
        return
    for i, item in enumerate(provenance):
        if not isinstance(item, dict):
            continue
        source_type = item.get("source_type")
        tier = item.get("source_tier")
        expected_tier = SOURCE_TYPE_TIER.get(source_type)
        if expected_tier is None:
            errors.append(
                f"provenance[{i}] 的 source_type={source_type!r} 不在 "
                f"SOURCE_TYPE_TIER 映射表中：新来源类型必须先扩充映射表再入库"
            )
            continue
        if tier != expected_tier:
            errors.append(
                f"provenance[{i}] source_type↔source_tier 错配: "
                f"source_type={source_type!r} 要求 tier={expected_tier!r}，"
                f"实际 tier={tier!r}"
            )


def _check_temporal_cutoff_compliance(
    metadata: dict[str, Any],
    annotations: list[dict[str, Any]],
    errors: list[str],
) -> None:
    """cutoff 合规独立核对：assignment 中 cutoff 之后的样本不得标 calibration。

    与 _check_temporal_split 的重算核对互补：那里以 annotations 为基准找
    缺失/错配条目，这里以 split_assignment 为基准直接核验 cutoff 关系
    （覆盖 assignment 中存在但 annotations 里缺失 sample_id 的残缺情况）。
    """
    if metadata.get("split_mode") != "temporal":
        return
    cutoff_raw = metadata.get("temporal_cutoff")
    if not isinstance(cutoff_raw, str):
        return
    cutoff_canonical = _canonicalize_cutoff_text(cutoff_raw)
    event_time_by_sample = {
        r.get("sample_id"): r.get("event_time")
        for r in annotations if isinstance(r, dict)
    }
    for i, entry in enumerate(metadata.get("split_assignment") or []):
        if not isinstance(entry, dict):
            continue
        if entry.get("split") != "calibration":
            continue
        sample_id = entry.get("sample_id")
        event_time = event_time_by_sample.get(sample_id)
        if not isinstance(event_time, str):
            continue
        canonical = _canonicalize_event_time(event_time)
        if canonical is not None and canonical > cutoff_canonical:
            errors.append(
                f"temporal_cutoff 合规: sample {str(sample_id)[:16]}… event_time="
                f"{event_time!r} 晚于 cutoff={cutoff_raw!r} "
                "但被标 calibration（cutoff 之后必须为 holdout）"
            )


def _check_batch_purpose_and_signature_stats(
    metadata: dict[str, Any],
    annotations: list[dict[str, Any]],
    errors: list[str],
) -> None:
    """Q2 review policy v2 机器化（contracts/q2-review-policy-v2.yaml）。

    (a) batch_purpose 闭集核对：声明值必须在 BATCH_PURPOSES 内；字段缺席
        视为 effectiveness_dataset（从严，policy v2 §0）。
    (b) repeat 行核对：逐行 repeat_signature=true 必须携带 signature_id
        （v2 PR1）；repeat_signature=false 不得携带 signature_id。
    (c) effectiveness_dataset 硬门（v2 核心）：声明（或缺省视为）
        effectiveness_dataset 的批次中，任何样本 repeat_signature=true
        → 硬报错——效果/准确率/泛化结论要求样本独立，signature 重复
        样本禁止进入此类批次。
    (d) signature_stats 一致性（v2 PR2）：metadata 声明的
        signature_stats 与逐行 repeat_signature/signature_id 重算一致
        （total_samples==counts.total、repeat_signature_count、
        unique_signature_count；不一致报一条含两组数字的错误）。
    """
    batch_purpose = metadata.get("batch_purpose", DEFAULT_BATCH_PURPOSE)
    if batch_purpose not in BATCH_PURPOSES:
        errors.append(
            f"batch_purpose={batch_purpose!r} 不在闭集 {list(BATCH_PURPOSES)} 内"
            "（contracts/q2-review-policy-v2.yaml §0）"
        )
        return

    repeat_rows: list[tuple[int, dict[str, Any]]] = []
    malformed: list[str] = []
    for i, record in enumerate(annotations):
        if not isinstance(record, dict):
            continue
        if "repeat_signature" not in record:
            continue
        flag = record.get("repeat_signature")
        if not isinstance(flag, bool):
            malformed.append(
                f"annotations[{i}] repeat_signature 非布尔: {flag!r}"
            )
            continue
        if flag:
            if not record.get("signature_id"):
                malformed.append(
                    f"annotations[{i}] repeat_signature=true 但缺 signature_id"
                    "（q2-review-policy-v2 PR1）"
                )
            repeat_rows.append((i, record))
        elif "signature_id" in record:
            malformed.append(
                f"annotations[{i}] repeat_signature=false 却携带 signature_id"
                "（q2-review-policy-v2 PR1：仅 repeat 样本携带）"
            )
    errors.extend(malformed)

    if batch_purpose == "effectiveness_dataset" and repeat_rows:
        detail = "; ".join(
            f"annotations[{i}] {str(r.get('sample_id'))[:12]}…"
            f"(signature_id={r.get('signature_id')!r})"
            for i, r in repeat_rows[:5]
        )
        more = f" …等共 {len(repeat_rows)} 条" if len(repeat_rows) > 5 else ""
        errors.append(
            "q2-review-policy-v2 硬门: batch_purpose='effectiveness_dataset' "
            f"但存在 {len(repeat_rows)} 条 repeat_signature=true 样本"
            f"（effectiveness/accuracy/generalization 结论要求 root-cause "
            f"signature strict unique，v1 R1 维持）: {detail}{more}——"
            "此类样本只能进入 annotation_protocol_pilot 批次"
        )

    # (d) signature_stats 与逐行重算一致（声明了才核对）
    declared = metadata.get("signature_stats")
    if not isinstance(declared, dict):
        return
    recomputed_repeat = len(repeat_rows)
    # unique_signature_count 重算口径（q2-review-policy-v2 PR2，与 v2 §3
    # Pilot-0 打标语义一致）：
    #   - repeat 行：按 signature_id 去重分组，每组贡献 1 个 unique
    #     signature（组内首见成员 repeat_signature=false 且不携带
    #     signature_id——首见已由 lineage Q2 重放固定，不在行上重复登记）；
    #   - 首见行 + 未打标行：各自贡献 1 个 unique signature（打标范围
    #     只圈定 Q2 过密 artifact 的 repeat 成员，其余样本按 v1 R1 未
    #     参与去重 = 各自独立 root cause）。
    # 即 unique = total - repeat（首见行是每组的第 1 个成员，计在其
    # signature 组头内）。对账恒等式（PR2）：unique + repeat == total。
    unlabeled_or_first_seen = len(annotations) - recomputed_repeat
    unique_signatures = unlabeled_or_first_seen

    checks: list[tuple[str, Any, Any]] = []
    total = metadata.get("counts", {}).get("total")
    if isinstance(declared.get("total_samples"), int) and isinstance(total, int):
        checks.append(("total_samples", declared["total_samples"], total))
    if isinstance(declared.get("repeat_signature_count"), int):
        checks.append(
            ("repeat_signature_count", declared["repeat_signature_count"], recomputed_repeat)
        )
    if isinstance(declared.get("unique_signature_count"), int):
        checks.append(
            ("unique_signature_count", declared["unique_signature_count"], unique_signatures)
        )
    for name, dv, rv in checks:
        if dv != rv:
            errors.append(
                f"metadata.signature_stats.{name}={dv} != 逐行 repeat_signature/"
                f"signature_id 重算 {rv}（q2-review-policy-v2 PR2）"
            )
    # 对账恒等式：unique + repeat == total（仅当声明了全部三项）
    if all(isinstance(declared.get(k), int) for k in
           ("total_samples", "unique_signature_count", "repeat_signature_count")):
        dv_sum = declared["unique_signature_count"] + declared["repeat_signature_count"]
        if dv_sum != declared["total_samples"]:
            errors.append(
                f"metadata.signature_stats: unique({declared['unique_signature_count']})"
                f" + repeat({declared['repeat_signature_count']}) != "
                f"total({declared['total_samples']})（对账恒等式）"
            )


def _check_q2_quarantine_review(
    annotations: list[dict[str, Any]],
    provenance: list[dict[str, Any]] | None,
    lineage_dispositions: dict[str, str] | None = None,
    require_lineage: bool = False,
    errors: list[str] | None = None,
) -> list[str]:
    """Q2 bundle 硬门（P0-1 修复 + 外审六件收口 #1：sample 级 latest disposition）。

    检查：任一 artifact 在本 bundle 中关联的样本数 > Q2_SAMPLE_COUNT_THRESHOLD
    （10）时，该 artifact 的**每个样本**在 lineage_dispositions 中的
    latest disposition 必须是 "approve" 才放行；reject / pending / 缺失
    一律报错（错误信息列出违规 sample_id 前缀）——把 admission-policy
    Q2 从"采集时检查"升级为"数据集入口的 sample 级防线"，artifact 级
    放行集合不再足够（同 artifact 内允许部分样本被复核拒绝）。

    lineage_dispositions：sample_id → 该样本 lineage 中 latest disposition
    （approve/reject/pending；由 build_lineage_dispositions 构造）。
    为 None 时：require_lineage=True → 直接报错（official validation
    禁止 lineage=None）；False → 跳过 lineage 视角核对（无 lineage 可供
    的调用方降级路径；正式 bundle 校验必须传 True + 实际 dispositions）。
    """
    own_errors: list[str] = errors if errors is not None else []
    if not isinstance(provenance, list):
        provenance = []
    if not isinstance(annotations, list):
        return own_errors
    if lineage_dispositions is None:
        if require_lineage:
            own_errors.append(
                "Q2 bundle 硬门: official validation 禁止 lineage=None——"
                "必须提供 lineage_dispositions（build_lineage_dispositions 构造）"
                "以核对 Q2 过密 artifact 各样本的 latest disposition"
            )
        return own_errors
    samples_per_artifact: dict[str, list[str]] = {}
    for record in annotations:
        if not isinstance(record, dict):
            continue
        artifact_id = record.get("artifact_id")
        sample_id = record.get("sample_id")
        if isinstance(artifact_id, str) and artifact_id and isinstance(sample_id, str):
            samples_per_artifact.setdefault(artifact_id, []).append(sample_id)
    for artifact_id, sample_ids in sorted(samples_per_artifact.items()):
        if len(sample_ids) <= Q2_SAMPLE_COUNT_THRESHOLD:
            continue
        unapproved = [
            sid for sid in sample_ids if lineage_dispositions.get(sid) != "approve"
        ]
        if unapproved:
            detail: list[str] = []
            for sid in sorted(set(unapproved)):
                detail.append(f"{sid[:12]}…(disposition={lineage_dispositions.get(sid)!r})")
            own_errors.append(
                f"Q2 bundle 硬门: artifact {artifact_id[:16]}… 关联样本数 "
                f"{len(sample_ids)} > {Q2_SAMPLE_COUNT_THRESHOLD}（dataset 级累计），"
                f"但以下样本的 lineage latest disposition 非 approve："
                f"{'; '.join(detail)}——过密 artifact 的每个样本必须经人工复核"
                "放行（admission-policy Q2 状态机 + q2-review-policy v1）"
            )
    return own_errors


def build_lineage_dispositions(lineage_path: str | Path) -> dict[str, str]:
    """从 lineage.jsonl 构造 sample_id → latest disposition 映射。

    语义（外审六件收口 #1）：
      - quarantine_reviewed 事件：取 detail.disposition（approve/reject）——
        同 sample 多次复核时**后者覆盖前者**（latest wins）；
      - sample_admitted 事件：直接视为 approve（admission 放行语义）；
      - sample_quarantined 事件：视为 pending（待复核）；
      - 事件按落盘顺序回放，终值为 latest：同 sample 先 approve 后 reject
        → reject；先 admit 后 sample_quarantined（post-hoc 重分类）→ pending；
      - 非 Q2 的 admitted 样本自然为 approve；Q2 未复核算 pending。

    返回仅包含出现过上述事件的 sample（其余样本不在映射中，Q2 硬门
    视为缺失 → 报错）。
    """
    dispositions: dict[str, str] = {}
    path = Path(lineage_path)
    if not path.exists():
        return dispositions
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        sample_id = event.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        event_type = event.get("event_type")
        if event_type == "quarantine_reviewed":
            disposition = (event.get("detail") or {}).get("disposition")
            if disposition in ("approve", "reject"):
                dispositions[sample_id] = disposition
            elif disposition == "pending":
                dispositions[sample_id] = "pending"
        elif event_type == "sample_admitted":
            dispositions[sample_id] = "approve"
        elif event_type == "sample_quarantined":
            dispositions[sample_id] = "pending"
    return dispositions


def validate_pilot_bundle(
    pilot_dir: str | Path,
    require_blind_package: bool = True,
) -> list[str]:
    """Pilot-0 bundle 的唯一正式校验入口（外审六件收口 #1，fail-closed）。

    读 pilot_dir 下：
      - metadata.json（必需）
      - admitted-failures.jsonl（必需：官方集行格式，同 failures.jsonl 行）
      - blind-annotation-package.jsonl（pre-annotation 阶段必需：标注视图，
        与官方集 sample_id 集合核对；require_blind_package=False 时跳过——
        双标完成后（labels 阶段）blind package 的角色由 annotations.jsonl
        接管，该参数即该生命周期的显式表达。本阶段（pre-annotation）默认
        True 强制。）
      - annotations.jsonl（可不存在 = 空 list：Pilot-0 双标未完成阶段，
        annotations 侧检查退化为空集检查）
      - lineage.jsonl（必需：构造 sample 级 latest dispositions）
      - contracts/split-policy.yaml（仓库 contracts/ 目录，版本 binding 用）

    组装后调 validate_dataset_bundle(require_lineage=True)。任何文件缺失 /
    JSON 损坏 / 校验违规都以非空错误列表返回（fail-closed）——本函数
    不抛文件异常，统一转错误列表，便于调用方与 pytest 断言。
    """
    errors: list[str] = []
    root = Path(pilot_dir)
    if not root.is_dir():
        return [f"pilot_dir 不存在或非目录: {root}"]

    def _load_jsonl(name: str, required: bool) -> list[dict[str, Any]] | None:
        path = root / name
        if not path.exists():
            if required:
                errors.append(f"{name} 缺失（validate_pilot_bundle 必需文件）")
            return None
        rows: list[dict[str, Any]] = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{name} 第 {i + 1} 行 JSON 解析失败: {exc}")
        return rows

    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        errors.append("metadata.json 缺失（validate_pilot_bundle 必需文件）")
        return errors
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return errors + [f"metadata.json JSON 解析失败: {exc}"]

    admitted = _load_jsonl("admitted-failures.jsonl", required=True)
    if admitted is None:
        return errors
    annotations = _load_jsonl("annotations.jsonl", required=False) or []

    # blind package：pre-annotation 阶段必需（本阶段标注还没有发生，
    # blind package 是唯一的标注视图；缺失即 bundle 不完整）。双标完成
    # （labels 阶段）后其角色由 annotations.jsonl 接管——调用方以
    # require_blind_package=False 声明进入 labels 阶段（生命周期参数，
    # 不做阶段自动猜测：显式优于隐式）。
    blind = _load_jsonl("blind-annotation-package.jsonl",
                        required=require_blind_package)
    if blind is not None:
        admitted_ids = {
            r.get("sample_id") for r in admitted if isinstance(r, dict)
        }
        blind_ids = {
            r.get("sample_id") for r in blind if isinstance(r, dict)
        }
        if admitted_ids != blind_ids:
            only_admitted = sorted(str(s)[:12] for s in (admitted_ids - blind_ids) if s)
            only_blind = sorted(str(s)[:12] for s in (blind_ids - admitted_ids) if s)
            errors.append(
                "blind-annotation-package.jsonl 与 admitted-failures.jsonl 的 "
                "sample_id 集合不一致: 仅官方集有="
                f"{len(admitted_ids - blind_ids)}{only_admitted} "
                f"仅 blind 有={len(blind_ids - admitted_ids)}{only_blind}"
                "——标注视图必须与官方集同 sample_id 集合（少=漏装/多=旧版"
                "残留或覆盖，均不得开标）"
            )

    contracts_dir = Path(__file__).resolve().parents[2] / "contracts"
    split_policy: dict[str, Any] | None = None
    policy_path = contracts_dir / "split-policy.yaml"
    if policy_path.exists():
        try:
            import yaml

            split_policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - yaml 缺失/损坏
            errors.append(f"split-policy.yaml 读取失败: {exc}")
    else:
        errors.append(f"split-policy.yaml 缺失: {policy_path}")

    # lineage：pilot_dir 内 lineage.jsonl，缺省回退 real-v1 汇总 lineage
    lineage_path = root / "lineage.jsonl"
    if not lineage_path.exists():
        lineage_path = root.parent / "real-v1" / "lineage.jsonl"
    if not lineage_path.exists():
        errors.append(
            f"lineage.jsonl 缺失（试过 {root / 'lineage.jsonl'} 与 {lineage_path}）"
            "——official validation 禁止 lineage=None"
        )
        lineage_dispositions = None
    else:
        try:
            lineage_dispositions = build_lineage_dispositions(lineage_path)
        except json.JSONDecodeError as exc:
            errors.append(f"lineage.jsonl 解析失败: {exc}")
            lineage_dispositions = None

    # annotations 缺席（Pilot-0 双标未完成）时以 admitted-failures 行充当
    # annotations 参与 sample 层重算（total/per_class/split/join/Q2 门）——
    # 行内无双标字段，validate_annotation_pair 会报 annotation_pair 缺失，
    # 故待标注阶段豁免逐条双标语义核对（phase-aware），仅做 sample 层
    # 重算核对；正式双标完成后 annotations.jsonl 存在即走原生路径，此时
    # admitted 行仅作官方集与标注视图的集合一致性核对。
    pending_phase = not annotations
    if annotations:
        # 官方集一致性：admitted-failures 的 sample_id 集合 == annotations 的
        # sample_id 集合（downstream 唯一可消费的官方集必须与标注视图同集）。
        admitted_ids = {
            r.get("sample_id") for r in admitted if isinstance(r, dict)
        }
        annotation_ids = {
            r.get("sample_id") for r in annotations if isinstance(r, dict)
        }
        if admitted_ids != annotation_ids:
            errors.append(
                "admitted-failures.jsonl 与 annotations.jsonl 的 sample_id 集合不一致: "
                f"仅官方集有={len(admitted_ids - annotation_ids)} "
                f"仅标注有={len(annotation_ids - admitted_ids)}——官方集与标注视图"
                "必须同 sample_id 集合"
            )
    else:
        # 双标未完成：以 admitted-failures 行的最小 sample 键集合充当
        # annotations（剥离 FailureState 载荷与双标字段——bundle 门不做
        # 模型消费，只做 sample 层重算）。repeat_signature / signature_id
        # 是 v2 打标字段，随行保留进入 bundle 门（signature_stats 核对
        # 与 effectiveness_dataset 硬门都消费它们）。
        annotations = []
        for row in admitted:
            if not isinstance(row, dict):
                annotations.append(row)
                continue
            sample_keys = (
                "sample_id", "artifact_id", "case_id", "event_time",
                "occurrence_index", "source_group_id",
                "repeat_signature", "signature_id",
            )
            annotations.append({k: row[k] for k in sample_keys if k in row})

    errors.extend(
        validate_dataset_bundle(
            metadata,
            annotations,
            provenance=metadata.get("provenance"),
            split_policy=split_policy,
            lineage_dispositions=lineage_dispositions,
            require_lineage=True,
        )
    )
    if pending_phase:
        # 待标注阶段豁免逐条双标语义核对：过滤 annotation_pair 类报错
        errors = [
            e for e in errors if "annotation_pair" not in e and "agreement" not in e
        ]

    # 双标未完成阶段的补充核对：admitted-failures 行的 sample_id 公式重算
    # 与 provenance 台账 join（FailureState 结构由冻结 schema 在采集端保证）。
    if pending_phase:
        artifact_ids = {
            r.get("original_hash") for r in (metadata.get("provenance") or [])
            if isinstance(r, dict)
        }
        seen: set[str] = set()
        for i, row in enumerate(admitted):
            if not isinstance(row, dict):
                errors.append(f"admitted-failures.jsonl[{i}] 非对象")
                continue
            needed = ("sample_id", "artifact_id", "case_id", "event_time", "occurrence_index")
            sid = row.get("sample_id")
            if all(k in row for k in needed):
                expected = compute_sample_id(
                    row["artifact_id"], row["case_id"], row["event_time"],
                    row["occurrence_index"],
                )
                if sid != expected:
                    errors.append(
                        f"admitted-failures.jsonl[{i}] sample_id 与公式重算不一致: "
                        f"记录={sid} 期望={expected}"
                    )
            else:
                errors.append(f"admitted-failures.jsonl[{i}] 缺 sample_id 重算字段")
            if isinstance(sid, str):
                if sid in seen:
                    errors.append(f"admitted-failures.jsonl[{i}] sample_id 重复: {sid[:16]}…")
                seen.add(sid)
            if row.get("artifact_id") not in artifact_ids:
                errors.append(
                    f"admitted-failures.jsonl[{i}] artifact_id={row.get('artifact_id')!r} "
                    "不在 metadata.provenance 台账中（无来源的样本不得进入正式集）"
                )

    return errors


def validate_dataset_bundle(
    metadata: dict[str, Any],
    annotations: list[dict[str, Any]],
    provenance: list[dict[str, Any]] | None = None,
    split_policy: dict[str, Any] | None = None,
    lineage_dispositions: dict[str, str] | None = None,
    require_lineage: bool = False,
) -> list[str]:
    """数据集 bundle 级语义校验（collector 前硬门，外部审计 P0 裁定）。

    与 validate_metadata 的区别：validate_metadata 只核对 metadata **内部**
    数字彼此一致；本函数从**原始记录重算**并与 metadata 缓存逐项比对——
    生产者手填的"自洽"数字若与真实 annotations/split_assignment/provenance
    不符，在这里被捕获。

    参数：
      metadata: metadata.json 反序列化结果（必含 counts/split/split_assignment/
        split_mode/provenance 或由 provenance 参数独立提供台账）。
      annotations: 标注记录 list（annotation-schema.json 行格式；含 sample_id/
        artifact_id/case_id/event_time/occurrence_index/category/action/
        priority/annotation_pair/source_group_id）。
      provenance: 台账 list；缺省时读 metadata["provenance"]。
      split_policy: contracts/split-policy.yaml 反序列化结果（可选）；提供时
        核对 metadata.split_policy_version 与其实际生效文件一致。
      lineage_dispositions: sample_id → 该样本 lineage latest disposition
        （approve/reject/pending；build_lineage_dispositions 构造）。Q2
        bundle 硬门用——提供时，过密 artifact 的每个样本 latest disposition
        必须 approve；None 时按 require_lineage 决定（True 报错，False 跳过）。
      require_lineage: True 时 lineage_dispositions=None 即报错（official
        validation 禁止 lineage=None；validate_pilot_bundle 传 True）。

    重算并核对的八项（每项失配报一条含具体数字的错误）：
      1. sample 层唯一性：sample_id 无重复；每条记录 sample_id 与
         compute_sample_id 重算一致；逐条 validate_annotation_pair 语义；
         记录级 annotation_schema_version/guideline_version 版本绑定。
      2. counts.total == len(annotations)。
      3. counts.per_class == 从 annotations 最终 category 重算的分布。
      4. counts.domain_relevance == 样本级 domain_relevance 统计
         （继承规则：annotation.artifact_id → provenance.original_hash，
         artifact 级属性 projected to samples；join 缺失报错）。
      5. counts.source_tier_x_domain_relevance（若提供）== 同法重算交叉表。
      6. counts.split / split_assignment / groups_per_split / unique_source_groups：
         group 模式同组禁跨 split、sum(groups_per_split)==unique_source_groups；
         temporal 模式按 event_time/cutoff 逐样本重算、允许同组跨 split、
         union(groups across splits)==unique_source_groups。
      7. temporal_cutoff 合规（cutoff 之后 → holdout）与版本绑定三字段、
         source_type↔source_tier 绑定；提供 split_policy 时另核对
         research_question/grouping_strategy 与其一致（DCL v1.1 收口条件 2）。
      8. Q2 bundle 硬门（P0-1 修复 + 收口 #1）：过密 artifact（>10 样本）
         的每个样本 lineage latest disposition 必须 approve。
      9. batch purpose × controlled repeat（q2-review-policy v2）：
         batch_purpose 闭集；repeat_signature=true 必须携带 signature_id；
         batch_purpose=effectiveness_dataset（或缺省从严视为）时存在
         repeat_signature=true 样本 → 硬报错；signature_stats 与逐行
         重算一致（PR2）。

    全部为纯函数：不做 IO，返回违规列表（空列表 = 通过）。
    """
    errors: list[str] = []
    if not isinstance(metadata, dict):
        return ["metadata 必须为对象"]
    if not isinstance(annotations, list):
        return ["annotations 必须为 list（annotations.jsonl 反序列化结果）"]
    if provenance is None:
        provenance = metadata.get("provenance")

    counts = metadata.get("counts")
    if not isinstance(counts, dict):
        return ["counts 缺失或非对象，无法做 bundle 级计数核对"]

    # ---- 1. sample 层唯一性 + 记录级语义/版本 ----
    seen_samples: dict[str, int] = {}
    for i, record in enumerate(annotations):
        if not isinstance(record, dict):
            errors.append(f"annotations[{i}] 非对象")
            continue
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str):
            first_at = seen_samples.get(sample_id)
            if first_at is not None:
                errors.append(
                    f"sample_id 重复: {sample_id[:16]}…（annotations[{first_at}] 与 "
                    f"annotations[{i}]）——sample 层主键必须唯一"
                )
            else:
                seen_samples[sample_id] = i
        errors.extend(validate_annotation_pair(record))

    # ---- 2. total ----
    total = counts.get("total")
    actual_total = len(annotations)
    if isinstance(total, int) and total != actual_total:
        errors.append(
            f"counts.total={total} != len(annotations)={actual_total}"
        )

    # ---- 3. per_class（收口 #4：annotations 空 + label_status=pending 时跳过；
    #      per_class=null 合法当且仅当 label_status=='pending'）----
    recomputed_per_class = _recompute_per_class(annotations)
    declared_per_class = counts.get("per_class")
    if declared_per_class is None:
        if metadata.get("label_status") != "pending":
            errors.append(
                "counts.per_class=null 但 label_status != 'pending'：未完成标注时必须"
                "声明 label_status='pending'（禁止 UNKNOWN 占位伪装真实分布）"
            )
    elif isinstance(declared_per_class, dict):
        for category in _CATEGORIES:
            declared = declared_per_class.get(category, 0)
            actual = recomputed_per_class[category]
            if isinstance(declared, int) and declared != actual:
                errors.append(
                    f"counts.per_class[{category}]={declared} != 从 annotations "
                    f"重算 {actual}（事实源为最终 category，仲裁值优先）"
                )

    # ---- 4/5. artifact join → sample 级 relevance / tier / 交叉表 ----
    artifact_index = _artifact_fact_index(provenance, errors)
    recomputed_relevance, _, recomputed_cross = _recompute_sample_level_stats(
        annotations, artifact_index, errors
    )
    declared_relevance = counts.get("domain_relevance")
    if isinstance(declared_relevance, dict):
        for rel in _RELEVANCES:
            declared = declared_relevance.get(rel, 0)
            actual = recomputed_relevance[rel]
            if isinstance(declared, int) and declared != actual:
                errors.append(
                    f"counts.domain_relevance[{rel}]={declared} != 从 provenance "
                    f"join 重算 {actual}（artifact 级 fact projected to samples）"
                )
    declared_cross = counts.get("source_tier_x_domain_relevance")
    if isinstance(declared_cross, dict) and recomputed_cross is not None:
        all_keys = sorted(set(declared_cross) | set(recomputed_cross))
        for key in all_keys:
            declared = declared_cross.get(key, 0)
            actual = recomputed_cross.get(key, 0)
            if declared != actual:
                errors.append(
                    f"counts.source_tier_x_domain_relevance[{key!r}]={declared} != "
                    f"从 provenance join 重算 {actual}"
                )

    # ---- 6. split / split_assignment / groups ----
    _check_split_consistency(metadata, annotations, errors)

    # group 模式：assignment 声明的组数 vs unique_source_groups vs 真实组集合
    if metadata.get("split_mode") != "temporal":
        assignments = metadata.get("split_assignment")
        assigned_groups = {
            e.get("source_group_id")
            for e in (assignments or [])
            if isinstance(e, dict) and e.get("source_group_id")
        }
        annotation_groups = {
            r.get("source_group_id") for r in annotations if isinstance(r, dict)
        } - {None}
        unique_declared = counts.get("unique_source_groups")
        if isinstance(unique_declared, int):
            if len(assigned_groups) != unique_declared:
                errors.append(
                    f"counts.unique_source_groups={unique_declared} != "
                    f"split_assignment 去重组数 {len(assigned_groups)}"
                )
            if annotation_groups and len(assigned_groups | annotation_groups) != unique_declared:
                errors.append(
                    f"counts.unique_source_groups={unique_declared} != "
                    f"annotations∪split_assignment 去重组数 "
                    f"{len(assigned_groups | annotation_groups)}"
                )
        if annotation_groups and assigned_groups and annotation_groups != assigned_groups:
            errors.append(
                "split_assignment 的组集合 != annotations 的 source_group_id 集合: "
                f"仅 assignment 有={sorted(assigned_groups - annotation_groups)} "
                f"仅 annotations 有={sorted(annotation_groups - assigned_groups)}"
            )

    # ---- 7a. temporal_cutoff 合规（cutoff 之后 → holdout） ----
    _check_temporal_cutoff_compliance(metadata, annotations, errors)

    # ---- 7b. source_type ↔ source_tier 绑定 ----
    _check_source_type_tier_binding(provenance, errors)

    # ---- 7b-2. Q2 bundle 硬门（P0-1 修复 + 收口 #1：sample 级 disposition）----
    _check_q2_quarantine_review(
        annotations, provenance,
        lineage_dispositions=lineage_dispositions,
        require_lineage=require_lineage,
        errors=errors,
    )

    # ---- 7b-3. batch purpose × controlled repeat（q2-review-policy v2）----
    _check_batch_purpose_and_signature_stats(metadata, annotations, errors)

    # ---- 7c. metadata 版本绑定三字段 + metadata 内部自洽兜底 ----
    for field, expected in DATASET_CONTRACT_VERSIONS.items():
        actual = metadata.get(field)
        if actual != expected:
            errors.append(
                f"{field} 版本绑定失败: metadata={actual!r} 当前冻结值={expected!r}"
                "（版本由 DATASET_CONTRACT_VERSIONS 统一管理）"
            )
    if isinstance(split_policy, dict):
        policy_version = split_policy.get("split_policy_version")
        declared_policy_version = metadata.get("split_policy_version")
        if policy_version is not None and declared_policy_version != policy_version:
            errors.append(
                f"metadata.split_policy_version={declared_policy_version!r} != "
                f"实际生效 split-policy 文件的 {policy_version!r}"
            )
        # DCL v1.1 收口条件 2：研究问题与分组策略是数据集事实的一部分——
        # metadata 必须与实际生效 split-policy 文件一致，禁止两处漂移。
        for field in ("research_question", "grouping_strategy"):
            policy_value = split_policy.get(field)
            declared = metadata.get(field)
            if policy_value is not None and declared != policy_value:
                errors.append(
                    f"metadata.{field}={declared!r} != 实际生效 split-policy 文件的 "
                    f"{policy_value!r}（research_question/grouping_strategy binding："
                    f"如需更换须先在 split-policy 固化为新版本，再同步 metadata）"
                )

    return errors
