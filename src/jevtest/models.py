"""契约模型层：pydantic v2 模型是 contracts/*.json 冻结文本的单一事实源。"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ArmName = Literal["rule", "jev", "llm", "cascade"]
_ARM_KEY_PATTERN = "^(rule|jev|llm|cascade)$"
_ARM_KEY_RE = re.compile(_ARM_KEY_PATTERN)
_UnitInterval = Annotated[float, Field(ge=0, le=1)]


class Provenance(BaseModel):
    """数据来源标记（合成/真实分离纪律，PRD 裁定 2）。"""

    model_config = ConfigDict(extra="forbid")

    dataset_version: str = Field(min_length=1)
    synthetic: bool
    source: str | None = Field(default=None, description="真实日志来源或合成器 seed 标识")


class FailureState(BaseModel):
    """四臂实验统一输入。约束的是“提供什么”而非“使用什么”——任何臂不得获得本结构之外的额外信息（PRD §5.1 条款 1）。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, description="测试用例唯一标识")
    result: Literal["FAIL"] = Field(description="本契约只处理失败记录")
    previous_runs: list[Literal["PASS", "FAIL"]] = Field(
        max_length=10, description="时间正序的历史运行结果，≤10 条"
    )
    device: str = Field(min_length=1, description="设备型号，如 WA6638")
    sw_version: str = Field(min_length=1, description="软件版本")
    assertion: str = Field(min_length=1, description="失败断言摘要，如 association timeout")
    syslog_excerpt: str = Field(
        max_length=8192,
        description="系统日志摘要；截断策略 M1 定参（默认首尾各 4096 字节），四臂同截断",
    )
    env: dict[str, bool | str] = Field(
        description="环境探针布尔/字符串映射，如 AP_reachable / AC_reachable"
    )
    retry: Literal["not_executed", "passed", "failed"] = Field(description="重跑状态")
    provenance: Provenance


class TriageResult(BaseModel):
    """四臂统一输出（PRD §10.1 三问契约）。所有臂实现 predict(FailureState) -> TriageResult。"""

    category: Literal[
        "PRODUCT_DEFECT", "TEST_SCRIPT", "ENVIRONMENT", "CONFIGURATION", "FLAKY", "UNKNOWN"
    ] = Field(description="Q1 失败类别（裁定 4：M3 前冻结）")
    action: Literal[
        "RERUN",
        "COLLECT_MORE_LOGS",
        "CHECK_ENVIRONMENT",
        "ASSIGN_TEST_OWNER",
        "ASSIGN_PRODUCT_DEVELOPER",
        "HUMAN_TRIAGE",
    ] = Field(description="Q2 下一步动作（裁定 4：M3 前冻结）")
    priority: int = Field(
        ge=1,
        le=5,
        description="Q3 优先级，语义见 PRD 裁定 5（5=疑似阻断主干回归的产品缺陷 … 1=疑似 flaky）；"
        "决定“谁先看到”而非“谁先修”",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="消费方=漏斗路由器；阈值是策略，由调用方代码持有（官方语义）。"
        "Jev 臂此字段承载 Probs[Value]（正确概率），分布尖锐度放 confidence_sharpness",
    )
    reason: str = Field(
        min_length=1,
        max_length=200,
        description="人可读依据摘要。Rule 臂必须含规则 ID+触发字段值；"
        "缺失或空的 reason 视为无效输出，路由器按 UNKNOWN+HUMAN_TRIAGE 兜底",
    )
    probs: dict[str, Any] | None = Field(
        default=None, description="可选：逐选项概率分布（Jev Choice/Score 原生返回）"
    )
    confidence_sharpness: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="可选：Jev Confidence 原值（分布尖锐度，非校准概率）——与 confidence 分字段承载，禁止混用",
    )
    routing_path: str | None = Field(
        default=None, description="可选（Cascade 臂）：实际走过的路由，如 rule|jev|llm|human"
    )


class AbsentArm(BaseModel):
    """缺席臂的原因说明。"""

    arm: _ArmName
    reason: str


class ArmMetrics(BaseModel):
    """单臂计量指标。"""

    accuracy: _UnitInterval
    cost: float = Field(
        ge=0, description="整批总费用（货币单位在 cost_currency 声明）；mock 臂为 0"
    )
    cost_currency: str | None = None
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    routing_ratio: dict[str, _UnitInterval] = Field(
        description="分流比例，键为 rule/jev/llm/human，值和为 1"
    )
    sample_count: int | None = Field(default=None, ge=1)


def _inject_metrics_pattern(schema: dict[str, Any]) -> None:
    # pydantic 无法原生表达 patternProperties：把 metrics 的键模式注入导出 schema，
    # 使其与 benchmark.schema.json 冻结文本一致；运行时校验由 _check_metrics_keys 承担。
    metrics = schema.get("properties", {}).get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("additionalProperties"), dict):
        metrics["patternProperties"] = {_ARM_KEY_PATTERN: metrics["additionalProperties"]}
        metrics["additionalProperties"] = False


class BenchmarkExperiment(BaseModel):
    """四臂竞争实验的报告契约（PRD §5.1 条款 + Risk-001 最小 schema）。"""

    model_config = ConfigDict(extra="forbid", json_schema_extra=_inject_metrics_pattern)

    experiment_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_kind: Literal["synthetic", "real", "mixed"]
    rule_version: str = Field(min_length=1, description='Rule 臂基线版本，如 "0"')
    arms: list[_ArmName] = Field(
        min_length=1, description="实际参与的臂；缺席臂必须在 absent_arms 说明原因"
    )
    absent_arms: list[AbsentArm] | None = None
    metrics: dict[str, ArmMetrics] = Field(description="逐臂指标")
    per_sample_path: str | None = Field(
        default=None,
        description="逐样本 JSONL 路径，行格式 = TriageResult + arm + case_id + latency_ms + cost",
    )
    error_samples_path: str | None = Field(
        default=None,
        description="解析失败样本追踪 JSONL（E_PARSE_FAILURE 同步落盘，owner 审阅增强项）："
        "每行 {case_id, arm, raw_output, timestamp}——保证失败样本可追溯、可重放",
    )
    declaration: Literal["SYNTHETIC — 不代表企业效果", "REAL — 正式基准"] = Field(
        description="数据性质强制声明（裁定 2：合成/真实永远分开汇报）"
    )

    @field_validator("metrics")
    @classmethod
    def _check_metrics_keys(cls, value: dict[str, ArmMetrics]) -> dict[str, ArmMetrics]:
        invalid = [key for key in value if not _ARM_KEY_RE.fullmatch(key)]
        if invalid:
            raise ValueError(f"metrics keys must match {_ARM_KEY_PATTERN}, got {invalid}")
        return value
