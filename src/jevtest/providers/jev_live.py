"""Jev live provider：直调 TypeSafe Jev API（M2 竞争实验 Jev 臂，PRD §11 S2）。

设计要点（S1 探测结论 docs/jev-s1-probe.md）：
- Q1 category / Q2 action 用 Choice（各 6 Opt 带契约语义描述），Q3 priority 用 Score
  （5 级带裁定 5 语义描述），**一次批量调用**（官方 parallel_questions：批量比逐问便宜 10×+）。
- Score 的 score/probabilities 键为 0 起始索引（S1 实测 legend "0".."4"），归一
  priority = clamp(round(score)+1, 1, 5)。
- confidence = Probs[Value]（Q1 正确概率）；confidence_sharpness = Jev 原生
  Confidence（分布尖锐度，官方公式 (n·peak-1)/(n-1)）。两字段分离，禁止混用（PRD §10.2）。
- 重试交给 SDK RetryPolicy（429/529/连接错误/超时，指数退避尊重 retry-after）；
  最终失败原异常上抛——本适配器不吞异常返回假结果，兜底由计量层负责。
- 凭证只读环境变量 TYPESAFE_API_KEY（加载项目 .env），不入代码/日志。
- 出网安全（Mimosa 约束）：仅 https；请求前校验 host，拒绝 localhost/环回/私有/保留地址。
"""

from __future__ import annotations

import ipaddress
import hashlib
import json
import os
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from jevtest.models import FailureState, TriageResult
from jevtest.input_projection import project_failure_state
from jevtest.providers.base import InvalidProviderOutput, Provider

#: 默认官方端点（S1 探测确认：POST {base}/v1/systemone）。
DEFAULT_BASE_URL = "https://api.typesafe.ai"

#: Q1 失败类别：6 Opt 带描述（契约冻结枚举，PRD §10.1 / 裁定 4）。
CATEGORY_CRITERIA: dict[str, str] = {
    "PRODUCT_DEFECT": (
        "The product itself is defective: the failure is reproducible on this software/hardware "
        "version and not explained by lab environment, configuration or test script errors"
    ),
    "TEST_SCRIPT": (
        "The automation test script is buggy: wrong expected value, bad element locator, "
        "mishandled timing, or the script contradicts the product's documented behavior"
    ),
    "ENVIRONMENT": (
        "The lab environment is broken: device/AC/DHCP/RADIUS/NTP/gateway unreachable, "
        "topology down, or resource exhaustion outside the product under test"
    ),
    "CONFIGURATION": (
        "Wrong configuration: a setting required by the test was missing or wrong "
        "(license, VLAN, AAA setup), environment reachable but misconfigured"
    ),
    "FLAKY": (
        "Occasional flakiness: same test passed before with no relevant change, "
        "failure correlates with timing/race/load and is not consistently reproducible"
    ),
    "UNKNOWN": (
        "Cannot be classified into any of the above categories from the available evidence"
    ),
}

#: Q2 下一步动作：6 Opt 带描述（契约冻结枚举，PRD §10.1 / 裁定 4）。
ACTION_CRITERIA: dict[str, str] = {
    "RERUN": "Retry the same test once to check reproducibility (flaky suspects)",
    "COLLECT_MORE_LOGS": (
        "Gather more evidence before judging: full logs, debug traces, packet captures"
    ),
    "CHECK_ENVIRONMENT": "Verify and repair the lab environment probes (reachability, services)",
    "ASSIGN_TEST_OWNER": "Route to the test engineer owning this case (likely script/config issue)",
    "ASSIGN_PRODUCT_DEVELOPER": (
        "Route to product development as a suspected product defect"
    ),
    "HUMAN_TRIAGE": "Escalate to human expert triage (cannot be classified automatically)",
}

#: Q3 优先级：5 级描述（PRD 裁定 5 语义；数组下标 i 对应 priority i+1，Score 0 起始）。
PRIORITY_LEVELS: list[str] = [
    "1: suspected flaky or otherwise low impact; almost nobody needs to see it soon",
    "2: suspected test script issue; the test owner should look at it in normal course",
    "3: environment or configuration problem needing human confirmation; blocks this test run",
    "4: suspected product defect (non-blocking); product developer should see it soon",
    "5: suspected product defect blocking trunk regression; highest visibility, act immediately",
]

_CATEGORY_KEYS = frozenset(CATEGORY_CRITERIA)
_ACTION_KEYS = frozenset(ACTION_CRITERIA)

# ---------------------------------------------------------------------------
# 凭证与出网安全
# ---------------------------------------------------------------------------


def _load_env_file() -> None:
    """从 cwd 及其上级找 .env 加载（python-dotenv，不覆盖已有环境变量）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv 在依赖中，理论不可达
        return
    for candidate in [Path.cwd(), *Path.cwd().parents][:5]:
        env_file = candidate / ".env"
        if env_file.is_file():
            load_dotenv(dotenv_path=env_file, override=False)
            return


def load_typesafe_api_key() -> str:
    """只从环境变量读 key（先尝试加载项目 .env）。缺失时抛 RuntimeError，绝不返回假 key。"""
    _load_env_file()
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set; put it in the environment or the project .env"
        )
    return key


def _is_forbidden_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_https_url(url: str) -> str:
    """出网安全校验：仅 https；拒绝 localhost/环回/私有/保留地址（含 DNS 解析结果）。"""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"only https is allowed, got scheme={parsed.scheme!r}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")
    lowered = hostname.lower()
    if lowered == "localhost" or lowered.endswith(".localhost") or lowered.endswith(".local"):
        raise ValueError(f"host {lowered!r} is forbidden (localhost/local network name)")
    try:
        addr_infos = socket.getaddrinfo(lowered, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host {lowered!r}: {exc}") from exc
    for _, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if _is_forbidden_ip(ip):
            raise ValueError(
                f"host {lowered!r} resolves to forbidden address {ip} (loopback/private/reserved)"
            )
    return url


# ---------------------------------------------------------------------------
# state 序列化与问题构造
# ---------------------------------------------------------------------------


def serialize_state(state: FailureState) -> dict[str, Any]:
    """Build the shared, versioned inference view used by every provider adapter."""
    return project_failure_state(state).model_dump(mode="json")


def build_questions() -> dict[str, Any]:
    """构造三问（Q1/Q2 Choice + Q3 Score）。需要官方 SDK 类型对象，惰性导入。"""
    from typesafe_sdk import Choice, Score

    return {
        "category": Choice(
            instructions=(
                "This is a network device automation test failure record. "
                "Which failure category does it belong to?"
            ),
            criteria=CATEGORY_CRITERIA,
        ),
        "action": Choice(
            instructions=(
                "For this test failure record, what should be the next step for the test team?"
            ),
            criteria=ACTION_CRITERIA,
        ),
        "priority": Score(
            instructions=(
                "Rate the priority of this test failure. Priority decides who sees it first, "
                "not who fixes it first."
            ),
            criteria=PRIORITY_LEVELS,
        ),
    }


def _questions_prompt_version(questions: Mapping[str, Any]) -> str:
    """Hash the canonical JSON form sent by the SDK for this exact question set."""
    payload = {
        name: question.model_dump(mode="json") if hasattr(question, "model_dump") else question
        for name, question in questions.items()
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 答案映射（纯函数，便于离线单测）
# ---------------------------------------------------------------------------


def _as_answer_dict(answer: Any) -> dict[str, Any]:
    if isinstance(answer, Mapping):
        return dict(answer)
    if hasattr(answer, "model_dump"):
        return dict(answer.model_dump())
    raise InvalidProviderOutput(f"answer is neither a mapping nor a pydantic model: {answer!r}")


def _legend_offset(legend: Mapping[str, str] | None) -> int:
    """legend 键的最小值是 0 → 0 起始索引（priority = index+1）；无 legend 按 API 实测默认 0 起始。"""
    if legend is None:
        return 1
    keys = [key for key in legend if str(key).isdigit()]
    return 1 if keys and min(int(key) for key in keys) == 0 else 0


def score_to_priority(score: float, legend: Mapping[str, str] | None = None) -> int:
    """Score 数值 → priority 1..5：就近取整 + 索引偏移（0 起始 legend → +1）+ clamp。"""
    priority = int(round(score)) + _legend_offset(legend)
    return max(1, min(5, priority))


def _choice_prob(answer: Mapping[str, Any], field: str, allowed: frozenset[str]) -> tuple[str, float, float]:
    """从 Choice 答案取（选中值, Probs[Value], 原生 confidence）。"""
    choice = answer.get(field)
    if not isinstance(choice, str) or choice not in allowed:
        raise InvalidProviderOutput(f"choice answer {field}={choice!r} outside frozen enum")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, Mapping) or choice not in probabilities:
        raise InvalidProviderOutput(f"choice answer {field} missing probabilities[{choice!r}]")
    prob = float(probabilities[choice])
    confidence = answer.get("confidence")
    if confidence is None:
        raise InvalidProviderOutput(f"choice answer {field} missing native confidence")
    return choice, prob, float(confidence)


def map_answers_to_triage(answers: Mapping[str, Any]) -> TriageResult:
    """三问答案 dict → TriageResult（契约映射，纯函数）。

    - category/action：取 Choice 选中值；
    - priority：Score 数值就近取整（0 起始索引 +1）并 clamp 1..5；
    - confidence = Q1 Probs[Value]（正确概率）；confidence_sharpness = Q1 原生 Confidence；
    - probs 携带三问完整分布；reason 非空（选中值+概率摘要，无自由文本依据时的契约兜底）。
    """
    missing = {"category", "action", "priority"} - set(answers)
    if missing:
        raise InvalidProviderOutput(f"answers missing questions: {sorted(missing)}")

    category_answer = _as_answer_dict(answers["category"])
    action_answer = _as_answer_dict(answers["action"])
    priority_answer = _as_answer_dict(answers["priority"])

    category, category_prob, category_confidence = _choice_prob(
        category_answer, "choice", _CATEGORY_KEYS
    )
    action, action_prob, _action_confidence = _choice_prob(action_answer, "choice", _ACTION_KEYS)

    if priority_answer.get("type") != "score" and "score" not in priority_answer:
        raise InvalidProviderOutput(f"priority answer is not a score: {priority_answer!r}")
    if "score" not in priority_answer:
        raise InvalidProviderOutput("score answer missing numeric score")
    score = float(priority_answer["score"])
    legend_raw = priority_answer.get("legend")
    legend = dict(legend_raw) if isinstance(legend_raw, Mapping) else None
    offset = _legend_offset(legend)
    priority = score_to_priority(score, legend)
    priority_probs: dict[str, float] = {}
    if isinstance(priority_answer.get("probabilities"), Mapping):
        for key, value in priority_answer["probabilities"].items():
            try:
                priority_probs[str(int(key) + offset)] = float(value)
            except (TypeError, ValueError):
                continue

    probs: dict[str, Any] = {
        "category": {str(k): float(v) for k, v in (category_answer.get("probabilities") or {}).items()}
        if isinstance(category_answer.get("probabilities"), Mapping)
        else {},
        "action": {str(k): float(v) for k, v in (action_answer.get("probabilities") or {}).items()}
        if isinstance(action_answer.get("probabilities"), Mapping)
        else {},
        "priority": priority_probs,
        "priority_score_raw": score,
    }

    reason = (
        f"Q1={category}(p={category_prob:.2f}); "
        f"Q2={action}(p={action_prob:.2f}); "
        f"Q3={priority}(score={score:.2f})"
    )

    return TriageResult(
        category=category,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        priority=priority,
        confidence=category_prob,
        reason=reason[:200],
        probs=probs,
        confidence_sharpness=category_confidence,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class JevLiveProvider(Provider):
    """TypeSafe Jev live 臂：一次批量调用问三问，映射回 TriageResult。"""

    arm_name = "jev"
    is_mock = False
    provider_id = "typesafe"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "jev-latest",
        max_retries: int = 1,
        timeout_s: float = 60.0,
        transport: Any | None = None,
        retry_backoff_s: float = 0.5,
    ) -> None:
        validate_https_url(base_url)
        self._base_url = base_url
        self._model = model
        self.model_id = model
        self.prompt_version = None
        self._api_key = api_key if api_key is not None else load_typesafe_api_key()

        from typesafe_sdk import RetryPolicy, TypeSafeClient

        if max_retries < 0 or retry_backoff_s < 0:
            raise ValueError("retry limits and backoff must be non-negative")
        self._max_total_attempts = max_retries + 1
        self._retry_backoff_s = float(retry_backoff_s)
        # Disable SDK retries. The adapter owns the full attempt budget so every wire request
        # corresponds to one ledger row and parse retries cannot stack on top of SDK retries.
        self._retry = RetryPolicy(max_retries=0, timeout=timeout_s)
        self._client = TypeSafeClient(
            api_key=self._api_key,
            model=model,
            retry=self._retry,
            timeout=timeout_s,
            base_url=base_url,
            transport=transport,
        )
        self.calls_made = 0
        #: 最近一次调用的计量信息（S5 计量层消费）：input/output tokens 与响应 model 版本。
        self.last_usage: dict[str, Any] | None = None

    def predict(self, state: FailureState) -> TriageResult:
        return self._predict(state, max_attempts=None)

    def predict_with_budget(self, state: FailureState, max_attempts: int) -> TriageResult:
        return self._predict(state, max_attempts=max_attempts)

    def _predict(self, state: FailureState, *, max_attempts: int | None) -> TriageResult:
        self.reset_attempts()
        self.last_usage = None
        self.last_raw_output: Any = None
        first_invalid_output: Any = None
        questions = build_questions()
        prompt_version = _questions_prompt_version(questions)
        attempt_limit = self._max_total_attempts
        if max_attempts is not None:
            if max_attempts < 1:
                raise ValueError("max_attempts must be at least one")
            attempt_limit = min(attempt_limit, max_attempts)
        for attempt_index in range(attempt_limit):
            started = perf_counter()
            self.prompt_version = prompt_version
            try:
                self.calls_made += 1
                response = self._client.system_one(
                    serialize_state(state),
                    questions,
                    retry=self._retry,
                )
            except BaseException as exc:
                if self._is_invalid_response_structure(exc):
                    usage = self._usage_from_error_body(getattr(exc, "body", None))
                    input_tokens = self._optional_int(usage.get("input_tokens"))
                    output_tokens = self._optional_int(usage.get("output_tokens"))
                    self.last_usage = {
                        "model": None,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                    unavailable = {
                        "error_code": "invalid_response_structure",
                        "raw_output_available": False,
                    }
                    if first_invalid_output is None:
                        first_invalid_output = unavailable
                    self.last_raw_output = first_invalid_output
                    pricing_status, cost, cost_currency = self._attempt_pricing(input_tokens)
                    self.record_attempt(
                        status="failed",
                        latency_ms=(perf_counter() - started) * 1000.0,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        pricing_status=pricing_status,
                        cost=cost,
                        cost_currency=cost_currency,
                        error_code="invalid_response_structure",
                        error_summary="successful response did not match the provider response schema",
                    )
                    if attempt_index + 1 < attempt_limit:
                        continue
                    raise InvalidProviderOutput(
                        "system_one response structure was invalid",
                        raw_output=first_invalid_output,
                    ) from exc

                error_code = self._error_code(exc)
                self.record_attempt(
                    status="failed",
                    latency_ms=(perf_counter() - started) * 1000.0,
                    error_code=error_code,
                    error_summary=error_code,
                )
                if self._is_retryable_error(exc) and attempt_index + 1 < attempt_limit:
                    self._wait_before_retry(exc, attempt_index)
                    continue
                raise

            usage = getattr(response, "usage", None)
            input_tokens = self._optional_int(getattr(usage, "input_tokens", None))
            output_tokens = self._optional_int(getattr(usage, "output_tokens", None))
            self.last_usage = {
                "model": getattr(response, "model", None),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            self.record_resolved_model(getattr(response, "model", None))
            raw_answers = getattr(response, "answers", None)
            raw_output: Any = raw_answers
            try:
                if raw_answers is None:
                    raise InvalidProviderOutput(
                        "system_one response has no answers map", raw_output=None
                    )
                if hasattr(raw_answers, "model_dump_json"):
                    answers = json.loads(raw_answers.model_dump_json())
                else:
                    answers = {
                        key: json.loads(value.model_dump_json())
                        if hasattr(value, "model_dump_json") else value
                        for key, value in raw_answers.items()
                    }
                raw_output = answers
                result = map_answers_to_triage(answers)
            except InvalidProviderOutput as exc:
                raw_output = exc.raw_output if exc.raw_output is not None else raw_output
                if first_invalid_output is None:
                    first_invalid_output = raw_output
                self.last_raw_output = first_invalid_output
                pricing_status, cost, cost_currency = self._attempt_pricing(input_tokens)
                self.record_attempt(
                    status="failed",
                    latency_ms=(perf_counter() - started) * 1000.0,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    pricing_status=pricing_status,
                    cost=cost,
                    cost_currency=cost_currency,
                    error_code="invalid_provider_output",
                    error_summary="response answers violated the triage contract",
                )
                if attempt_index + 1 < attempt_limit:
                    continue
                raise InvalidProviderOutput(
                    "system_one response did not produce a contract-valid result",
                    raw_output=first_invalid_output,
                ) from exc
            pricing_status, cost, cost_currency = self._attempt_pricing(input_tokens)
            self.record_attempt(
                status="succeeded",
                latency_ms=(perf_counter() - started) * 1000.0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                pricing_status=pricing_status,
                cost=cost,
                cost_currency=cost_currency,
            )
            return result
        raise AssertionError("provider attempt loop exited without result or exception")

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_invalid_response_structure(exc: BaseException) -> bool:
        try:
            from typesafe_sdk import TypeSafeAPIResponseValidationError
        except ImportError:
            return False
        return isinstance(exc, TypeSafeAPIResponseValidationError)

    @classmethod
    def _usage_from_error_body(cls, body: Any) -> dict[str, Any]:
        if isinstance(body, (bytes, bytearray)):
            try:
                body = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}
        elif isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                return {}
        if not isinstance(body, Mapping):
            return {}
        usage = body.get("usage")
        if not isinstance(usage, Mapping):
            return {}
        return {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }

    @staticmethod
    def _attempt_pricing(input_tokens: int | None) -> tuple[str, float | None, str | None]:
        if input_tokens is None:
            return "unknown", None, None
        # Share the frozen S1 rate used by benchmark accounting; output tokens are free.
        from jevtest.pricing import JEV_INPUT_USD_PER_TOKEN

        return "known", round(input_tokens * JEV_INPUT_USD_PER_TOKEN, 12), "USD"

    @staticmethod
    def _error_code(exc: BaseException) -> str:
        status = getattr(exc, "status", None)
        if status is not None:
            return f"http_{status}"
        return "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__.lower()

    @classmethod
    def _is_retryable_error(cls, exc: BaseException) -> bool:
        try:
            from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError
        except ImportError:
            return False
        if isinstance(exc, TypeSafeAPIConnectionError):
            return True
        if isinstance(exc, TypeSafeAPIError):
            status = getattr(exc, "status", None)
            return status in {408, 429} or (isinstance(status, int) and 500 <= status < 600)
        return False

    def _wait_before_retry(self, exc: BaseException, attempt_index: int) -> None:
        retry_after_ms = getattr(exc, "retry_after_ms", None)
        delay = (
            max(0.0, float(retry_after_ms) / 1000.0)
            if retry_after_ms is not None
            else min(self._retry_backoff_s * (2**attempt_index), 5.0)
        )
        if delay > 0:
            time.sleep(delay)

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None and hasattr(client, "close"):
            client.close()

    def __enter__(self) -> "JevLiveProvider":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
