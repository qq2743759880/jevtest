"""LLM 臂（T07）：火山方舟 Volcano Ark `ark-code-latest`，OpenAI 兼容端点接入 Provider ABC。

设计要点（对应任务详单 T07 / PRD §5.1 条款 1 / §8 裁定 4、5）：
- **输入对等**：user content 就是 `state.model_dump_json()` 的原样字节——与 Rule/Jev 臂
  完全相同的 FailureState、同截断的 syslog_excerpt，不附加任何特权字段或 ground truth。
- **输出同构**：predict 返回 TriageResult；解析/契约违规统一抛 `InvalidProviderOutput`
  （绝不吞异常返回假结果），由上层 `predict_validated` / `fallback.sanitize` 兜底。
- **配置不硬编码**：base_url / model 运行时读 `config/providers.json` 的 ark 段；
  api_key 只从环境变量 `ARK_API_KEY` 读取，禁止出现在代码/日志/异常文本中。
- **出网安全**：仅 https；请求前校验 base_url 的 host，拒绝 localhost/环回/私有/
  保留/链路本地/多播地址（Mimosa 约束）。
- **计量**：`calls_made` / `accumulated_cost` / 累计 token 计数对齐 metrics 层记账口径
  （cost = 真实 API 计费；未配置单价时记账为 0.0 并保留 token 数，供计费可见性）。
"""

from __future__ import annotations

import ipaddress
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from jevtest.models import FailureState, TriageResult
from jevtest.input_projection import project_failure_state, projection_json
from jevtest.providers.base import InvalidProviderOutput, Provider
from jevtest.resource_paths import resource_path

#: 默认 providers 配置路径（仓库根 config/providers.json；包以 `pip install -e .` 安装时可达）。
_DEFAULT_CONFIG_PATH = resource_path("config", "providers.json")

#: 覆盖默认配置路径的环境变量名（可选）。
CONFIG_PATH_ENV_VAR = "JEVTEST_PROVIDERS_CONFIG"

_AUTH_ENV_VAR = "ARK_API_KEY"

_MAX_REASON_CHARS = 200

_VALID_CATEGORIES = frozenset(
    {"PRODUCT_DEFECT", "TEST_SCRIPT", "ENVIRONMENT", "CONFIGURATION", "FLAKY", "UNKNOWN"}
)
_VALID_ACTIONS = frozenset(
    {
        "RERUN",
        "COLLECT_MORE_LOGS",
        "CHECK_ENVIRONMENT",
        "ASSIGN_TEST_OWNER",
        "ASSIGN_PRODUCT_DEVELOPER",
        "HUMAN_TRIAGE",
    }
)
_REQUIRED_KEYS = frozenset({"category", "action", "priority", "confidence", "reason"})

_SYSTEM_PROMPT = """You are a strict test-failure triage judge for network device (WLAN AC/AP) automation tests. You receive exactly one failure record and must answer three questions (Q1/Q2/Q3) with evidence from the record only. Do not invent facts; if evidence is insufficient, choose UNKNOWN/HUMAN_TRIAGE.

Q1 category — pick exactly one of the six frozen enums:
- PRODUCT_DEFECT: the product under test (device firmware/software) itself misbehaves; the failure signature points at product logic (e.g. protocol stack timeout, unexpected teardown, crash), not the test or environment.
- TEST_SCRIPT: the automation test code itself is wrong (bad assertion/selector/test logic); the product likely behaves correctly.
- ENVIRONMENT: test environment or external dependency is unhealthy (probes like AP_reachable/AC_reachable/gateway_reachable false, DHCP/NTP/RADIUS server unreachable, lab link down).
- CONFIGURATION: device or testbench configuration mismatch (wrong config applied, feature disabled, param out of expected range).
- FLAKY: nondeterministic outcome — history shows PASS<->FAIL flips with consistent environment probes; retry would likely pass.
- UNKNOWN: evidence is insufficient or contradictory; nothing above fits.

Q2 action — pick exactly one of the six frozen enums:
- RERUN: re-execute the same case (fits suspected flaky).
- COLLECT_MORE_LOGS: gather more logs/traces before judging further.
- CHECK_ENVIRONMENT: inspect environment probes / lab health (fits ENVIRONMENT).
- ASSIGN_TEST_OWNER: hand to the test-case owner (fits TEST_SCRIPT).
- ASSIGN_PRODUCT_DEVELOPER: hand to product developers (fits PRODUCT_DEFECT).
- HUMAN_TRIAGE: escalate to human triage; mandatory fallback for UNKNOWN.

Q3 priority — integer 1..5 (decides who sees it first, not who fixes it first):
- 5: suspected product defect blocking mainline regression.
- 4: suspected product defect, non-blocking.
- 3: environment/configuration problem needing human confirmation.
- 2: suspected test-script problem.
- 1: suspected flaky / low impact.

confidence — your self-assessed probability that the triage is correct, a number in [0,1].
reason — one human-readable evidence line, at most 200 characters, MUST cite concrete evidence from the record (assertion text, env probe values, previous_runs pattern, retry status, syslog lines). Never output an empty reason.

Output requirements:
- Reply with STRICT JSON only, no markdown, no prose before or after.
- JSON object with exactly these keys: {"category": <one of the six>, "action": <one of the six>, "priority": <int 1..5>, "confidence": <number 0..1>, "reason": <string, <=200 chars, with evidence>}."""


class LLMJudgeConfigError(Exception):
    """LLM 臂配置错误（缺 key / 配置文件缺失或格式不对 / base_url 违反出网安全）。"""


def _validate_https_host(base_url: str) -> None:
    """出网安全硬约束：仅 https；拒绝 localhost/环回/私有/保留/链路本地/多播 host。"""
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    if parsed.scheme != "https":
        raise LLMJudgeConfigError(f"base_url must use https, got scheme {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise LLMJudgeConfigError("base_url has no hostname")
    if host == "localhost" or host.endswith(".localhost"):
        raise LLMJudgeConfigError("localhost host is not allowed")
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return  # 域名（非 IP 字面量）：字面量层面的私有/保留检查通过
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_reserved
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_unspecified
    ):
        raise LLMJudgeConfigError(f"host {host} resolves to a private/reserved address class")


def load_ark_config(config_path: str | Path | None = None) -> dict[str, str]:
    """运行时读取 ark 段配置，返回 {openai_compatible_base, model}。禁止硬编码端点/模型。"""
    path = Path(config_path) if config_path is not None else Path(
        os.environ.get(CONFIG_PATH_ENV_VAR, _DEFAULT_CONFIG_PATH)
    )
    if not path.is_file():
        raise LLMJudgeConfigError(f"providers config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ark = data["providers"]["ark"]
        base = ark["openai_compatible_base"]
        model = ark["model"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LLMJudgeConfigError(f"invalid providers config at {path}: {exc}") from exc
    if not isinstance(base, str) or not base:
        raise LLMJudgeConfigError("ark.openai_compatible_base missing or empty")
    if not isinstance(model, str) or not model:
        raise LLMJudgeConfigError("ark.model missing or empty")
    _validate_https_host(base)
    return {"openai_compatible_base": base, "model": model}


def build_system_prompt() -> str:
    """System prompt：三问定义 + 6+6 枚举语义 + priority 1-5 语义 + 严格 JSON 输出要求。"""
    return _SYSTEM_PROMPT


def _prompt_version(response_mode: str) -> str:
    """Content hash for the exact system prompt and response mode of one wire attempt."""
    content = json.dumps(
        {"system_prompt": build_system_prompt(), "response_mode": response_mode},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_user_content(state: FailureState) -> str:
    """Serialize the same versioned inference facts sent by the Jev adapter."""
    return projection_json(project_failure_state(state))


def build_messages(state: FailureState) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_content(state)},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型回复文本中提取 JSON 对象：优先整体解析，退而求其次扫描首个平衡的 {...}。"""
    if not text or not text.strip():
        raise InvalidProviderOutput("empty model response")
    candidates = [text.strip()]
    stripped = text.strip()
    if stripped.startswith("```"):
        # markdown 围栏：去掉首行 ```json 与末行 ```
        body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        body = body.rsplit("```", 1)[0]
        candidates.insert(0, body.strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    # 首个平衡大括号扫描（容忍模型在 JSON 前后加了说明文字）
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    raise InvalidProviderOutput("no parsable JSON object found in model response")


def parse_llm_triage(text: str) -> TriageResult:
    """把模型回复解析为 TriageResult；任何违规都抛 InvalidProviderOutput（不静默放行）。"""
    payload = extract_json_object(text)
    missing = _REQUIRED_KEYS - payload.keys()
    if missing:
        raise InvalidProviderOutput(f"model JSON missing required keys: {sorted(missing)}")

    category = _normalize_enum(payload["category"], _VALID_CATEGORIES, "category")
    action = _normalize_enum(payload["action"], _VALID_ACTIONS, "action")
    priority = _coerce_priority(payload["priority"])
    confidence = _coerce_confidence(payload["confidence"])
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidProviderOutput("model reason is empty or not a string")
    reason = reason.strip()[:_MAX_REASON_CHARS]
    # confidence_sharpness / probs 是 Jev 臂特有承载，LLM 臂不填。
    return TriageResult(
        category=category,
        action=action,
        priority=priority,
        confidence=confidence,
        reason=reason,
    )


def _normalize_enum(value: Any, allowed: frozenset[str], field: str) -> str:
    if not isinstance(value, str):
        raise InvalidProviderOutput(f"model field {field!r} is not a string: {value!r}")
    normalized = value.strip().upper()
    if normalized not in allowed:
        raise InvalidProviderOutput(f"model field {field!r} outside frozen enum: {value!r}")
    return normalized


def _coerce_priority(value: Any) -> int:
    if isinstance(value, bool):
        raise InvalidProviderOutput(f"model priority is a bool: {value!r}")
    try:
        priority = int(str(value).strip())
    except (ValueError, TypeError) as exc:
        raise InvalidProviderOutput(f"model priority not an integer: {value!r}") from exc
    if not 1 <= priority <= 5:
        raise InvalidProviderOutput(f"model priority out of range 1..5: {priority}")
    return priority


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise InvalidProviderOutput(f"model confidence is a bool: {value!r}")
    try:
        confidence = float(value)
    except (ValueError, TypeError) as exc:
        raise InvalidProviderOutput(f"model confidence not a number: {value!r}") from exc
    if not 0.0 <= confidence <= 1.0:
        raise InvalidProviderOutput(f"model confidence out of range 0..1: {confidence}")
    return confidence


class LLMJudgeProvider(Provider):
    """强 LLM 对照臂（Ark ark-code-latest，OpenAI 兼容端点）。arm_name="llm"。"""

    arm_name = "llm"
    is_mock = False
    provider_id = "ark"

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        client: Any | None = None,
        temperature: float = 0.2,
        retry_temperature: float = 0.0,
        timeout_s: float = 120.0,
        cost_input_per_1k: float | None = None,
        cost_output_per_1k: float | None = None,
        transport: Any | None = None,
    ) -> None:
        """`client` 仅用于测试注入（duck-typed：暴露 chat.completions.create）。

        cost 单价参数（每 1k token 价格）由实验方在计费口径冻结后传入；未配置时
        accumulated_cost 记 0.0 但 token 数照记（metrics 层 cost 口径仍可区分）。
        """
        config = load_ark_config(config_path)
        self._base_url = config["openai_compatible_base"]
        self._model = config["model"]
        self.model_id = self._model
        # The system prompt has not yet been assigned a signed version identifier.
        self.prompt_version = None
        self.temperature = float(temperature)
        self.retry_temperature = float(retry_temperature)
        self.timeout_s = float(timeout_s)
        self._cost_input_per_1k = cost_input_per_1k
        self._cost_output_per_1k = cost_output_per_1k

        self.calls_made = 0
        self.accumulated_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get(_AUTH_ENV_VAR)
            if not api_key:
                raise LLMJudgeConfigError(
                    f"environment variable {_AUTH_ENV_VAR} is not set"
                )
            from openai import OpenAI

            http_client = None
            if transport is not None:
                import httpx

                http_client = httpx.Client(transport=transport, timeout=self.timeout_s)

            # key 只在客户端实例内使用，绝不入日志/异常文本。
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=api_key,
                max_retries=0,
                timeout=self.timeout_s,
                http_client=http_client,
            )

    def predict(self, state: FailureState) -> TriageResult:
        return self._predict(state, max_attempts=None)

    def predict_with_budget(self, state: FailureState, max_attempts: int) -> TriageResult:
        return self._predict(state, max_attempts=max_attempts)

    def _predict(self, state: FailureState, *, max_attempts: int | None) -> TriageResult:
        """单样本判定：至多 2 次真实调用——先 JSON mode；输出不可解析/违反契约时温度 0 重试 1 次。

        端点若拒绝 response_format（400），第 2 次自动降级为纯文本重试（不额外消耗次数）。
        其余 API 层错误（认证/限流/超时等）原样上抛，绝不吞掉返回假结果。
        """
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.reset_attempts()
        self.last_usage: dict[str, Any] | None = None
        self.last_raw_output: Any = None
        messages = build_messages(state)
        attempts: list[tuple[float, bool]] = [
            (self.temperature, True),
            (self.retry_temperature, False),
        ]
        if max_attempts is not None:
            attempts = attempts[:max_attempts]
        errors: list[str] = []
        raw_outputs: list[Any] = []
        for temperature, json_mode in attempts:
            try:
                text = self._call(messages, temperature, json_mode=json_mode)
            except _BadRequestForJsonMode as exc:
                if json_mode:
                    errors.append("endpoint rejected response_format (HTTP 400)")
                    # A caller-provided attempt budget is also a wire-request
                    # budget. The live-advisory profile passes one, so it must
                    # surface this failure instead of issuing a text-mode retry.
                    if max_attempts is not None and len(self.last_attempts) >= max_attempts:
                        cause = exc.__cause__
                        raise cause if cause is not None else exc
                    continue  # 降级为无 json mode 的重试，不额外消耗次数
                cause = exc.__cause__
                raise cause if cause is not None else exc
            except InvalidProviderOutput as exc:
                errors.append(f"attempt(json_mode={json_mode}): {exc}")
                raw_outputs.append(exc.raw_output)
                continue
            except Exception as exc:
                errors.append(type(exc).__name__)
                if raw_outputs:
                    self.last_raw_output = raw_outputs[0]
                if self._is_retryable_api_error(exc) and len(self.last_attempts) < len(attempts):
                    continue
                raise
            try:
                result = parse_llm_triage(text)
            except InvalidProviderOutput as exc:
                errors.append(f"attempt(json_mode={json_mode}): {exc}")
                raw_outputs.append(text)
                if self.last_raw_output is None:
                    self.last_raw_output = text
                self._mark_last_attempt_failed(
                    "invalid_provider_output", "response was not valid triage JSON"
                )
            else:
                self._mark_last_attempt_succeeded()
                return result
        raise InvalidProviderOutput(
            "llm arm failed to produce contract-valid JSON after retry: " + "; ".join(errors),
            raw_output=raw_outputs[0] if raw_outputs else self.last_raw_output,
        )

    def _call(self, messages: list[dict[str, str]], temperature: float, json_mode: bool) -> str:
        """一次真实 API 调用；记账 calls_made / tokens / cost。API 错误原样上抛（不吞）。"""
        import openai as _openai

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self.prompt_version = _prompt_version("json" if json_mode else "text")
        started = perf_counter()
        self.calls_made += 1
        try:
            response = self._client.chat.completions.create(**kwargs)
        except _openai.BadRequestError as exc:
            # 端点可能不支持 response_format=json_object：交由上层降级为纯文本重试。
            self.record_attempt(
                status="failed",
                latency_ms=(perf_counter() - started) * 1000.0,
                error_code="http_400_json_mode_rejected" if json_mode else "http_400",
            )
            raise _BadRequestForJsonMode(str(exc)) from exc
        except BaseException as exc:
            status_code = getattr(exc, "status_code", None)
            invalid_structure = isinstance(exc, _openai.APIResponseValidationError)
            body = getattr(exc, "body", None) if invalid_structure else None
            usage = body.get("usage") if isinstance(body, dict) else None
            prompt_tokens = self._optional_int(getattr(usage, "prompt_tokens", None))
            completion_tokens = self._optional_int(getattr(usage, "completion_tokens", None))
            if isinstance(usage, dict):
                prompt_tokens = self._optional_int(usage.get("prompt_tokens"))
                completion_tokens = self._optional_int(usage.get("completion_tokens"))
            if usage is not None:
                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
                if prompt_tokens is not None:
                    self.total_prompt_tokens += prompt_tokens
                if completion_tokens is not None:
                    self.total_completion_tokens += completion_tokens
            cost, pricing_status = self._price_usage(prompt_tokens, completion_tokens)
            error_code = (
                "invalid_response_structure" if invalid_structure else
                f"http_{status_code}" if status_code is not None
                else "timeout" if isinstance(exc, _openai.APITimeoutError)
                else "connection_error" if isinstance(exc, _openai.APIConnectionError)
                else type(exc).__name__.lower()
            )
            self.record_attempt(
                status="failed",
                latency_ms=(perf_counter() - started) * 1000.0,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cost=cost,
                cost_currency="USD" if pricing_status == "known" else None,
                pricing_status=pricing_status,
                error_code=error_code,
                error_summary=error_code,
            )
            if cost is not None:
                self.accumulated_cost = round(self.accumulated_cost + cost, 12)
            if invalid_structure:
                raise InvalidProviderOutput(
                    "model returned an invalid response structure",
                    raw_output={"error_code": error_code},
                ) from exc
            raise
        usage = getattr(response, "usage", None)
        prompt_tokens = self._optional_int(getattr(usage, "prompt_tokens", None))
        completion_tokens = self._optional_int(getattr(usage, "completion_tokens", None))
        self.last_usage = {
            "model": getattr(response, "model", None),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        } if usage is not None else None
        self.record_resolved_model(getattr(response, "model", None))
        cost, pricing_status = self._price_usage(prompt_tokens, completion_tokens)
        self.record_attempt(
            status="succeeded",
            latency_ms=(perf_counter() - started) * 1000.0,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost=cost,
            cost_currency="USD" if pricing_status == "known" else None,
            pricing_status=pricing_status,
        )
        if prompt_tokens is not None:
            self.total_prompt_tokens += prompt_tokens
        if completion_tokens is not None:
            self.total_completion_tokens += completion_tokens
        if cost is not None:
            self.accumulated_cost = round(self.accumulated_cost + cost, 12)
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            self._mark_last_attempt_failed(
                "invalid_response_structure", "response choices/message content missing"
            )
            raise InvalidProviderOutput(
                "model returned an invalid response structure",
                raw_output={"error_code": "invalid_response_structure"},
            ) from exc
        if content is None:
            self._mark_last_attempt_failed("empty_content", "response contained no content")
            raise InvalidProviderOutput("model response has empty content", raw_output=None)
        return content

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    def _price_usage(
        self, prompt_tokens: int | None, completion_tokens: int | None
    ) -> tuple[float | None, str]:
        if (
            prompt_tokens is None
            or completion_tokens is None
            or self._cost_input_per_1k is None
            or self._cost_output_per_1k is None
        ):
            return None, "unknown"
        cost = (prompt_tokens / 1000.0 * self._cost_input_per_1k) + (
            completion_tokens / 1000.0 * self._cost_output_per_1k
        )
        return round(cost, 12), "known"

    @staticmethod
    def _is_retryable_api_error(exc: Exception) -> bool:
        import openai

        if isinstance(exc, openai.APIConnectionError):
            return True
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", None)
            return status in {408, 429} or (isinstance(status, int) and 500 <= status < 600)
        return False

    def _mark_last_attempt_failed(self, error_code: str, error_summary: str) -> None:
        if not self.last_attempts:
            return
        row = self.last_attempts[-1]
        self.last_attempts[-1] = row.model_copy(
            update={
                "status": "failed",
                "error_code": error_code,
                "error_summary": error_summary[:500],
            }
        )

    def _mark_last_attempt_succeeded(self) -> None:
        if not self.last_attempts:
            return
        row = self.last_attempts[-1]
        self.last_attempts[-1] = row.model_copy(
            update={"status": "succeeded", "error_code": None, "error_summary": None}
        )

    def close(self) -> None:
        """Release the OpenAI-compatible HTTP client owned by this provider."""

        client = getattr(self, "_client", None)
        if client is not None and hasattr(client, "close"):
            client.close()


class _BadRequestForJsonMode(Exception):
    """内部信号：400 类错误，可能是 response_format 不被端点支持，触发无 json mode 重试。"""
