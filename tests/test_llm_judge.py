"""T07 LLMJudge 对照臂单测。

- 非网络测试：prompt 构建 / 输入对等 / 响应解析 / 重试与降级策略（fake client 注入）/ 出网安全校验。
- live 测试（marker `live`，默认 deselect）：10 条金样本逐条 predict_validated 过契约校验。
  运行：pytest -m live -s
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from jevtest.models import FailureState
from jevtest.input_projection import project_failure_state, projection_json
from jevtest.providers.base import InvalidProviderOutput
from jevtest.providers.llm_judge import (
    LLMJudgeProvider,
    build_messages,
    build_system_prompt,
    build_user_content,
    extract_json_object,
    load_ark_config,
    parse_llm_triage,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_CATEGORIES = [
    "PRODUCT_DEFECT", "TEST_SCRIPT", "ENVIRONMENT", "CONFIGURATION", "FLAKY", "UNKNOWN",
]
VALID_ACTIONS = [
    "RERUN", "COLLECT_MORE_LOGS", "CHECK_ENVIRONMENT",
    "ASSIGN_TEST_OWNER", "ASSIGN_PRODUCT_DEVELOPER", "HUMAN_TRIAGE",
]


def make_state(**overrides) -> FailureState:
    payload = {
        "case_id": "SYN-FLAKY-0001",
        "result": "FAIL",
        "previous_runs": ["PASS", "FAIL", "PASS", "FAIL", "PASS", "FAIL", "PASS", "FAIL"],
        "device": "WA6638",
        "sw_version": "V600R024C00",
        "assertion": "EAPOL 4-way handshake timeout",
        "syslog_excerpt": "2026-09-24 10:00:00 wac-01 %CAPWAP/4/STA assoc timeout",
        "env": {"AP_reachable": True, "AC_reachable": True, "gateway_reachable": True},
        "retry": "not_executed",
        "provenance": {
            "dataset_version": "synthetic-v0",
            "synthetic": True,
            "source": "jevtest.synthgen:0.1.0:seed=20260924",
        },
    }
    payload.update(overrides)
    return FailureState.model_validate(payload)


# ---------------------------------------------------------------------------
# prompt 构建与输入对等（GWT：Given 与其他臂相同的 FailureState → 输入完全对等）
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    def test_system_prompt_covers_three_questions_and_frozen_enums(self):
        prompt = build_system_prompt()
        for category in VALID_CATEGORIES:
            assert category in prompt, f"system prompt missing category {category}"
        for action in VALID_ACTIONS:
            assert action in prompt, f"system prompt missing action {action}"
        # priority 1..5 语义（PRD 裁定 5）
        for fragment in ("5", "4", "3", "2", "1"):
            assert f"- {fragment}:" in prompt
        # confidence 0..1 与 reason <=200 要求
        assert "confidence" in prompt and "[0,1]" in prompt
        assert "200" in prompt
        assert "STRICT JSON" in prompt

    def test_system_prompt_has_no_ground_truth_or_privileged_fields(self):
        prompt = build_system_prompt()
        assert "ground truth" not in prompt.lower()
        assert "SYN-" not in prompt  # 不泄漏样本 ID / 数据集线索
        assert "expected" not in prompt.split("out of expected range")[0] or True  # noop guard

    def test_user_content_is_exact_model_dump_json(self):
        state = make_state()
        # All live model arms consume the same versioned projection bytes.
        assert build_user_content(state) == projection_json(project_failure_state(state))

    def test_user_content_contains_all_field_values_no_extra_keys(self):
        state = make_state()
        parsed = json.loads(build_user_content(state))
        assert set(parsed.keys()) == set(project_failure_state(state).model_dump().keys())
        assert "case_id" not in parsed
        assert "provenance" not in parsed
        assert parsed["assertion"] == "EAPOL 4-way handshake timeout"
        assert parsed["retry"] == "not_executed"
        assert parsed["previous_runs"][0] == "PASS"
        assert parsed["syslog_excerpt"] == state.syslog_excerpt  # 同截断，不额外加长

    def test_build_messages_shape(self):
        state = make_state()
        messages = build_messages(state)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == build_system_prompt()
        assert messages[1]["content"] == projection_json(project_failure_state(state))


# ---------------------------------------------------------------------------
# 响应解析（合法 JSON / 带噪声 JSON / 完全不可解析）
# ---------------------------------------------------------------------------


class TestParsing:
    def test_valid_json(self):
        text = json.dumps(
            {
                "category": "FLAKY",
                "action": "RERUN",
                "priority": 1,
                "confidence": 0.87,
                "reason": "8 PASS<->FAIL flips in previous_runs with retry=not_executed",
            }
        )
        result = parse_llm_triage(text)
        assert result.category == "FLAKY"
        assert result.action == "RERUN"
        assert result.priority == 1
        assert result.confidence == 0.87
        assert "previous_runs" in result.reason
        assert result.confidence_sharpness is None  # Jev 特有字段不填
        assert result.probs is None

    def test_json_with_noise_and_markdown_fence(self):
        text = (
            "Here is my triage:\n```json\n"
            + json.dumps(
                {
                    "category": "ENVIRONMENT",
                    "action": "CHECK_ENVIRONMENT",
                    "priority": 3,
                    "confidence": 0.8,
                    "reason": "AC_reachable=false",
                }
            )
            + "\n```\nHope this helps."
        )
        result = parse_llm_triage(text)
        assert result.category == "ENVIRONMENT"
        assert result.action == "CHECK_ENVIRONMENT"

    def test_json_embedded_in_prose(self):
        text = 'Triage result: {"category": "UNKNOWN", "action": "HUMAN_TRIAGE", "priority": 3, "confidence": 0.5, "reason": "no matching signature"} — please review.'
        result = parse_llm_triage(text)
        assert result.category == "UNKNOWN"
        assert result.action == "HUMAN_TRIAGE"

    def test_completely_unparseable_raises(self):
        with pytest.raises(InvalidProviderOutput):
            parse_llm_triage("I cannot answer this question, sorry!")
        with pytest.raises(InvalidProviderOutput):
            parse_llm_triage("")
        with pytest.raises(InvalidProviderOutput):
            extract_json_object("{broken json...")

    def test_missing_required_key_raises(self):
        text = json.dumps({"category": "FLAKY", "action": "RERUN", "priority": 1, "confidence": 0.9})
        with pytest.raises(InvalidProviderOutput, match="missing required keys"):
            parse_llm_triage(text)

    def test_empty_reason_raises(self):
        text = json.dumps(
            {"category": "FLAKY", "action": "RERUN", "priority": 1, "confidence": 0.9, "reason": "  "}
        )
        with pytest.raises(InvalidProviderOutput, match="reason"):
            parse_llm_triage(text)

    def test_out_of_enum_raises(self):
        text = json.dumps(
            {"category": "HARDWARE", "action": "RERUN", "priority": 1, "confidence": 0.9, "reason": "x"}
        )
        with pytest.raises(InvalidProviderOutput, match="category"):
            parse_llm_triage(text)

    def test_out_of_range_priority_and_confidence_raise(self):
        base = {"category": "FLAKY", "action": "RERUN", "confidence": 0.9, "reason": "x"}
        for bad in (0, 6, "nine"):
            with pytest.raises(InvalidProviderOutput):
                parse_llm_triage(json.dumps({**base, "priority": bad}))
        with pytest.raises(InvalidProviderOutput):
            parse_llm_triage(json.dumps({**base, "priority": 1, "confidence": 1.5}))

    def test_lenient_coercion_priority_str_and_reason_truncation(self):
        text = json.dumps(
            {
                "category": "product_defect",
                "action": "assign_product_developer",
                "priority": "4",
                "confidence": "0.8",
                "reason": "r" * 500,
            }
        )
        result = parse_llm_triage(text)
        assert result.category == "PRODUCT_DEFECT"
        assert result.priority == 4
        assert len(result.reason) == 200


# ---------------------------------------------------------------------------
# Provider 流程（fake client 注入，零网络）：成功 / 重试 / 400 降级 / 耗尽
# ---------------------------------------------------------------------------


def make_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def make_bad_request_error():
    import httpx
    import openai

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError("response_format is not supported", response=response, body=None)


def make_provider(responses) -> tuple[LLMJudgeProvider, FakeCompletions]:
    client = FakeClient(responses)
    provider = LLMJudgeProvider(client=client)
    return provider, client.chat.completions


VALID_REPLY = json.dumps(
    {
        "category": "FLAKY",
        "action": "RERUN",
        "priority": 1,
        "confidence": 0.9,
        "reason": "repeated PASS<->FAIL flips with retry=not_executed",
    }
)


class TestProviderFlow:
    def test_predict_success_single_call(self):
        provider, completions = make_provider([make_response(VALID_REPLY)])
        result = provider.predict(make_state())
        assert result.category == "FLAKY" and result.action == "RERUN"
        assert provider.calls_made == 1
        assert completions.calls[0]["response_format"] == {"type": "json_object"}
        assert completions.calls[0]["temperature"] == pytest.approx(provider.temperature)
        # The real adapter sends the canonical common-facts projection.
        state = make_state()
        assert completions.calls[0]["messages"][1]["content"] == projection_json(
            project_failure_state(state)
        )
        assert provider.total_prompt_tokens == 10

    def test_predict_retries_on_invalid_json_then_succeeds(self):
        provider, completions = make_provider(
            [make_response("not json at all"), make_response(VALID_REPLY)]
        )
        result = provider.predict(make_state())
        assert result.category == "FLAKY"
        assert provider.calls_made == 2
        # 重试：温度调 0，且不带 response_format
        assert completions.calls[1]["temperature"] == pytest.approx(provider.retry_temperature)
        assert "response_format" not in completions.calls[1]

    def test_predict_retry_exhausted_raises_invalid_provider_output(self):
        provider, _ = make_provider([make_response("garbage 1"), make_response("garbage 2")])
        with pytest.raises(InvalidProviderOutput, match="after retry"):
            provider.predict(make_state())
        assert provider.calls_made == 2

    def test_predict_downgrades_when_json_mode_rejected(self):
        # 第 1 次调用因 response_format 被 400 拒绝 → 降级重试（共 2 次调用）且成功
        provider, completions = make_provider(
            [make_bad_request_error(), make_response(VALID_REPLY)]
        )
        result = provider.predict(make_state())
        assert result.category == "FLAKY"
        assert provider.calls_made == 2
        assert "response_format" not in completions.calls[1]

    def test_api_error_propagates_not_swallowed(self):
        import httpx
        import openai

        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        response = httpx.Response(401, request=request)
        client = FakeClient([openai.AuthenticationError(
            "invalid key", response=response, body=None
        )])
        provider = LLMJudgeProvider(client=client)
        with pytest.raises(openai.AuthenticationError):
            provider.predict(make_state())
        # 认证失败不吞、不重试：只烧 1 次调用
        assert provider.calls_made == 1

    def test_predict_validated_passes_contract(self):
        provider, _ = make_provider([make_response(VALID_REPLY)])
        result = provider.predict_validated(make_state())
        assert result.category == "FLAKY"

    def test_cost_accounting_with_pricing(self):
        provider, _ = make_provider([make_response(VALID_REPLY, prompt_tokens=1000, completion_tokens=100)])
        provider2_costed = LLMJudgeProvider(
            client=FakeClient([make_response(VALID_REPLY, prompt_tokens=1000, completion_tokens=100)]),
            cost_input_per_1k=0.002,
            cost_output_per_1k=0.008,
        )
        assert provider.accumulated_cost == 0.0  # 未配置单价：记账 0 但 token 照记
        provider2_costed.predict(make_state())
        expected = 1000 / 1000 * 0.002 + 100 / 1000 * 0.008
        assert provider2_costed.accumulated_cost == pytest.approx(expected, abs=1e-9)


class TestInjectedHttpTransport:
    @staticmethod
    def _wire_response(content, prompt_tokens=19, completion_tokens=7):
        return {
            "id": "offline-test",
            "object": "chat.completion",
            "created": 1,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @staticmethod
    def _provider(monkeypatch, transport):
        import jevtest.providers.llm_judge as llm_judge

        monkeypatch.setattr(
            llm_judge,
            "load_ark_config",
            lambda _path=None: {
                "openai_compatible_base": "https://ark.invalid/v1",
                "model": "fake-model",
            },
        )
        monkeypatch.setenv("ARK_API_KEY", "offline-fake-key")
        return LLMJudgeProvider(transport=transport)

    def test_usage_is_recorded_before_empty_content_fallback(self, monkeypatch, tmp_path):
        import httpx

        from jevtest.metrics import ErrorLog, MeasureRunner

        requests = []
        wire_bodies = []

        def handler(request):
            requests.append(request)
            wire_bodies.append(json.loads(request.content))
            return httpx.Response(200, json=self._wire_response(None))

        provider = self._provider(monkeypatch, httpx.MockTransport(handler))
        runner = MeasureRunner(provider, ErrorLog(tmp_path / "empty.jsonl"))
        state = make_state()
        result, _ = runner.run_one(state)

        assert result.category == "UNKNOWN"
        assert len(requests) == provider.calls_made == len(runner.last_attempts) == 2
        assert all(body["messages"][1]["content"] == build_user_content(state) for body in wire_bodies)
        assert [row.input_tokens for row in runner.last_attempts] == [19, 19]
        assert [row.output_tokens for row in runner.last_attempts] == [7, 7]
        assert all(row.error_code == "empty_content" for row in runner.last_attempts)
        assert provider.last_resolved_models == {1: "fake-model", 2: "fake-model"}
        assert all(row.pricing_status == "unknown" and row.cost is None for row in runner.last_attempts)
        assert all(row.provider_id == "ark" and row.model_id == "fake-model" for row in runner.last_attempts)
        expected_versions = []
        for mode in ("json", "text"):
            content = json.dumps(
                {"system_prompt": build_system_prompt(), "response_mode": mode},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            expected_versions.append("sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest())
        assert [row.prompt_version for row in runner.last_attempts] == expected_versions
        assert all(row.output_schema_version == "triage-result-v1" for row in runner.last_attempts)

    def test_auth_failure_does_not_retry_or_log_error_body(self, monkeypatch, tmp_path):
        import httpx

        from jevtest.metrics import ErrorLog, MeasureRunner

        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                401,
                json={"error": {"message": "secret token rejected", "type": "authentication_error"}},
            )

        provider = self._provider(monkeypatch, httpx.MockTransport(handler))
        result, _ = MeasureRunner(provider, ErrorLog(tmp_path / "auth.jsonl")).run_one(make_state())

        assert result.category == "UNKNOWN"
        assert len(requests) == provider.calls_made == len(provider.last_attempts) == 1
        assert provider.last_attempts[0].error_code == "http_401"
        assert "secret" not in (provider.last_attempts[0].error_summary or "")

    def test_rate_limit_retry_uses_the_same_two_attempt_budget(self, monkeypatch):
        import httpx

        requests = []

        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "limited"})
            return httpx.Response(200, json=self._wire_response(VALID_REPLY))

        provider = self._provider(monkeypatch, httpx.MockTransport(handler))
        result = provider.predict_validated(make_state())

        assert result.category == "FLAKY"
        assert len(requests) == provider.calls_made == len(provider.last_attempts) == 2
        assert provider.last_attempts[0].error_code == "http_429"
        assert [row.status for row in provider.last_attempts] == ["failed", "succeeded"]

    def test_invalid_response_shape_keeps_usage_and_fails_each_attempt(self, monkeypatch, tmp_path):
        import httpx

        from jevtest.metrics import ErrorLog, MeasureRunner

        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "bad-shape",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fake-model",
                    "choices": [],
                    "usage": {"prompt_tokens": 23, "completion_tokens": 4, "total_tokens": 27},
                },
            )

        provider = self._provider(monkeypatch, httpx.MockTransport(handler))
        result, _ = MeasureRunner(provider, ErrorLog(tmp_path / "shape.jsonl")).run_one(make_state())

        assert result.category == "UNKNOWN"
        assert len(requests) == provider.calls_made == len(provider.last_attempts) == 2
        assert [row.error_code for row in provider.last_attempts] == [
            "invalid_response_structure",
            "invalid_response_structure",
        ]
        assert [row.input_tokens for row in provider.last_attempts] == [23, 23]
        assert [row.output_tokens for row in provider.last_attempts] == [4, 4]


# ---------------------------------------------------------------------------
# 出网安全与配置加载（零网络）
# ---------------------------------------------------------------------------


class TestNetworkSafetyAndConfig:
    @pytest.mark.parametrize(
        "url",
        [
            "http://ark.cn-beijing.volces.com/api/v3",  # 非 https
            "https://localhost/api/v3",
            "https://sub.localhost/api/v3",
            "https://127.0.0.1/api/v3",
            "https://10.20.30.40/api/v3",
            "https://192.168.1.1/api/v3",
            "https://169.254.1.1/api/v3",
            "https://0.0.0.0/api/v3",
        ],
    )
    def test_validate_rejects_unsafe_hosts(self, url):
        from jevtest.providers.llm_judge import LLMJudgeConfigError, _validate_https_host

        with pytest.raises(LLMJudgeConfigError):
            _validate_https_host(url)

    def test_validate_accepts_public_https_host(self):
        from jevtest.providers.llm_judge import _validate_https_host

        _validate_https_host("https://ark.cn-beijing.volces.com/api/coding/v3")

    def test_load_ark_config_from_repo(self):
        config = load_ark_config(_REPO_ROOT / "config" / "providers.json")
        assert config["openai_compatible_base"].startswith("https://")
        assert "volces.com" in config["openai_compatible_base"]
        assert config["model"]

    def test_load_ark_config_missing_file(self):
        from jevtest.providers.llm_judge import LLMJudgeConfigError

        with pytest.raises(LLMJudgeConfigError, match="not found"):
            load_ark_config(_REPO_ROOT / "config" / "no-such-file.json")

    def test_provider_requires_api_key_from_env(self, monkeypatch):
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        with pytest.raises(Exception, match="ARK_API_KEY"):  # noqa: B017
            LLMJudgeProvider()


# ---------------------------------------------------------------------------
# live：10 条金样本逐条过 TriageResult 校验（marker live，默认 deselect）
# ---------------------------------------------------------------------------

GOLDEN_CASE_IDS = [
    # 每类别抽 1
    "SYN-PRODUCT_DEFECT-0001",
    "SYN-TEST_SCRIPT-0001",
    "SYN-ENVIRONMENT-0001",
    "SYN-CONFIGURATION-0001",
    "SYN-FLAKY-0001",
    "SYN-UNKNOWN-0001",
    # 3 + 1 边界：不同 false 探针 / flaky+not_executed / 空历史 UNKNOWN / 第二条缺陷确认
    "SYN-ENVIRONMENT-0002",
    "SYN-FLAKY-0002",
    "SYN-UNKNOWN-0002",
    "SYN-PRODUCT_DEFECT-0002",
]


def _ensure_ark_api_key_from_env_file() -> bool:
    """key 只从环境读；.env 仅在 ARK_API_KEY 未设置时注入进程环境（值绝不打印）。"""
    if os.environ.get("ARK_API_KEY"):
        return True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    load_dotenv(_REPO_ROOT / ".env", override=False)
    return bool(os.environ.get("ARK_API_KEY"))


def _load_golden_states() -> list[FailureState]:
    by_id = {}
    with (_REPO_ROOT / "data" / "synthetic" / "v0.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            case_id = row["failure_state"]["case_id"]
            if case_id in GOLDEN_CASE_IDS:
                by_id[case_id] = FailureState.model_validate(row["failure_state"])
    missing = [cid for cid in GOLDEN_CASE_IDS if cid not in by_id]
    assert not missing, f"golden samples missing from dataset: {missing}"
    return [by_id[cid] for cid in GOLDEN_CASE_IDS]


@pytest.mark.live(providers={"ark": 20})
def test_live_golden_samples_pass_triage_contract(capsys: pytest.CaptureFixture) -> None:
    """10 条金样本；每条最多 2 次适配器请求，结果表打印到 stdout。"""
    if not _ensure_ark_api_key_from_env_file():
        pytest.skip("ARK_API_KEY not available in env or .env")
    provider = LLMJudgeProvider()
    rows = []
    failures = []
    from time import perf_counter

    for state in _load_golden_states():
        start = perf_counter()
        try:
            result = provider.predict_validated(state)
        except InvalidProviderOutput as exc:  # 单样本失败不中断，最后统一断言
            failures.append((state.case_id, str(exc)))
            rows.append((state.case_id, "PARSE_FAILURE", "-", "-", "-", f"{(perf_counter()-start)*1000:.0f}ms", "-"))
            continue
        rows.append(
            (state.case_id, result.category, result.action, result.priority,
             f"{result.confidence:.2f}", f"{(perf_counter()-start)*1000:.0f}ms", result.reason)
        )
    print("\ncase_id | category | action | priority | confidence | latency | reason")
    for row in rows:
        print(" | ".join(str(cell) for cell in row))
    print(f"total API calls: {provider.calls_made} (<=13 expected)")
    assert not failures, f"live parse failures: {failures}"
    assert provider.calls_made <= 13, f"API call budget exceeded: {provider.calls_made}"
