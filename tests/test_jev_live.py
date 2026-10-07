"""T06 tests：JevLiveProvider。

两层：
- 非网络单测（默认跑）：state 序列化确定性、答案映射纯函数（喂构造的假 API 响应）、
  priority clamp、出网安全校验、key 加载纪律。
- live 契约符合性（`-m live` 才跑，真实 API 调用 ≤13 次）：10 条金样本
  （data/synthetic/v0.jsonl 每类别 1 条）+ 3 条边界（空 env / 单翻转 / 未收录错误码），
  逐条 predict_validated 过 TriageResult 校验、reason 非空、confidence∈[0,1]。
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest

from jevtest.models import FailureState, Provenance, TriageResult
from jevtest.input_projection import project_failure_state, projection_sha256
from jevtest.metrics import ErrorLog, MeasureRunner
from jevtest.providers.base import InvalidProviderOutput
from jevtest.providers.jev_live import (
    JevLiveProvider,
    map_answers_to_triage,
    score_to_priority,
    serialize_state,
    validate_https_url,
)

DATA_V0 = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "v0.jsonl"

CATEGORIES = [
    "PRODUCT_DEFECT",
    "TEST_SCRIPT",
    "ENVIRONMENT",
    "CONFIGURATION",
    "FLAKY",
    "UNKNOWN",
]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "live: real TypeSafe API calls (≤13); run explicitly with -m live"
    )


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------


def _state_from_row(row: dict) -> FailureState:
    return FailureState.model_validate(row["failure_state"])


def _load_gold_samples() -> list[FailureState]:
    """v0.jsonl 每类别取第 1 条，共 6 条；顺序按 CATEGORIES 冻结。"""
    by_category: dict[str, FailureState] = {}
    with DATA_V0.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            expected = row["expected"]
            if expected in CATEGORIES and expected not in by_category:
                by_category[expected] = _state_from_row(row)
    missing = [c for c in CATEGORIES if c not in by_category]
    assert not missing, f"dataset missing categories: {missing}"
    return [by_category[c] for c in CATEGORIES]


def _boundary_state(**overrides) -> FailureState:
    base = dict(
        case_id="SYN-BOUNDARY-0001",
        result="FAIL",
        previous_runs=["PASS", "PASS", "FAIL"],
        device="WA6638",
        sw_version="V600R024C00",
        assertion="association timeout (802.11 status 1)",
        syslog_excerpt="%CAPWAP/3/STA 00:E0:FC:1D:DA:8E failure: association timeout",
        env={"AP_reachable": True, "AC_reachable": True},
        retry="not_executed",
        provenance=Provenance(
            dataset_version="synthetic-v0", synthetic=True, source="test-fixture"
        ),
    )
    base.update(overrides)
    return FailureState.model_validate(base)


def _fake_choice(choice: str, probabilities: dict[str, float], confidence: float) -> dict:
    return {
        "type": "choice",
        "choice": choice,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def _fake_answers(
    category: str = "FLAKY",
    action: str = "RERUN",
    score: float = 2.13,
) -> dict:
    """S1 探测实调结构等价的假 API 响应（Score 0 起始 legend）。"""
    return {
        "category": _fake_choice(
            category,
            {c: 0.52 if c == category else 0.096 for c in CATEGORIES},
            0.41,
        ),
        "action": _fake_choice(
            action,
            {"RERUN": 0.43, "COLLECT_MORE_LOGS": 0.14, "CHECK_ENVIRONMENT": 0.14,
             "ASSIGN_TEST_OWNER": 0.01, "ASSIGN_PRODUCT_DEVELOPER": 0.03, "HUMAN_TRIAGE": 0.25},
            0.31,
        ),
        "priority": {
            "type": "score",
            "score": score,
            "confidence": 0.23,
            "legend": {str(i): f"{i + 1}: level" for i in range(5)},
            "probabilities": {str(i): 0.2 for i in range(5)},
        },
    }


# ---------------------------------------------------------------------------
# 非网络单测
# ---------------------------------------------------------------------------


class TestSerializeState:
    def test_deterministic_same_input_same_output(self) -> None:
        state = _boundary_state()
        assert serialize_state(state) == serialize_state(state)

    def test_env_keys_sorted_and_provenance_excluded(self) -> None:
        state = _boundary_state(
            env={"z_probe": True, "a_probe": False, "m_probe": "ok"},
        )
        payload = serialize_state(state)
        assert list(payload["env"]) == ["a_probe", "m_probe", "z_probe"]
        assert "provenance" not in payload
        assert "case_id" not in payload
        assert payload == project_failure_state(state).model_dump(mode="json")
        assert projection_sha256(project_failure_state(state))
        # 契约字段全在，字段名自明
        assert set(payload) == {
            "schema_version", "result", "previous_runs", "device", "sw_version",
            "assertion", "syslog_excerpt", "env", "retry",
        }


class TestProviderTransportAccounting:
    @staticmethod
    def _wire_body(category: str = "FLAKY") -> dict:
        return {
            "model": "fake-model",
            "usage": {"input_tokens": 17, "output_tokens": 8},
            "answers": _fake_answers(category=category),
        }

    def _provider(self, monkeypatch, transport, **kwargs):
        import jevtest.providers.jev_live as jev_live

        monkeypatch.setattr(jev_live, "validate_https_url", lambda url: url)
        return JevLiveProvider(
            api_key="offline-fake-key",
            base_url="https://example.invalid",
            max_retries=1,
            retry_backoff_s=0,
            transport=transport,
            **kwargs,
        )

    def test_repeated_invalid_wire_outputs_are_two_attempts_without_replay(
        self, monkeypatch, tmp_path
    ):
        import httpx2

        requests = []
        wire_bodies = []

        def handler(request):
            requests.append(request)
            wire_bodies.append(json.loads(request.content))
            return httpx2.Response(200, json=self._wire_body(category="NOT_A_CATEGORY"))

        provider = self._provider(monkeypatch, httpx2.MockTransport(handler))
        error_log = ErrorLog(tmp_path / "errors.jsonl")
        runner = MeasureRunner(provider, error_log)
        result, sample = runner.run_one(_boundary_state())

        assert result.category == "UNKNOWN"
        assert sample.parse_failure is True
        assert len(requests) == provider.calls_made == 2
        assert len(requests) == len(runner.last_attempts) == 2
        assert all(body["state"] == serialize_state(_boundary_state()) for body in wire_bodies)
        assert [row.input_tokens for row in runner.last_attempts] == [17, 17]
        assert [row.provider_id for row in runner.last_attempts] == ["typesafe", "typesafe"]
        assert [row.model_id for row in runner.last_attempts] == ["jev-latest", "jev-latest"]
        assert provider.last_resolved_models == {1: "fake-model", 2: "fake-model"}
        questions_json = json.dumps(
            wire_bodies[0]["questions"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        expected_version = "sha256:" + hashlib.sha256(questions_json.encode("utf-8")).hexdigest()
        assert [row.prompt_version for row in runner.last_attempts] == [expected_version, expected_version]
        assert all(row.output_schema_version == "triage-result-v1" for row in runner.last_attempts)
        assert all(row.status == "failed" for row in runner.last_attempts)
        assert all(row.pricing_status == "known" for row in runner.last_attempts)
        assert [row.cost for row in runner.last_attempts] == [0.000000714, 0.000000714]
        assert all(row.cost_currency == "USD" for row in runner.last_attempts)
        assert json.loads((tmp_path / "errors.jsonl").read_text(encoding="utf-8"))["raw_output"]["category"]["choice"] == "NOT_A_CATEGORY"

    def test_invalid_success_response_structure_retains_usage_and_retry_ledger(
        self, monkeypatch, tmp_path
    ):
        import httpx2

        requests = []

        def handler(request):
            requests.append(request)
            return httpx2.Response(
                200,
                json={
                    "model": "fake-model",
                    "usage": {"input_tokens": 19, "output_tokens": 6},
                    "answers": {"category": {"type": "choice", "choice": "FLAKY"}},
                },
            )

        provider = self._provider(monkeypatch, httpx2.MockTransport(handler))
        error_log = ErrorLog(tmp_path / "invalid-structure.jsonl")
        runner = MeasureRunner(provider, error_log)
        result, sample = runner.run_one(_boundary_state())

        assert result.category == "UNKNOWN"
        assert sample.parse_failure is True
        assert len(requests) == provider.calls_made == len(runner.last_attempts) == 2
        assert [row.error_code for row in runner.last_attempts] == [
            "invalid_response_structure",
            "invalid_response_structure",
        ]
        assert [row.input_tokens for row in runner.last_attempts] == [19, 19]
        assert [row.output_tokens for row in runner.last_attempts] == [6, 6]
        assert [row.cost for row in runner.last_attempts] == [0.000000798, 0.000000798]
        assert all(row.pricing_status == "known" for row in runner.last_attempts)
        assert all(row.cost_currency == "USD" for row in runner.last_attempts)
        assert all(row.status == "failed" for row in runner.last_attempts)
        logged = json.loads((tmp_path / "invalid-structure.jsonl").read_text(encoding="utf-8"))
        assert logged["raw_output"] == {
            "error_code": "invalid_response_structure",
            "raw_output_available": False,
        }

    def test_rate_limit_retries_within_one_total_budget(self, monkeypatch):
        import httpx2

        requests = []

        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx2.Response(429, headers={"retry-after-ms": "0"}, json={"error": "limited"})
            return httpx2.Response(200, json=self._wire_body())

        provider = self._provider(monkeypatch, httpx2.MockTransport(handler))
        result = provider.predict_validated(_boundary_state())

        assert result.category == "FLAKY"
        assert len(requests) == len(provider.last_attempts) == 2
        assert provider.last_attempts[0].error_code == "http_429"
        assert provider.last_attempts[0].status == "failed"
        assert provider.last_attempts[1].status == "succeeded"

    def test_auth_failure_does_not_retry(self, monkeypatch, tmp_path):
        import httpx2

        requests = []

        def handler(request):
            requests.append(request)
            return httpx2.Response(401, json={"error": "invalid credentials"})

        provider = self._provider(monkeypatch, httpx2.MockTransport(handler))
        result, _ = MeasureRunner(provider, ErrorLog(tmp_path / "auth.jsonl")).run_one(_boundary_state())

        assert result.category == "UNKNOWN"
        assert len(requests) == len(provider.last_attempts) == 1
        assert provider.last_attempts[0].error_code == "http_401"
        assert provider.last_attempts[0].pricing_status == "unknown"

    def test_timeout_retries_once_then_retains_both_failures(self, monkeypatch, tmp_path):
        import httpx2

        requests = []

        def handler(request):
            requests.append(request)
            raise httpx2.ReadTimeout("simulated timeout", request=request)

        provider = self._provider(monkeypatch, httpx2.MockTransport(handler))
        result, _ = MeasureRunner(provider, ErrorLog(tmp_path / "timeout.jsonl")).run_one(_boundary_state())

        assert result.category == "UNKNOWN"
        assert len(requests) == len(provider.last_attempts) == 2
        assert [row.error_code for row in provider.last_attempts] == ["timeout", "timeout"]

    def test_cancellation_propagates_with_completed_attempt_preserved(self, monkeypatch):
        import asyncio
        import httpx2

        requests = []

        def handler(request):
            requests.append(request)
            raise asyncio.CancelledError()

        provider = self._provider(monkeypatch, httpx2.MockTransport(handler))
        runner = MeasureRunner(provider)
        with pytest.raises(asyncio.CancelledError):
            runner.run_one(_boundary_state())

        assert len(requests) == len(runner.last_attempts) == 1
        assert runner.last_attempts[0].status == "failed"
        assert runner.last_attempts[0].error_code == "cancellederror"


class TestScoreToPriority:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (2.13, 3),   # S1 探测实调值
            (1.43, 2),   # 官方文档级间落值示例
            (0.0, 1),    # 下界
            (4.0, 5),    # 上界
            (4.9, 5),    # 越上界 clamp
            (-3.7, 1),   # 越下界 clamp
            (2.5, 3),    # 级间中点：round(2.5)=2（banker's 舍入）+1 = 3
        ],
    )
    def test_zero_indexed_legend_offset_and_clamp(self, score: float, expected: int) -> None:
        legend = {str(i): f"level {i + 1}" for i in range(5)}
        assert score_to_priority(score, legend) == expected

    def test_no_legend_defaults_to_zero_indexed(self) -> None:
        assert score_to_priority(2.13, None) == 3

    def test_one_indexed_legend(self) -> None:
        legend = {str(i): f"level {i}" for i in range(1, 6)}
        assert score_to_priority(2.13, legend) == 2

    def test_result_always_in_1_5(self) -> None:
        for raw in [i / 10 for i in range(-50, 55)]:
            assert 1 <= score_to_priority(raw, None) <= 5


class TestMapAnswers:
    def test_full_mapping_fields(self) -> None:
        result = map_answers_to_triage(_fake_answers())
        assert isinstance(result, TriageResult)
        assert result.category == "FLAKY"
        assert result.action == "RERUN"
        assert result.priority == 3
        # confidence = Probs[Value]（Q1 正确概率），sharpness = Jev 原生 Confidence——分字段
        assert result.confidence == pytest.approx(0.52)
        assert result.confidence_sharpness == pytest.approx(0.41)
        assert result.probs is not None
        assert result.probs["category"]["FLAKY"] == pytest.approx(0.52)
        assert result.probs["priority_score_raw"] == pytest.approx(2.13)
        assert result.probs["priority"]["3"] == pytest.approx(0.2)

    def test_reason_nonempty_and_contains_all_three_answers(self) -> None:
        result = map_answers_to_triage(_fake_answers())
        assert result.reason
        assert "Q1=FLAKY" in result.reason
        assert "Q2=RERUN" in result.reason
        assert "Q3=3" in result.reason

    def test_reason_under_200_chars(self) -> None:
        assert len(map_answers_to_triage(_fake_answers()).reason) <= 200

    def test_missing_question_raises(self) -> None:
        answers = _fake_answers()
        del answers["action"]
        with pytest.raises(InvalidProviderOutput):
            map_answers_to_triage(answers)

    def test_choice_outside_frozen_enum_raises(self) -> None:
        answers = _fake_answers(category="NOT_A_CATEGORY")
        with pytest.raises(InvalidProviderOutput):
            map_answers_to_triage(answers)

    def test_missing_probabilities_raises(self) -> None:
        answers = _fake_answers()
        answers["category"] = {"type": "choice", "choice": "FLAKY", "confidence": 0.4}
        with pytest.raises(InvalidProviderOutput):
            map_answers_to_triage(answers)

    def test_missing_native_confidence_raises(self) -> None:
        answers = _fake_answers()
        answers["category"] = {
            "type": "choice",
            "choice": "FLAKY",
            "probabilities": {c: 0.52 if c == "FLAKY" else 0.096 for c in CATEGORIES},
        }
        with pytest.raises(InvalidProviderOutput):
            map_answers_to_triage(answers)

    def test_accepts_pydantic_answer_objects(self) -> None:
        """喂 SDK 原生答案对象（live 路径实际形态）也应映射一致。"""
        from typesafe_sdk import ChoiceAnswer, ScoreAnswer

        fake = _fake_answers()
        # SDK ScoreAnswer 的 legend/probabilities 键是 int（API 返回字符串键，SDK 模型转 int）
        fake["priority"]["probabilities"] = {int(k): v for k, v in fake["priority"]["probabilities"].items()}
        fake["priority"]["legend"] = {int(k): v for k, v in fake["priority"]["legend"].items()}
        answers = {
            "category": ChoiceAnswer.model_validate(fake["category"]),
            "action": ChoiceAnswer.model_validate(fake["action"]),
            "priority": ScoreAnswer.model_validate(fake["priority"]),
        }
        result = map_answers_to_triage(answers)
        assert result.category == "FLAKY"
        assert result.priority == 3


class TestOutboundSafety:
    @pytest.mark.parametrize(
        "url",
        [
            "http://api.typesafe.ai",                # 非 https
            "https://localhost",                      # localhost
            "https://sub.localhost",                  # localhost 子域
            "https://127.0.0.1/v1",                   # 环回
            "https://10.20.30.40/v1",                 # 私有
            "https://192.168.1.1/v1",                 # 私有
            "https://169.254.1.1/v1",                 # 链路本地
            "https://0.0.0.0/v1",                     # 未指定
        ],
    )
    def test_rejects_unsafe_urls(self, url: str) -> None:
        with pytest.raises(ValueError):
            validate_https_url(url)

    def test_accepts_public_ip_literal(self) -> None:
        # 字面公网 IP 不需要 DNS，离线可测；真实 host 的 https 校验由 live 路径覆盖
        assert validate_https_url("https://1.1.1.1") == "https://1.1.1.1"


class TestKeyDiscipline:
    def test_missing_key_raises_runtime_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)  # 无 .env 的目录
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        from jevtest.providers import jev_live

        # Pytest may place tmp_path below the checkout; prevent ancestor .env
        # discovery from making this missing-key test depend on local secrets.
        monkeypatch.setattr(jev_live, "_load_env_file", lambda: None)

        with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY is not set"):
            jev_live.load_typesafe_api_key()

    def test_key_read_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-for-unittest-only")
        from jevtest.providers.jev_live import load_typesafe_api_key

        assert load_typesafe_api_key() == "dummy-for-unittest-only"


# ---------------------------------------------------------------------------
# live 契约符合性（真实 API，≤13 次调用）
# ---------------------------------------------------------------------------

LIVE_CALLS_EXPECTED_MAX = 13


@pytest.fixture(scope="module")
def live_provider() -> "object":
    pytest.importorskip("typesafe_sdk")
    from jevtest.providers.jev_live import JevLiveProvider

    provider = JevLiveProvider(max_retries=2, timeout_s=60.0)
    yield provider
    provider.close()


def _assert_contract(result: TriageResult) -> None:
    assert isinstance(result, TriageResult)
    assert result.category in CATEGORIES
    assert result.priority in (1, 2, 3, 4, 5)
    assert 0.0 <= result.confidence <= 1.0
    assert result.reason and result.reason.strip()
    assert len(result.reason) <= 200
    if result.confidence_sharpness is not None:
        assert 0.0 <= result.confidence_sharpness <= 1.0


def _live_predict(live_provider, sample_name: str, state: FailureState) -> TriageResult:
    """predict_validated + 契约断言；设 JEV_LIVE_RESULTS 时把逐样本结果落 JSONL（验收记录用）。"""
    result = live_provider.predict_validated(state)
    _assert_contract(result)
    results_path = os.environ.get("JEV_LIVE_RESULTS")
    if results_path:
        # 路径约束：只允许写进 runs/ 目录，防 env 注入任意路径（路径穿越）
        out = Path(results_path).resolve()
        allowed_root = Path(__file__).resolve().parent.parent / "runs"
        if not out.is_relative_to(allowed_root.resolve()):
            raise ValueError(f"JEV_LIVE_RESULTS must stay under {allowed_root}")
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "sample": sample_name,
            "case_id": state.case_id,
            "category": result.category,
            "action": result.action,
            "priority": result.priority,
            "confidence": result.confidence,
            "confidence_sharpness": result.confidence_sharpness,
            "reason": result.reason,
            "usage": getattr(live_provider, "last_usage", None),
        }
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


class TestLiveGoldSamples:
    """10 条金样本 = 数据集 7 条（每类别 1 条 + PRODUCT_DEFECT 第 2 条）+ 3 条边界。
    每个测试 1 次 predict；live fixture max_retries=2，因此每个测试最多 3 次 wire 请求。"""

    @pytest.mark.live(providers={"typesafe": 3})
    def test_gold_sample_01_product_defect(self, live_provider) -> None:
        _live_predict(live_provider, "gold01_PRODUCT_DEFECT", _load_gold_samples()[0])

    @pytest.mark.live(providers={"typesafe": 3})
    def test_gold_sample_02_test_script(self, live_provider) -> None:
        _live_predict(live_provider, "gold02_TEST_SCRIPT", _load_gold_samples()[1])

    @pytest.mark.live(providers={"typesafe": 3})
    def test_gold_sample_03_environment(self, live_provider) -> None:
        _live_predict(live_provider, "gold03_ENVIRONMENT", _load_gold_samples()[2])

    @pytest.mark.live(providers={"typesafe": 3})
    def test_gold_sample_04_configuration(self, live_provider) -> None:
        _live_predict(live_provider, "gold04_CONFIGURATION", _load_gold_samples()[3])

    @pytest.mark.live(providers={"typesafe": 3})
    def test_gold_sample_05_flaky(self, live_provider) -> None:
        _live_predict(live_provider, "gold05_FLAKY", _load_gold_samples()[4])

    @pytest.mark.live(providers={"typesafe": 3})
    def test_gold_sample_06_unknown(self, live_provider) -> None:
        _live_predict(live_provider, "gold06_UNKNOWN", _load_gold_samples()[5])

    @pytest.mark.live(providers={"typesafe": 3})
    def test_gold_sample_07_second_product_defect(self, live_provider) -> None:
        """PRODUCT_DEFECT 第 2 条（数据集该类第 2 行），凑满 10 条金样本。"""
        rows = [
            json.loads(line)
            for line in DATA_V0.open(encoding="utf-8")
            if json.loads(line)["expected"] == "PRODUCT_DEFECT"
        ]
        _live_predict(live_provider, "gold07_PRODUCT_DEFECT_2nd", _state_from_row(rows[1]))

    @pytest.mark.live(providers={"typesafe": 3})
    def test_boundary_empty_env(self, live_provider) -> None:
        state = _boundary_state(case_id="BOUNDARY-EMPTY-ENV", env={})
        _live_predict(live_provider, "boundary_empty_env", state)

    @pytest.mark.live(providers={"typesafe": 3})
    def test_boundary_single_flip(self, live_provider) -> None:
        state = _boundary_state(
            case_id="BOUNDARY-SINGLE-FLIP", previous_runs=["PASS", "FAIL"], retry="not_executed"
        )
        _live_predict(live_provider, "boundary_single_flip", state)

    @pytest.mark.live(providers={"typesafe": 3})
    def test_boundary_uncatalogued_error_code(self, live_provider) -> None:
        """未收录错误码语义：断言与日志带不存在于契约错误码表的私有错误码。"""
        state = _boundary_state(
            case_id="BOUNDARY-UNKNOWN-CODE",
            assertion="operation failed with undocumented error 0xE9999999",
            syslog_excerpt="%WLAN/7/STA 00:E0:FC:FF:FF:FF failure: internal error 0xE9999999 (code not in error-codes-v0 catalog)",
        )
        _live_predict(live_provider, "boundary_uncatalogued_error_code", state)
