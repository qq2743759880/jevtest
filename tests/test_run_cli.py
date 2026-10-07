"""T12 `jevtest run` 子命令单测（非网络）。

覆盖任务 GWT：
- (a) 合法 FailureState JSON → exit 0，stdout 含三问字段（类别/下一步/优先级）与 reason；
- (b) 非法 JSON / 缺字段 → exit 2（E_SCHEMA_INVALID_INPUT）且 stderr 指明违规字段；
- (c) --arm jev-mock 端到端（零网络、确定性）；
- (d) --arm llm / --arm jev 在无 key 环境 → exit 4（E_PROVIDER_AUTH）与提示，无崩溃栈；
- (e) --out PATH 同时落盘 TriageResult JSON；
- (f) 同 seed 分层抽样字节级一致（README 公开条件①的复现性证据之一）。

live 测试（marker live，真实 API，预算 <=1 次调用）在文件尾部。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevtest.cli import main as cli_main
from jevtest.benchmark import load_states

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPO_ROOT / "data" / "synthetic" / "v0.jsonl"


def _failure_dict() -> dict:
    """一条合法 FailureState dict（AP_reachable=false → Rule 臂 R2 命中，确定性）。"""
    return {
        "case_id": "T12_RUN_CLI_001",
        "result": "FAIL",
        "previous_runs": ["PASS", "PASS", "FAIL"],
        "device": "WA6638",
        "sw_version": "V600R024C00",
        "assertion": "association timeout",
        "syslog_excerpt": "%CAPWAP/3/STA association timeout",
        "env": {"AP_reachable": False, "AC_reachable": True},
        "retry": "not_executed",
        "provenance": {
            "dataset_version": "synthetic-v0",
            "synthetic": True,
            "source": "test fixture",
        },
    }


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "failure.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# (a) 合法输入
# ---------------------------------------------------------------------------


def test_run_rule_valid_exit0_and_three_questions(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = _write(tmp_path, _failure_dict())
    rc = cli_main(["run", str(path), "--arm", "rule"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "case: T12_RUN_CLI_001" in out
    # 三问字段 + confidence + reason
    for field in ("类别", "下一步", "优先级", "置信度", "依据"):
        assert field in out
    assert "R2: env[AP_reachable]=false" in out  # reason 含规则 ID + 触发字段值
    assert "ENVIRONMENT" in out and "CHECK_ENVIRONMENT" in out
    assert "arm=rule" in out and "rule_version=0" in out and "latency" in out


def test_run_rule_writes_out_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = _write(tmp_path, _failure_dict())
    out_path = tmp_path / "result.json"
    rc = cli_main(["run", str(path), "--arm", "rule", "--out", str(out_path)])
    assert rc == 0
    capsys.readouterr()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    for key in ("category", "action", "priority", "confidence", "reason"):
        assert key in payload
    assert payload["category"] == "ENVIRONMENT"
    assert 1 <= payload["priority"] <= 5
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["reason"]


def test_run_default_arm_is_rule(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """默认零依赖可用：不带 --arm 也能跑（=rule）。"""
    path = _write(tmp_path, _failure_dict())
    rc = cli_main(["run", str(path)])
    assert rc == 0
    assert "arm=rule" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (b) 非法输入 → E_SCHEMA_INVALID_INPUT, exit 2
# ---------------------------------------------------------------------------


def test_run_invalid_json_exit2(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    rc = cli_main(["run", str(path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "E_SCHEMA_INVALID_INPUT" in err
    assert "Traceback" not in err


def test_run_missing_field_exit2_names_field(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    payload = _failure_dict()
    del payload["assertion"]
    path = _write(tmp_path, payload)
    rc = cli_main(["run", str(path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "E_SCHEMA_INVALID_INPUT" in err
    assert "assertion" in err


def test_run_bad_enum_exit2_names_field(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    payload = _failure_dict()
    payload["retry"] = "sometimes"  # 非冻结枚举值
    path = _write(tmp_path, payload)
    rc = cli_main(["run", str(path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "retry" in err
    assert "E_SCHEMA_INVALID_INPUT" in err


def test_run_extra_field_exit2(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    payload = _failure_dict()
    payload["ground_truth"] = "PRODUCT_DEFECT"  # 契约 additionalProperties=false
    path = _write(tmp_path, payload)
    rc = cli_main(["run", str(path)])
    assert rc == 2
    assert "ground_truth" in capsys.readouterr().err


def test_run_missing_file_exit2(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    rc = cli_main(["run", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "E_SCHEMA_INVALID_INPUT" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# (c) jev-mock 端到端（零网络）
# ---------------------------------------------------------------------------


def test_run_jev_mock_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = _write(tmp_path, _failure_dict())
    rc = cli_main(["run", str(path), "--arm", "jev-mock", "--out", str(tmp_path / "mock.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "arm=jev-mock" in out
    for field in ("类别", "下一步", "优先级", "置信度", "依据"):
        assert field in out
    assert "cost_state: mock" in out
    written = json.loads((tmp_path / "mock.json").read_text(encoding="utf-8"))
    assert written["category"] == "ENVIRONMENT"  # mock 启发式：env false → ENVIRONMENT
    assert written["reason"].startswith("jev-mock heuristic")


# ---------------------------------------------------------------------------
# (d) 无 key → E_PROVIDER_AUTH, exit 4（monkeypatch 清 env 模拟，零网络）
# ---------------------------------------------------------------------------


def test_run_llm_without_key_exit4(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    from jevtest.providers import jev_live

    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.setattr(jev_live, "_load_env_file", lambda: None)  # 隔离真实 .env
    path = _write(tmp_path, _failure_dict())
    rc = cli_main(["run", str(path), "--arm", "llm"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "E_PROVIDER_AUTH" in err
    assert "ARK_API_KEY" in err
    assert ".env" in err  # 指引：key 放 .env
    assert "Traceback" not in err


def test_run_jev_without_key_exit4(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    from jevtest.providers import jev_live

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jev_live, "_load_env_file", lambda: None)
    path = _write(tmp_path, _failure_dict())
    rc = cli_main(["run", str(path), "--arm", "jev"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "E_PROVIDER_AUTH" in err
    assert "TYPESAFE_API_KEY" in err
    assert "Traceback" not in err


def test_run_invalid_arm_rejected(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["run", str(_write(tmp_path, _failure_dict())), "--arm", "gpt"])
    assert excinfo.value.code == 2  # argparse choices


# ---------------------------------------------------------------------------
# (f) 同 seed 分层抽样字节级一致（README 公开条件①证据）
# ---------------------------------------------------------------------------


def test_same_seed_stratified_sampling_byte_identical():
    """同 seed 两次分层抽样，选中的 (expected, state) 序列逐字节一致。"""
    runs = [
        load_states(_DATASET_PATH, limit=12, sampling="stratified", seed=7)
        for _ in range(2)
    ]
    dumps = [
        json.dumps(
            [{"expected": rec["expected"], "state": rec["state"].model_dump()} for rec in run],
            sort_keys=True,
            ensure_ascii=False,
        )
        for run in runs
    ]
    assert dumps[0] == dumps[1]
    # 分层：12 条样本覆盖多个类别（数据集按类别分块，first 只会取到单一类别块）
    labels = {rec["expected"] for rec in runs[0]}
    assert len(labels) >= 5


def test_stratified_covers_more_categories_than_first():
    """--sampling stratified 的用法警示依据：first 取到单一类别块，stratified 覆盖多类别。"""
    first = load_states(_DATASET_PATH, limit=6, sampling="first")
    stratified = load_states(_DATASET_PATH, limit=6, sampling="stratified", seed=0)
    assert len({rec["expected"] for rec in first}) == 1
    assert len({rec["expected"] for rec in stratified}) >= 5


# ---------------------------------------------------------------------------
# live：单样本；Jev 有 1 次重试，因此最多 2 次适配器请求。
# ---------------------------------------------------------------------------


@pytest.mark.live(providers={"typesafe": 2})
def test_run_jev_live_single_case(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    rc = cli_main(["run", str(_REPO_ROOT / "examples" / "failure-example.json"), "--arm", "jev"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "arm=jev" in out
    for field in ("类别", "下一步", "优先级", "置信度", "依据"):
        assert field in out
