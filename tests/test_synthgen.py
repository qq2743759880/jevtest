"""T04 合成数据生成器测试：可复现性 / 契约合法性 / 类别配额 / 因果性抽查。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from jevtest.models import FailureState
from jevtest.synthgen import CATEGORIES, DATASET_VERSION, generate_dataset, summarize

SEED = 20260924

TEST_SCRIPT_MARKERS = ("hardcoded", "comparison bug")
CONFIGURATION_MARKERS = ("mismatch",)


def _flip_count(runs: list[str]) -> int:
    return sum(1 for prev, curr in zip(runs, runs[1:]) if prev != curr)


def _iter_lines(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _by_category(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {category: [] for category in CATEGORIES}
    for rec in _iter_lines(path):
        grouped[rec["expected"]].append(rec)
    return grouped


def _has_env_false(env: dict) -> bool:
    return any(value is False for value in env.values())


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synthgen") / "v0.jsonl"
    return generate_dataset(SEED, str(out))


# (a) Given 生成命令带 seed，Then 重跑字节级一致（可复现性 GWT）
def test_same_seed_byte_identical(tmp_path: Path) -> None:
    first = generate_dataset(SEED, str(tmp_path / "a.jsonl"))
    second = generate_dataset(SEED, str(tmp_path / "b.jsonl"))
    assert first.read_bytes() == second.read_bytes()


# (d) Given 不同 seed，Then 产出不同数据
def test_different_seed_produces_different_bytes(tmp_path: Path) -> None:
    baseline = generate_dataset(SEED, str(tmp_path / "base.jsonl"))
    other = generate_dataset(SEED + 1, str(tmp_path / "other.jsonl"))
    assert baseline.read_bytes() != other.read_bytes()


# (b) Given 任一样本，When 过 FailureState 模型校验，Then 通过
def test_every_line_validates_against_failure_state(dataset: Path) -> None:
    for rec in _iter_lines(dataset):
        failure_state = FailureState.model_validate(rec["failure_state"])
        assert failure_state.result == "FAIL"
        assert failure_state.provenance.dataset_version == DATASET_VERSION
        assert failure_state.provenance.synthetic is True
        assert rec["expected"] in CATEGORIES


# (c) Given 数据集统计，Then 类别分布达到配额（6 类各 >=25 + UNKNOWN 边界 >=20）
def test_category_quotas(dataset: Path) -> None:
    stats = summarize(dataset)
    assert stats["total"] == 200
    assert sum(stats["categories"].values()) == stats["total"]
    for category in CATEGORIES:
        assert stats["categories"][category] >= 25, category
    assert stats["categories"]["UNKNOWN"] >= 25 + 20


def test_summarize_matches_file(dataset: Path) -> None:
    stats = summarize(dataset)
    counted = sum(1 for rec in _iter_lines(dataset) if rec["expected"] in CATEGORIES)
    assert stats["total"] == counted


# (e) 因果性抽查：每个类别抽 2 条，断言其特征字段确实由因果规则构造
def test_flaky_samples_show_flips_and_no_retry(dataset: Path) -> None:
    for rec in _by_category(dataset)["FLAKY"][:2]:
        fs = rec["failure_state"]
        assert _flip_count(fs["previous_runs"]) >= 2
        assert fs["retry"] == "not_executed"


def test_product_defect_samples_all_fail_and_retry_failed(dataset: Path) -> None:
    for rec in _by_category(dataset)["PRODUCT_DEFECT"][:2]:
        fs = rec["failure_state"]
        assert fs["previous_runs"], "previous_runs 全 FAIL 要求非空"
        assert set(fs["previous_runs"]) == {"FAIL"}
        assert fs["retry"] == "failed"
        assert not _has_env_false(fs["env"])


def test_environment_samples_have_false_probe(dataset: Path) -> None:
    for rec in _by_category(dataset)["ENVIRONMENT"][:2]:
        assert _has_env_false(rec["failure_state"]["env"])


def test_configuration_samples_env_true_with_config_assertion(dataset: Path) -> None:
    for rec in _by_category(dataset)["CONFIGURATION"][:2]:
        fs = rec["failure_state"]
        assert not _has_env_false(fs["env"])
        assert any(marker in fs["assertion"] for marker in CONFIGURATION_MARKERS)


def test_test_script_samples_show_self_contradictory_assertion(dataset: Path) -> None:
    for rec in _by_category(dataset)["TEST_SCRIPT"][:2]:
        fs = rec["failure_state"]
        assert any(marker in fs["assertion"] for marker in TEST_SCRIPT_MARKERS)


def test_unknown_samples_lack_all_causal_markers(dataset: Path) -> None:
    # UNKNOWN 边界样本不得同时满足任何一类的因果特征
    for rec in _by_category(dataset)["UNKNOWN"][:2]:
        fs = rec["failure_state"]
        is_flaky = _flip_count(fs["previous_runs"]) >= 2 and fs["retry"] == "not_executed"
        is_product_defect = bool(fs["previous_runs"]) and set(fs["previous_runs"]) == {"FAIL"} and fs["retry"] == "failed"
        is_environment = _has_env_false(fs["env"])
        assert not is_flaky
        assert not is_product_defect
        assert not is_environment


def test_reproducible_via_fixed_seed_default_quota(tmp_path: Path) -> None:
    # 最小配额参数（25+20）也应满足：6 类各 >=25、UNKNOWN >=45、共 >=170
    out = generate_dataset(SEED, str(tmp_path / "min.jsonl"), per_category=25, unknown_count=20)
    stats = summarize(out)
    assert stats["total"] == 170
    for category in CATEGORIES:
        assert stats["categories"][category] >= 25, category
    assert stats["categories"]["UNKNOWN"] >= 45
