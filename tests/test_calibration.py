"""T10 tests：校准预演（合成预演模式）。

两层：
- 非网络单测（默认跑）：bin_calibration 分箱数学（手算断言）、语义分离字段、
  preview 入口确实用 stratified 抽样（monkeypatch 断言调用参数）。
- live 预演（`-m live` 才跑）：真实跑 60 条 stratified 样本（每类 10 条），
  60 次 Jev 调用（预算 60 次），落盘 data/calibration/preview-<seed>.jsonl。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevtest import calibration
from jevtest.calibration import bin_calibration, run_calibration_preview
from jevtest.models import FailureState, Provenance, TriageResult
from jevtest.providers.base import Provider

DATA_V0 = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "v0.jsonl"


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------


def _rec(confidence: float, correct: bool, sharpness: float | None = None) -> dict:
    return {
        "confidence": confidence,
        "confidence_sharpness": sharpness if sharpness is not None else confidence,
        "correct": correct,
    }


def _state(case_id: str = "SYN-TEST-0001") -> FailureState:
    return FailureState(
        case_id=case_id,
        result="FAIL",
        previous_runs=["FAIL"],
        device="dev-1",
        sw_version="V600R024C00",
        assertion="assertion failed",
        syslog_excerpt="log line",
        env={"AP_reachable": True},
        retry="failed",
        provenance=Provenance(
            dataset_version="synthetic-v0", source="test", synthetic=True
        ),
    )


class _StubProvider(Provider):
    """离线 stub：固定 confidence/sharpness 的假 Jev 臂（MeasureRunner 可计量）。"""

    arm_name = "jev"
    is_mock = False

    def __init__(self, confidence: float = 0.95, sharpness: float = 0.9) -> None:
        self._confidence = confidence
        self._sharpness = sharpness
        self.calls = 0

    def predict(self, state: FailureState) -> TriageResult:
        self.calls += 1
        return TriageResult(
            category="PRODUCT_DEFECT",
            action="ASSIGN_PRODUCT_DEVELOPER",
            priority=4,
            confidence=self._confidence,
            reason="stub",
            probs={
                "category": {"PRODUCT_DEFECT": self._confidence, "TEST_SCRIPT": 1 - self._confidence}
            },
            confidence_sharpness=self._sharpness,
        )


# ---------------------------------------------------------------------------
# bin_calibration：分箱数学（手算断言）
# ---------------------------------------------------------------------------


class TestBinCalibration:
    def test_ten_perfect_high_confidence_records(self) -> None:
        """10 条全 confidence=0.95 且全对 → 该箱 accuracy=1.0、count=10。"""
        records = [_rec(0.95, True) for _ in range(10)]
        bins = bin_calibration(records, n_bins=5)
        assert len(bins) == 5
        top = bins[4]
        assert top["bin"] == "0.8-1.0"
        assert top["count"] == 10
        assert top["accuracy"] == 1.0
        # 其余箱空
        assert all(b["count"] == 0 for b in bins[:4])
        assert all(b["accuracy"] is None for b in bins[:4])

    def test_hand_computed_mixed_bins(self) -> None:
        """手算混合分布：箱边界左闭右开，末箱右闭。"""
        records = [
            _rec(0.05, True),   # 箱 0-0.2
            _rec(0.15, False),  # 箱 0-0.2
            _rec(0.20, True),   # 0.2 落 0.2-0.4（左闭右开）
            _rec(0.30, False),  # 箱 0.2-0.4
            _rec(0.50, True),   # 箱 0.4-0.6
            _rec(0.70, True),   # 箱 0.6-0.8
            _rec(1.00, True),   # 1.0 落末箱 0.8-1.0（右闭）
        ]
        bins = bin_calibration(records, n_bins=5)
        assert (bins[0]["count"], bins[0]["accuracy"]) == (2, 0.5)
        assert (bins[1]["count"], bins[1]["accuracy"]) == (2, 0.5)
        assert (bins[2]["count"], bins[2]["accuracy"]) == (1, 1.0)
        assert (bins[3]["count"], bins[3]["accuracy"]) == (1, 1.0)
        assert (bins[4]["count"], bins[4]["accuracy"]) == (1, 1.0)

    def test_sharpness_binning_is_independent(self) -> None:
        """sharpness 单独分箱：与 confidence 分箱互不影响（语义分离，PRD §10.2）。"""
        records = [
            _rec(0.95, True, sharpness=0.30),
            _rec(0.10, False, sharpness=0.90),
            _rec(0.85, False, sharpness=0.95),
        ]
        by_conf = bin_calibration(records, n_bins=5, value_field="confidence")
        by_sharp = bin_calibration(records, n_bins=5, value_field="confidence_sharpness")
        assert by_conf[4]["count"] == 2  # 0.95, 0.85
        assert by_conf[4]["accuracy"] == 0.5
        assert by_sharp[4]["count"] == 2  # 0.90, 0.95
        assert by_sharp[4]["accuracy"] == 0.0
        # confidence 箱内平均 sharpness 也随之可算：0.8-1.0 箱 mean_sharpness=(0.30+0.95)/2
        assert by_conf[4]["mean_sharpness"] == pytest.approx(0.625)

    def test_none_value_records_excluded_not_guessed(self) -> None:
        """confidence_sharpness 缺失的记录不计入 sharpness 分箱（如实留空）。"""
        records = [
            {"confidence": 0.9, "confidence_sharpness": None, "correct": True},
            {"confidence": 0.9, "confidence_sharpness": 0.8, "correct": False},
        ]
        by_sharp = bin_calibration(records, n_bins=5, value_field="confidence_sharpness")
        assert by_sharp[4]["count"] == 1
        assert by_sharp[4]["accuracy"] == 0.0

    def test_invalid_n_bins_rejected(self) -> None:
        with pytest.raises(ValueError):
            bin_calibration([_rec(0.5, True)], n_bins=0)


# ---------------------------------------------------------------------------
# run_calibration_preview：stratified 抽样确实被使用（monkeypatch 断言调用参数）
# ---------------------------------------------------------------------------


class TestPreviewUsesStratifiedSampling:
    def test_preview_calls_load_states_with_stratified(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """preview 入口必须 load_states(..., sampling='stratified', seed=...)（守护栏 3）。"""
        captured: dict = {}

        def fake_load_states(path, limit=None, sampling="first", seed=0):
            captured["limit"] = limit
            captured["sampling"] = sampling
            captured["seed"] = seed
            # 返回两条假记录（ stratified 本应返回的形态：expected + state）
            return [
                {"expected": "PRODUCT_DEFECT", "state": _state("SYN-A")},
                {"expected": "TEST_SCRIPT", "state": _state("SYN-B")},
            ]

        monkeypatch.setattr(calibration, "load_states", fake_load_states)
        stub = _StubProvider()
        out = run_calibration_preview(
            DATA_V0,
            seed=7,
            per_category=10,
            out_dir=tmp_path,
            provider_factory=lambda: stub,
        )
        # 6 类 × 10 条 = 60 的配额上限；stratified + seed 必须逐字传下去
        assert captured["sampling"] == "stratified"
        assert captured["seed"] == 7
        assert captured["limit"] == 60
        assert stub.calls == 2
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
        assert len(rows) == 2
        # 逐样本字段齐全（语义分字段：confidence 与 confidence_sharpness 并存）
        row = rows[0]
        for field in calibration.RECORD_FIELDS:
            assert field in row
        assert row["confidence"] == 0.95
        assert row["confidence_sharpness"] == 0.9
        assert row["correct"] is True  # stub 固定 PRODUCT_DEFECT，第一条 expected 同类


# ---------------------------------------------------------------------------
# live：60 条 stratified 预演；默认 Jev 每条最多 2 次适配器请求（1 次重试）。
# ---------------------------------------------------------------------------


@pytest.mark.live(providers={"typesafe": 120})
class TestLivePreview:
    def test_live_preview_sixty_stratified_samples(self) -> None:
        # 产物落项目内 data/calibration（与 benchmark 的 runs/ 同为项目相对路径约定），
        # 该 JSONL 就是 docs/synthetic-calibration-preview.md 的数据源（可复算）。
        out = run_calibration_preview(DATA_V0, seed=20260924, per_category=10)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
        assert len(rows) == 60
        # 全类别覆盖：6 类 × 10 条
        by_expected: dict[str, int] = {}
        for row in rows:
            by_expected[row["expected"]] = by_expected.get(row["expected"], 0) + 1
        assert all(v == 10 for v in by_expected.values())
        assert len(by_expected) == 6
        # 字段完整性：confidence∈[0,1]，sharpness 同域且分字段
        for row in rows:
            assert 0.0 <= row["confidence"] <= 1.0
            if row["confidence_sharpness"] is not None:
                assert 0.0 <= row["confidence_sharpness"] <= 1.0
