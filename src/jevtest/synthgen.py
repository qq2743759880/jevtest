"""T04 合成数据生成器：按类别因果生成 M1 数据集（seed 可复现）。

设计约束：
- 裁定 2（合成真实性边界三原则）：合成器规则只来自可公开验证来源——
  协议行为取自 IEEE 802.11-2020 / IEEE 802.1Q / RFC 5415 / RFC 2131，
  设备型号取自任务单认可的华为公开产品列表。逐条来源见 docs/synthetic-sources.md。
- 按类别因果生成：ground truth 写在 JSONL 行顶层 `expected` 字段
  （FailureState 契约 additionalProperties=false，标签与 failure_state 并列），
  各类别的特征字段由因果规则构造，而非随机贴标签。
- 可复现：全程 local random.Random(seed)；禁用全局 random / 时间 / 哈希随机化
  （迭代集合一律先排序）；JSON 序列化 sort_keys 固定键序。
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from jevtest.models import FailureState, Provenance

GENERATOR_VERSION = "0.1.0"
DATASET_VERSION = "synthetic-v0"

CATEGORIES: tuple[str, ...] = (
    "PRODUCT_DEFECT",
    "TEST_SCRIPT",
    "ENVIRONMENT",
    "CONFIGURATION",
    "FLAKY",
    "UNKNOWN",
)

# --- 公开来源的常量池（来源清单见 docs/synthetic-sources.md） -----------------

# 设备型号：华为 WLAN AP 公开产品列表（任务单认可的公开型号：WA6520/WA6638/WA6320 等）
_DEVICES: tuple[str, ...] = ("WA6520", "WA6638", "WA6320", "WA6120", "AirEngine5761-11")
# 软件版本：华为 V200Rxxx / V600Rxxx 公开版本命名格式
_SW_VERSIONS: tuple[str, ...] = ("V200R023C00", "V200R024C00", "V600R024C00")
# MAC OUI：00:E0:FC 为 IEEE OUI 注册表中 Huawei 公开登记的前缀
_MAC_OUI = "00:E0:FC"

# IEEE 802.11-2020 公开的关联状态/原因码（Table 9-45/9-46）
_80211_STATUS_1 = "802.11 status 1 (unspecified failure)"
_80211_STATUS_17 = "802.11 status 17 (AP unable to handle additional STA)"
_80211_REASON_15 = "802.11 reason 15 (4-way handshake timeout)"

# 环境探针键：契约 env 允许布尔/字符串映射（failure-state.schema.json §env）
_ENV_PROBES: tuple[str, ...] = (
    "AP_reachable",
    "AC_reachable",
    "DHCP_server_reachable",
    "RADIUS_server_reachable",
    "gateway_reachable",
)
_ENV_NOISE_KEYS: tuple[str, ...] = ("NTP_server_reachable", "SMB_server_reachable")
_CONTROLLER_IPS: tuple[str, ...] = ("10.20.30.41", "10.20.30.42", "10.20.30.43")

# 真实协议失败语义（PRODUCT_DEFECT / FLAKY / ENVIRONMENT 共用的底层失败表现）
_ASSOC_TIMEOUT = f"association timeout ({_80211_STATUS_1}, IEEE 802.11-2020 11.3.2)"
_EAPOL_TIMEOUT = f"EAPOL 4-way handshake timeout ({_80211_REASON_15}, IEEE 802.11-2020 12.7.6.4)"
_CAPWAP_ECHO_TIMEOUT = "CAPWAP control channel down: echo timeout (RFC 5415 7.2)"
_DHCP_NO_OFFER = "DHCP DISCOVER retransmit exhausted, no OFFER (RFC 2131 4.4.1)"

_PRODUCT_DEFECT_ASSERTIONS: tuple[str, ...] = (
    _ASSOC_TIMEOUT,
    _EAPOL_TIMEOUT,
    _CAPWAP_ECHO_TIMEOUT,
    _DHCP_NO_OFFER,
    "AP radio reset loop: {n} unexpected radio restarts within {m} min",
)

# TEST_SCRIPT：断言自身矛盾特征——期望值写死错误端口 / 陈旧规格 / 类型比较 bug
_TEST_SCRIPT_ASSERTIONS: tuple[str, ...] = (
    "assert mgmt_port == 8080, got 8443 — script hardcoded wrong port (spec says 8443)",
    "assert telnet_port == 2323, got 23 — script hardcoded wrong port",
    "assert service_vlan == {v}, got {v} — script expected dict stale, hardcoded wrong VLAN",
    "assert snmp_status == 'down', got 'up' — script hardcoded inverted expectation",
    "assert sta_count == {n}, got '{n}' — str/int comparison bug",
    "assert throughput == 940, got 940.0 — int/float comparison bug",
)

# CONFIGURATION：env 全 true 但配置类断言（VLAN/PSK/SSID/合规域 mismatch）
_CONFIGURATION_ASSERTIONS: tuple[str, ...] = (
    "VLAN mismatch: STA tagged VLAN {a}, AP service VLAN configured {b} (IEEE 802.1Q)",
    "WPA2-PSK passphrase mismatch — 4-way handshake timeout ({_r}, IEEE 802.11-2020 12.7.6.4)",
    "SSID mismatch: STA probes 'guest-ops', WLAN service configured 'guest-ops-v2'",
    "regulatory domain mismatch: AP country IE CN, AC domain US (IEEE 802.11-2020 11.9)",
    "802.1Q uplink trunk mismatch: allowed VLAN list excludes {v} (IEEE 802.1Q)",
)

# UNKNOWN 边界子型的模糊断言 / 未收录错误码
_UNKNOWN_VAGUE_ASSERTIONS: tuple[str, ...] = ("test failed", "step 3 aborted", "unexpected teardown")
_UNKNOWN_UNLISTED_CODES: tuple[int, ...] = (0x7F3A, 0x1B02, 0x5E11, 0x2C40)

_SYSLOG_FACILITIES: tuple[str, ...] = ("WLAN/4", "WLAN/3", "DOT1X/4", "CAPWAP/3")
_HOSTNAMES: tuple[str, ...] = ("wac-01", "wac-02", "agile-distributor-1")
_LOG_DATES: tuple[str, ...] = (
    "2026-09-24",
    "2026-09-23",
    "2026-09-22",
    "2026-09-21",
)


# --- 确定性噪声注入（非目标字段） -------------------------------------------


def _mac(rng: random.Random) -> str:
    return "{:s}:{:02X}:{:02X}:{:02X}".format(
        _MAC_OUI, rng.randrange(256), rng.randrange(256), rng.randrange(256)
    )


def _timestamp(rng: random.Random) -> str:
    # 固定日期池内随机时刻——不使用系统时间，保证可复现
    date = rng.choice(_LOG_DATES)
    return f"{date} {rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}"


def _syslog(rng: random.Random, event: str) -> str:
    """生成 2-4 行华为公开日志格式片段（%模块/级别/助记符，见 synthetic-sources.md）。"""
    lines: list[str] = []
    for _ in range(rng.randint(2, 4)):
        ts = _timestamp(rng)
        host = rng.choice(_HOSTNAMES)
        facility = rng.choice(_SYSLOG_FACILITIES)
        lines.append(f"{ts} {host} %{facility}/{event.split('|')[0]}: {event}")
    return "\n".join(lines)


def _env_all_true(rng: random.Random) -> dict[str, bool | str]:
    env: dict[str, bool | str] = {probe: True for probe in _ENV_PROBES}
    extra = rng.choice(_ENV_NOISE_KEYS)
    env[extra] = True
    env["controller_ip"] = rng.choice(_CONTROLLER_IPS)
    return env


def _env_with_false(rng: random.Random) -> dict[str, bool | str]:
    env = _env_all_true(rng)
    failing = rng.choice(_ENV_PROBES)
    env[failing] = False
    if rng.random() < 0.3:  # 30% 注入第二个故障探针
        remaining = tuple(p for p in _ENV_PROBES if p != failing)
        env[rng.choice(remaining)] = False
    return env


def _history_all_fail(rng: random.Random) -> list[str]:
    return ["FAIL"] * rng.randint(2, 6)


def _history_flips(rng: random.Random) -> list[str]:
    """交替 PASS/FAIL 历史：长度 4-8，翻转次数 = len-1 >= 3（满足 FLAKY >= 2）。"""
    length = rng.randint(4, 8)
    start = rng.choice(("PASS", "FAIL"))
    return [start if i % 2 == 0 else ("FAIL" if start == "PASS" else "PASS") for i in range(length)]


def _history_single_flip(rng: random.Random) -> list[str]:
    """恰好 1 次翻转（UNKNOWN 边界：不足以判 FLAKY）。"""
    length = rng.randint(2, 5)
    flip_at = rng.randrange(1, length)
    start = rng.choice(("PASS", "FAIL"))
    return [start if i < flip_at else ("FAIL" if start == "PASS" else "PASS") for i in range(length)]


# --- 分类别因果构造 ----------------------------------------------------------


def _build_product_defect(idx: int, rng: random.Random, seed: int) -> dict[str, Any]:
    """产品缺陷 = 真实错误语义断言 + previous_runs 全 FAIL + retry=failed（重跑仍失败）。"""
    assertion = rng.choice(_PRODUCT_DEFECT_ASSERTIONS).format(n=rng.randint(3, 9), m=rng.randint(5, 30))
    return _record(
        "PRODUCT_DEFECT",
        idx,
        rng,
        seed,
        assertion=assertion,
        previous_runs=_history_all_fail(rng),
        env=_env_all_true(rng),
        retry="failed",
    )


def _build_test_script(idx: int, rng: random.Random, seed: int) -> dict[str, Any]:
    """脚本缺陷 = 断言自身矛盾（期望值写死错误端口 / 陈旧规格 / 类型比较 bug）。"""
    assertion = rng.choice(_TEST_SCRIPT_ASSERTIONS).format(v=rng.choice((100, 200, 300)), n=rng.randint(8, 64))
    return _record(
        "TEST_SCRIPT",
        idx,
        rng,
        seed,
        assertion=assertion,
        previous_runs=_history_all_fail(rng),
        env=_env_all_true(rng),
        retry="failed",
    )


def _build_environment(idx: int, rng: random.Random, seed: int) -> dict[str, Any]:
    """环境问题 = env 布尔探针含 false（rule-baseline-v0 R2 因果字段）。"""
    assertion = rng.choice((_ASSOC_TIMEOUT, _EAPOL_TIMEOUT, _CAPWAP_ECHO_TIMEOUT, _DHCP_NO_OFFER))
    # 历史只用单翻转形态：避免与 FLAKY 因果特征（>=2 翻转 + not_executed）同现导致 ground truth 模糊
    return _record(
        "ENVIRONMENT",
        idx,
        rng,
        seed,
        assertion=assertion,
        previous_runs=_history_single_flip(rng),
        env=_env_with_false(rng),
        retry=rng.choice(("not_executed", "failed")),
    )


def _build_configuration(idx: int, rng: random.Random, seed: int) -> dict[str, Any]:
    """配置问题 = env 全 true + 配置类断言（mismatch 语义）。"""
    assertion = rng.choice(_CONFIGURATION_ASSERTIONS).format(
        a=rng.randint(10, 90), b=rng.randint(10, 90), v=rng.choice((100, 200, 300)), _r=_80211_REASON_15
    )
    # 同 ENVIRONMENT：历史限单翻转，避免与 FLAKY 因果特征同现
    return _record(
        "CONFIGURATION",
        idx,
        rng,
        seed,
        assertion=assertion,
        previous_runs=_history_single_flip(rng),
        env=_env_all_true(rng),
        retry=rng.choice(("not_executed", "failed")),
    )


def _build_flaky(idx: int, rng: random.Random, seed: int) -> dict[str, Any]:
    """Flaky = previous_runs >= 2 次翻转 + retry=not_executed（rule-baseline-v0 R1 因果字段）。"""
    assertion = rng.choice((_ASSOC_TIMEOUT, _EAPOL_TIMEOUT, "ping loss spike to 100% for 30s"))
    return _record(
        "FLAKY",
        idx,
        rng,
        seed,
        assertion=assertion,
        previous_runs=_history_flips(rng),
        env=_env_all_true(rng),
        retry="not_executed",
    )


def _build_unknown(idx: int, rng: random.Random, seed: int) -> dict[str, Any]:
    """UNKNOWN 边界 = 字段残缺 / 仅 1 次翻转 / 未收录错误码（均不满足其他类因果特征）。"""
    sub = idx % 3
    if sub == 0:  # 仅 1 次翻转 + not_executed：不够 FLAKY 的翻转数
        return _record(
            "UNKNOWN",
            idx,
            rng,
            seed,
            assertion=rng.choice(_UNKNOWN_VAGUE_ASSERTIONS + (_ASSOC_TIMEOUT,)),
            previous_runs=_history_single_flip(rng),
            env=_env_all_true(rng),
            retry="not_executed",
        )
    if sub == 1:  # 未收录错误码：不落入 error_codes_v0 表
        code = rng.choice(_UNKNOWN_UNLISTED_CODES)
        assertion = f"failure with unlisted code 0x{code:04X} (not in error_codes_v0 table)"
        return _record(
            "UNKNOWN",
            idx,
            rng,
            seed,
            assertion=assertion,
            previous_runs=rng.choice((["FAIL"], ["PASS", "FAIL"], ["PASS", "PASS", "FAIL"])),
            env=_env_all_true(rng),
            retry=rng.choice(("not_executed", "passed")),
        )
    # sub == 2: 字段残缺——syslog 缺失、env 探针几乎为空、断言语义模糊
    return _record(
        "UNKNOWN",
        idx,
        rng,
        seed,
        assertion=rng.choice(_UNKNOWN_VAGUE_ASSERTIONS),
        previous_runs=rng.choice(([], ["FAIL"], ["PASS", "FAIL"])),
        env=rng.choice(({}, {"AP_reachable": True})),
        retry=rng.choice(("not_executed", "passed")),
        syslog_excerpt="no syslog captured",
    )


_BUILDERS = {
    "PRODUCT_DEFECT": _build_product_defect,
    "TEST_SCRIPT": _build_test_script,
    "ENVIRONMENT": _build_environment,
    "CONFIGURATION": _build_configuration,
    "FLAKY": _build_flaky,
    "UNKNOWN": _build_unknown,
}


# --- 记录组装与数据集落盘 ----------------------------------------------------


def _record(  # noqa: PLR0913 - 各字段即因果特征，逐个显式传入
    category: str,
    idx: int,
    rng: random.Random,
    seed: int,
    *,
    assertion: str,
    previous_runs: list[str],
    env: dict[str, bool | str],
    retry: str,
    syslog_excerpt: str | None = None,
) -> dict[str, Any]:
    failure_state = FailureState(
        case_id=f"SYN-{category}-{idx:04d}",
        result="FAIL",
        previous_runs=previous_runs,
        device=rng.choice(_DEVICES),
        sw_version=rng.choice(_SW_VERSIONS),
        assertion=assertion,
        syslog_excerpt=syslog_excerpt if syslog_excerpt is not None else _syslog(rng, f"STA {_mac(rng)} failure: {assertion}"),
        env=env,
        retry=retry,  # type: ignore[arg-type]
        provenance=Provenance(
            dataset_version=DATASET_VERSION,
            synthetic=True,
            source=f"jevtest.synthgen:{GENERATOR_VERSION}:seed={seed}",
        ),
    )
    return {
        "expected": category,
        "failure_state": json.loads(failure_state.model_dump_json()),
    }


def generate_dataset(seed: int, out_path: str, per_category: int = 30, unknown_count: int = 20) -> Path:
    """生成合成数据集 JSONL。默认 6 类各 30 条 + UNKNOWN 边界追加 20 条 = 200 行。

    布局：每行 {"expected": "<ground truth>", "failure_state": {...}}，
    failure_state 严格过 FailureState 模型校验（契约冻结文本的运行时形态）。
    全程 random.Random(seed)，同 seed 字节级一致。
    """
    rng = random.Random(seed)
    lines: list[str] = []
    for category in CATEGORIES:
        count = per_category + (unknown_count if category == "UNKNOWN" else 0)
        builder = _BUILDERS[category]
        for idx in range(1, count + 1):
            rec = builder(idx, rng, seed)
            lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL 数据集（供 summarize 与消费方复用）。"""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize(path: str | Path) -> dict[str, Any]:
    """类别分布统计：{"total": int, "categories": {类别: 条数}}（键序固定可复现）。"""
    distribution: dict[str, int] = {category: 0 for category in CATEGORIES}
    total = 0
    for rec in iter_records(path):
        expected = rec["expected"]
        distribution[expected] = distribution.get(expected, 0) + 1
        total += 1
    return {"total": total, "categories": distribution}
