"""Regression tests for the pytest live-provider and network safety gates."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_DIR = Path(__file__).resolve().parent
_AUTH_ENV_NAMES = (
    "JEVTEST_LIVE_OPT_IN",
    "JEVTEST_LIVE_PROVIDER_ALLOWLIST",
    "JEVTEST_LIVE_CALL_BUDGETS",
)


def _run_live_probe(
    tmp_path: Path,
    *,
    mode: str,
    authorization: dict[str, str],
    probe_tests: int = 1,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    probe = _TESTS_DIR / f"_pytest_live_probe_{uuid.uuid4().hex}.py"
    sentinel = tmp_path / "live-probe-ran.txt"
    source = ["import os", "from pathlib import Path", "import pytest", ""]
    for index in range(probe_tests):
        name = "test_live_probe" if probe_tests == 1 else f"test_live_probe_{index}"
        source.extend(
            (
                "@pytest.mark.live(providers={'typesafe': 1})",
                f"def {name}():",
                "    path = Path(os.environ['JEVTEST_SENTINEL_PATH'])",
                "    with path.open('a', encoding='utf-8') as output:",
                f"        output.write('ran{index}\\n')",
                "",
            )
        )
    probe.write_text("\n".join(source), encoding="utf-8")

    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    for name in _AUTH_ENV_NAMES:
        env.pop(name, None)
    env.update(authorization)
    # Credentials alone must never activate the test. These are inert sentinels.
    env["TYPESAFE_API_KEY"] = "key-is-not-authorization"
    env["ARK_API_KEY"] = "key-is-not-authorization"
    env["JEVTEST_SENTINEL_PATH"] = str(sentinel)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if mode == "live-marker":
        command.extend(("-m", "live", str(probe)))
    elif mode == "direct-node":
        command.append(f"{probe}::test_live_probe")
    elif mode == "ordinary":
        command.append(str(probe))
    else:
        raise AssertionError(f"unknown probe mode: {mode}")

    try:
        result = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    finally:
        probe.unlink(missing_ok=True)
    return result, sentinel


@pytest.mark.parametrize("mode", ("ordinary", "live-marker", "direct-node"))
def test_live_node_skips_without_opt_in_even_when_keys_exist(tmp_path: Path, mode: str) -> None:
    result, sentinel = _run_live_probe(tmp_path, mode=mode, authorization={})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout
    assert not sentinel.exists(), "the live test body ran without explicit authorization"


@pytest.mark.parametrize(
    ("authorization", "should_run"),
    (
        ({"JEVTEST_LIVE_OPT_IN": "1"}, False),
        (
            {
                "JEVTEST_LIVE_OPT_IN": "1",
                "JEVTEST_LIVE_PROVIDER_ALLOWLIST": "typesafe",
            },
            False,
        ),
        (
            {
                "JEVTEST_LIVE_OPT_IN": "1",
                "JEVTEST_LIVE_PROVIDER_ALLOWLIST": "typesafe",
                "JEVTEST_LIVE_CALL_BUDGETS": "typesafe=1",
            },
            True,
        ),
        (
            {
                "JEVTEST_LIVE_OPT_IN": "1",
                "JEVTEST_LIVE_PROVIDER_ALLOWLIST": "other-provider",
                "JEVTEST_LIVE_CALL_BUDGETS": "typesafe=1",
            },
            False,
        ),
        (
            {
                "JEVTEST_LIVE_OPT_IN": "1",
                "JEVTEST_LIVE_PROVIDER_ALLOWLIST": "typesafe",
                "JEVTEST_LIVE_CALL_BUDGETS": "typesafe=0",
            },
            False,
        ),
    ),
)
def test_live_opt_in_requires_provider_and_call_budget(
    tmp_path: Path, authorization: dict[str, str], should_run: bool
) -> None:
    result, sentinel = _run_live_probe(tmp_path, mode="direct-node", authorization=authorization)

    assert result.returncode == 0, result.stdout + result.stderr
    if should_run:
        assert "1 passed" in result.stdout
        assert sentinel.read_text(encoding="utf-8").splitlines() == ["ran0"]
    else:
        assert "1 skipped" in result.stdout
        assert not sentinel.exists()


def test_live_selection_budget_covers_all_selected_nodes(tmp_path: Path) -> None:
    authorization = {
        "JEVTEST_LIVE_OPT_IN": "1",
        "JEVTEST_LIVE_PROVIDER_ALLOWLIST": "typesafe",
        "JEVTEST_LIVE_CALL_BUDGETS": "typesafe=1",
    }
    result, sentinel = _run_live_probe(
        tmp_path, mode="ordinary", authorization=authorization, probe_tests=2
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 skipped" in result.stdout
    assert not sentinel.exists()

    authorization["JEVTEST_LIVE_CALL_BUDGETS"] = "typesafe=2"
    result, sentinel = _run_live_probe(
        tmp_path, mode="ordinary", authorization=authorization, probe_tests=2
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert sentinel.read_text(encoding="utf-8").splitlines() == ["ran0", "ran1"]


def test_default_network_guard_blocks_egress_but_keeps_local_transports() -> None:
    # Host lookup is synthetic/offline; connecting to the synthetic address is blocked.
    assert socket.getaddrinfo("example.invalid", 443)
    assert socket.gethostbyname("example.invalid") == "8.8.8.8"
    assert socket.gethostbyname_ex("example.invalid")[2] == ["8.8.8.8"]
    assert socket.gethostbyaddr("203.0.113.9")[0] == "synthetic.invalid"
    assert socket.getnameinfo(("203.0.113.9", 443), 0) == ("8.8.8.8", "443")
    assert socket.gethostbyname("localhost") == "127.0.0.1"
    assert socket.gethostbyaddr("127.0.0.1")[0] == "localhost"
    assert socket.getnameinfo(("127.0.0.1", 80), 0) == ("127.0.0.1", "80")
    with pytest.raises(OSError, match="non-loopback network blocked by pytest"):
        socket.create_connection(("example.invalid", 443), timeout=0.1)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with udp:
        with pytest.raises(OSError, match="non-loopback network blocked by pytest"):
            udp.sendto(b"blocked", ("203.0.113.9", 53))

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.settimeout(2)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    received: list[bytes] = []

    def serve_one() -> None:
        with listener:
            connection, _ = listener.accept()
            with connection:
                received.append(connection.recv(32))
                connection.sendall(b"local-ok")

    server = threading.Thread(target=serve_one, daemon=True)
    server.start()
    with socket.create_connection(listener.getsockname(), timeout=2) as client:
        client.sendall(b"loopback")
        assert client.recv(32) == b"local-ok"
    server.join(timeout=2)
    assert not server.is_alive()
    assert received == [b"loopback"]

    import httpx

    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="fake-ok"))
    with httpx.Client(transport=transport) as client:
        response = client.get("https://fake.invalid/provider")
    assert response.text == "fake-ok"

    stdio = subprocess.run(
        [sys.executable, "-c", "print('stdio-ok')"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert stdio.returncode == 0
    assert stdio.stdout.strip() == "stdio-ok"
