"""Pytest safety defaults for tests that can contact live providers.

Live tests are skipped unless the caller explicitly opts in, names every
provider, and grants a per-provider call budget. API keys and dotenv files are
credentials, never authorization. Ordinary tests get a network guard that
allows loopback, local subprocess stdio, and in-memory/fake transports.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Mapping
from typing import Any

import pytest


_OPT_IN_ENV = "JEVTEST_LIVE_OPT_IN"
_PROVIDERS_ENV = "JEVTEST_LIVE_PROVIDER_ALLOWLIST"
_BUDGETS_ENV = "JEVTEST_LIVE_CALL_BUDGETS"
_LIVE_SELECTION_BUDGET_ERROR: str | None = None


class OutboundNetworkBlocked(OSError):
    """Raised when an offline test attempts to use a non-loopback network."""


def _parse_csv(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _parse_budgets(raw: str | None) -> dict[str, int]:
    budgets: dict[str, int] = {}
    if not raw:
        return budgets
    for entry in raw.split(","):
        provider, separator, value = entry.partition("=")
        provider = provider.strip().lower()
        if not separator or not provider or provider in budgets:
            return {}
        try:
            budget = int(value.strip())
        except ValueError:
            return {}
        if budget < 1:
            return {}
        budgets[provider] = budget
    return budgets


def _live_authorization_error(item: pytest.Item) -> str | None:
    """Return a fail-closed skip reason unless this exact live test is authorized."""
    if os.environ.get(_OPT_IN_ENV) != "1":
        return f"live test requires explicit {_OPT_IN_ENV}=1"

    marker = item.get_closest_marker("live")
    requirements = marker.kwargs.get("providers") if marker is not None else None
    if not isinstance(requirements, Mapping) or not requirements:
        return "live test has no declared provider call budget"

    normalized: dict[str, int] = {}
    for provider, calls in requirements.items():
        if not isinstance(provider, str) or not isinstance(calls, int) or calls < 1:
            return "live test has an invalid provider call budget declaration"
        normalized[provider.strip().lower()] = calls

    allowlist = _parse_csv(os.environ.get(_PROVIDERS_ENV))
    missing_providers = normalized.keys() - allowlist
    if missing_providers:
        return (
            f"live test requires provider allowlist entries: "
            f"{', '.join(sorted(missing_providers))}"
        )

    budgets = _parse_budgets(os.environ.get(_BUDGETS_ENV))
    insufficient = [
        provider
        for provider, required_calls in normalized.items()
        if budgets.get(provider, 0) < required_calls
    ]
    if insufficient:
        return (
            "live test requires call budgets at least "
            + ", ".join(f"{provider}={normalized[provider]}" for provider in sorted(insufficient))
        )
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live(providers={provider: max_calls}): live provider test; requires explicit opt-in, "
        "provider allowlist, and sufficient per-provider call budgets",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("live") is None:
        return
    reason = _live_authorization_error(item)
    if reason is None:
        reason = _LIVE_SELECTION_BUDGET_ERROR
    if reason is not None:
        pytest.skip(reason)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Require the caller's budget to cover all selected live tests cumulatively."""
    global _LIVE_SELECTION_BUDGET_ERROR
    _LIVE_SELECTION_BUDGET_ERROR = None
    if os.environ.get(_OPT_IN_ENV) != "1":
        return

    live_items = [
        item for item in session.items if item.get_closest_marker("live") is not None
    ]
    if not live_items:
        return

    selected_requirements: dict[str, int] = {}
    for item in live_items:
        marker = item.get_closest_marker("live")
        requirements = marker.kwargs.get("providers") if marker is not None else None
        if not isinstance(requirements, Mapping):
            continue  # Individual setup skips tests without a valid declaration.
        for provider, calls in requirements.items():
            if isinstance(provider, str) and isinstance(calls, int) and calls > 0:
                normalized_provider = provider.strip().lower()
                selected_requirements[normalized_provider] = (
                    selected_requirements.get(normalized_provider, 0) + calls
                )

    budgets = _parse_budgets(os.environ.get(_BUDGETS_ENV))
    insufficient = [
        provider
        for provider, required_calls in selected_requirements.items()
        if budgets.get(provider, 0) < required_calls
    ]
    if insufficient:
        _LIVE_SELECTION_BUDGET_ERROR = (
            "selected live tests require aggregate call budgets at least "
            + ", ".join(
                f"{provider}={selected_requirements[provider]}"
                for provider in sorted(insufficient)
            )
        )


def _is_loopback_host(host: Any) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if not isinstance(host, str):
        return False
    normalized = host.strip().lower().rstrip(".")
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(".localhost"):
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_ip_literal(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if not isinstance(host, str):
        return False
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        ipaddress.ip_address(normalized)
        return True
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _block_non_loopback_network(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Block accidental egress for the test session; authorized live items opt out."""
    is_authorized_live = (
        request.node.get_closest_marker("live") is not None
        and _live_authorization_error(request.node) is None
    )
    if is_authorized_live:
        return

    def blocked(reason: str) -> OutboundNetworkBlocked:
        return OutboundNetworkBlocked(f"non-loopback network blocked by pytest: {reason}")

    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto

    def guarded_gethostbyname(host: Any) -> str:
        if _is_loopback_host(host):
            return "127.0.0.1"
        if _is_ip_literal(host):
            try:
                normalized = host.decode("ascii", errors="ignore") if isinstance(host, bytes) else host
                parsed = ipaddress.ip_address(normalized.strip("[]"))
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and parsed.version == 4:
                return str(parsed)
        return "8.8.8.8"

    def guarded_gethostbyname_ex(host: Any):
        address = guarded_gethostbyname(host)
        canonical = "localhost" if _is_loopback_host(host) else "synthetic.invalid"
        return canonical, [], [address]

    def guarded_gethostbyaddr(address: str):
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return "synthetic.invalid", [], ["8.8.8.8"]
        if parsed.is_loopback:
            return "localhost", [], [str(parsed)]
        return "synthetic.invalid", [], [str(parsed)]

    def guarded_getnameinfo(sockaddr: Any, flags: int):
        host = sockaddr[0] if isinstance(sockaddr, tuple) and sockaddr else sockaddr
        service = str(sockaddr[1]) if isinstance(sockaddr, tuple) and len(sockaddr) > 1 else "0"
        if _is_loopback_host(host) or (_is_ip_literal(host) and ipaddress.ip_address(host).is_loopback):
            return str(host), service
        return "8.8.8.8", service

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any):
        if host is None or _is_ip_literal(host):
            return original_getaddrinfo(host, port, *args, **kwargs)
        normalized_host = host.decode("ascii", errors="ignore") if isinstance(host, bytes) else host
        if isinstance(normalized_host, str) and (
            normalized_host.strip().lower().rstrip(".") in {"localhost", "localhost.localdomain"}
            or normalized_host.strip().lower().rstrip(".").endswith(".localhost")
        ):
            family = kwargs.get("family", args[0] if args else socket.AF_UNSPEC)
            loopback_host = "::1" if family == socket.AF_INET6 else "127.0.0.1"
            return original_getaddrinfo(loopback_host, port, *args, **kwargs)
        # Keep DNS itself offline too. If a test later tries to connect, the
        # socket guard below rejects this synthetic address.
        family = kwargs.get("family", args[0] if args else socket.AF_UNSPEC)
        fake_host = "2001:4860:4860::8888" if family == socket.AF_INET6 else "8.8.8.8"
        return original_getaddrinfo(fake_host, port, *args, **kwargs)

    def guarded_connect(sock: socket.socket, address: Any):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0] if isinstance(address, tuple) and address else address
            if not _is_loopback_host(host):
                raise blocked(f"connect denied for {host!r}")
        return original_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0] if isinstance(address, tuple) and address else address
            if not _is_loopback_host(host):
                raise blocked(f"connect_ex denied for {host!r}")
        return original_connect_ex(sock, address)

    def guarded_sendto(sock: socket.socket, data: Any, *args: Any):
        address = args[-1] if args and isinstance(args[-1], tuple) else None
        if sock.family in (socket.AF_INET, socket.AF_INET6) and address:
            host = address[0]
            if not _is_loopback_host(host):
                raise blocked(f"sendto denied for {host!r}")
        return original_sendto(sock, data, *args)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", guarded_gethostbyname)
    monkeypatch.setattr(socket, "gethostbyname_ex", guarded_gethostbyname_ex)
    monkeypatch.setattr(socket, "gethostbyaddr", guarded_gethostbyaddr)
    monkeypatch.setattr(socket, "getnameinfo", guarded_getnameinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", guarded_sendto)
