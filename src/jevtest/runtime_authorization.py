"""Explicit, bounded runtime authorization for the live advisory profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class LiveAuthorizationError(ValueError):
    """Live execution was requested without a valid bounded runtime grant."""


@dataclass(frozen=True, slots=True)
class LiveRuntimeAuthorization:
    """In-process proof that live advisory was explicitly enabled at startup.

    The first version intentionally fixes one application attempt per external
    provider. Both SDK adapters have their own retries disabled under this
    profile, so this also bounds provider wire calls to two per request.
    """

    profile: Literal["live-advisory"]
    explicit_opt_in: bool
    provider_allowlist: frozenset[str]
    jev_attempt_budget: int
    llm_attempt_budget: int
    deadline_ms: int

    def __post_init__(self) -> None:
        if self.profile != "live-advisory" or not self.explicit_opt_in:
            raise LiveAuthorizationError("explicit live-advisory opt-in is required")
        if self.provider_allowlist != frozenset({"jev", "llm"}):
            raise LiveAuthorizationError("live-advisory requires the jev,llm provider allowlist")
        if self.jev_attempt_budget != 1 or self.llm_attempt_budget != 1:
            raise LiveAuthorizationError("live-advisory permits one attempt per provider")
        if not 1 <= self.deadline_ms <= 60_000:
            raise LiveAuthorizationError("live-advisory deadline must be between 1 and 60000 ms")

    @property
    def request_provider_attempt_budget(self) -> int:
        return self.jev_attempt_budget + self.llm_attempt_budget

    def budget_for(self, stage: str) -> int:
        if stage not in self.provider_allowlist:
            return 0
        return self.jev_attempt_budget if stage == "jev" else self.llm_attempt_budget


def authorize_live_advisory(
    *,
    explicit_opt_in: bool,
    deadline_ms: int = 30_000,
) -> LiveRuntimeAuthorization:
    """Build the only supported live grant after a CLI opt-in flag is present."""

    return LiveRuntimeAuthorization(
        profile="live-advisory",
        explicit_opt_in=explicit_opt_in,
        provider_allowlist=frozenset({"jev", "llm"}),
        jev_attempt_budget=1,
        llm_attempt_budget=1,
        deadline_ms=deadline_ms,
    )
