"""Deterministic offline LLM stand-in used only for Cascade demonstrations."""

from __future__ import annotations

from jevtest.models import FailureState, TriageResult
from jevtest.providers.base import Provider


class OfflineLLMStubProvider(Provider):
    """An explicitly mocked provider that abstains and always requests human review."""

    arm_name = "llm"
    is_mock = True
    provider_id = "offline-llm-stub"
    model_id = "unknown-only-v1"
    prompt_version = "offline-stub-v1"

    def predict(self, state: FailureState) -> TriageResult:
        del state
        return TriageResult(
            category="UNKNOWN",
            action="HUMAN_TRIAGE",
            priority=3,
            confidence=0.0,
            reason="offline LLM stub abstained; human review required",
        )
