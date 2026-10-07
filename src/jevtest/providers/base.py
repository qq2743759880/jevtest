"""Provider ABC with a per-invocation attempt ledger."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from jevtest.execution_contracts import AttemptRecordV1
from jevtest.models import FailureState, TriageResult


class InvalidProviderOutput(Exception):
    """Provider output violates the TriageResult contract; raw output is retained if known."""

    def __init__(self, message: str, *, raw_output: Any = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class Provider(ABC):
    """Uniform provider interface plus per-request accounting helpers."""

    arm_name: str
    is_mock: bool = False
    provider_id: str | None = None
    model_id: str | None = None
    prompt_version: str | None = None

    def reset_attempts(self) -> None:
        """Clear the per-request ledger before a new prediction."""
        self.last_attempts: list[AttemptRecordV1] = []
        self.last_resolved_models: dict[int, str] = {}

    def record_resolved_model(self, model: Any) -> None:
        """Record a model identifier only when the provider returned one."""
        if isinstance(model, str) and model.strip() and len(model) <= 200:
            self.last_resolved_models[len(self.last_attempts) + 1] = model

    def record_attempt(
        self,
        *,
        status: Literal["succeeded", "failed", "skipped"],
        latency_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: float | None = None,
        cost_currency: str | None = None,
        pricing_status: Literal["no_external_cost", "mock", "known", "unknown"] | None = None,
        error_code: str | None = None,
        error_summary: str | None = None,
        invoked: bool = True,
    ) -> AttemptRecordV1:
        """Append one attempt. Error summaries should never include credentials or raw bodies."""
        if not hasattr(self, "last_attempts"):
            self.reset_attempts()
        stage = self.arm_name
        if stage not in {"rule", "jev", "llm", "cascade", "human"}:
            raise ValueError(f"unsupported attempt stage: {stage!r}")
        if pricing_status is None:
            if self.is_mock:
                pricing_status, cost = "mock", 0.0
            elif stage == "rule":
                pricing_status, cost = "no_external_cost", 0.0
            else:
                pricing_status, cost = "unknown", None
        row = AttemptRecordV1(
            stage=stage,
            attempt_index=len(self.last_attempts) + 1,
            invoked=invoked,
            provider_mock=self.is_mock,
            provider_id=self.provider_id or self.arm_name,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            status=status,
            latency_ms=max(0.0, float(latency_ms)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            cost_currency=cost_currency,
            pricing_status=pricing_status,
            error_code=error_code,
            error_summary=(error_summary or "")[:500] or None,
        )
        self.last_attempts.append(row)
        return row

    @abstractmethod
    def predict(self, state: FailureState) -> TriageResult:
        """Single-sample prediction."""

    def predict_with_budget(self, state: FailureState, max_attempts: int) -> TriageResult:
        """Predict within a per-invocation request budget.

        Most providers make exactly one call and need no override. Adapters with internal
        retries override this method so an outer cascade budget can cap each wire request.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        return self.predict(state)

    def predict_validated(
        self, state: FailureState, *, max_attempts: int | None = None
    ) -> TriageResult:
        """Predict and enforce TriageResult validation without losing attempt evidence."""
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.reset_attempts()
        started = perf_counter()
        try:
            output = (
                self.predict(state)
                if max_attempts is None
                else self.predict_with_budget(state, max_attempts)
            )
        except InvalidProviderOutput:
            if not self.last_attempts:
                self.record_attempt(
                    status="failed",
                    latency_ms=(perf_counter() - started) * 1000.0,
                    error_code="invalid_provider_output",
                )
            raise
        except ValidationError as exc:
            if not self.last_attempts:
                self.record_attempt(
                    status="failed",
                    latency_ms=(perf_counter() - started) * 1000.0,
                    error_code="invalid_provider_output",
                )
            raise InvalidProviderOutput(
                f"provider arm={self.arm_name!r} raised ValidationError inside predict: {exc}"
            ) from exc
        except BaseException as exc:
            if not self.last_attempts:
                self.record_attempt(
                    status="failed",
                    latency_ms=(perf_counter() - started) * 1000.0,
                    error_code=type(exc).__name__.lower(),
                )
            raise

        try:
            # Pydantic does not revalidate existing model instances unless converted to a mapping.
            payload = output.model_dump() if isinstance(output, BaseModel) else output
            result = TriageResult.model_validate(payload)
        except ValidationError as exc:
            if not self.last_attempts:
                self.record_attempt(
                    status="failed",
                    latency_ms=(perf_counter() - started) * 1000.0,
                    error_code="invalid_provider_output",
                )
            raise InvalidProviderOutput(
                f"provider arm={self.arm_name!r} returned contract-violating output: {exc}",
                raw_output=payload,
            ) from exc
        if not self.last_attempts:
            self.record_attempt(
                status="succeeded",
                latency_ms=(perf_counter() - started) * 1000.0,
            )
        return result
