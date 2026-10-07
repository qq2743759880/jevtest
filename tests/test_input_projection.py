from __future__ import annotations

from jevtest.input_projection import project_failure_state, projection_json, projection_sha256
from jevtest.models import FailureState


def _state(**overrides: object) -> FailureState:
    fields: dict[str, object] = {
        "case_id": "gold-PRODUCT_DEFECT-001",
        "result": "FAIL",
        "previous_runs": ["FAIL"],
        "device": "WA6638",
        "sw_version": "V200R022C01",
        "assertion": "association timeout",
        "syslog_excerpt": "STA association timeout",
        "env": {"AP_reachable": True, "AC_reachable": True},
        "retry": "failed",
        "provenance": {
            "dataset_version": "gold-v1",
            "synthetic": False,
            "source": "label=PRODUCT_DEFECT",
        },
    }
    fields.update(overrides)
    return FailureState.model_validate(fields)


def test_projection_omits_case_and_dataset_labels():
    view = project_failure_state(_state())
    payload = view.model_dump(mode="json")
    assert "case_id" not in payload
    assert "provenance" not in payload
    assert "PRODUCT_DEFECT" not in projection_json(view)


def test_metadata_and_mapping_order_do_not_change_projection_hash():
    first = project_failure_state(_state(case_id="sample-a"))
    second = project_failure_state(
        _state(
            case_id="sample-b",
            provenance={"dataset_version": "other", "synthetic": True},
            env={"AC_reachable": True, "AP_reachable": True},
        )
    )
    assert first == second
    assert projection_sha256(first) == projection_sha256(second)


def test_observed_failure_fact_changes_projection_hash():
    baseline = project_failure_state(_state())
    changed = project_failure_state(_state(assertion="association refused"))
    assert projection_sha256(baseline) != projection_sha256(changed)
