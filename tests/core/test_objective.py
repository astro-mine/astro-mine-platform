"""ObjectiveSpec v0.1 acceptance tests (RM-P0-CORE-04).

Covers the issue's acceptance criterion: an ObjectiveSpec for the anchor scenario
binds each success criterion to a Bench metric with target + tolerance and validates;
plus loud rejection of malformed documents and a byte-stable Protobuf round-trip
(the SADF discipline, RM-P0-CORE-01).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from astro_mine.core import objective
from astro_mine.core.objective import enums

EXAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "examples" / "objectives").glob("*.objective.yaml")
)
ANCHOR = next(p for p in EXAMPLES if "lunar-polar-ice-prospecting" in p.name)


def _minimal() -> dict[str, Any]:
    """Smallest valid objective document."""
    return {
        "objective_version": "0.1",
        "objective": {
            "id": "obj",
            "name": "Obj",
            "success_criteria": [
                {
                    "id": "c1",
                    "binding": {
                        "metric": "water_mass_produced",
                        "unit": "kg",
                        "direction": "higher_better",
                        "target": 1000.0,
                        "tolerance": 100.0,
                    },
                }
            ],
        },
    }


def test_examples_present() -> None:
    assert EXAMPLES, "expected at least one objective example"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_loads_and_validates(path: Path) -> None:
    doc = objective.load_objective(path.read_text())
    assert isinstance(doc, objective.ObjectiveDocument)
    objective.validate_objective(doc)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_roundtrips_byte_stably(path: Path) -> None:
    doc = objective.load_objective(path.read_text())
    wire = objective.to_wire(doc)
    restored = objective.from_wire(wire)
    assert restored == doc
    assert objective.to_wire(restored) == wire


def test_anchor_binds_every_criterion_to_a_metric_with_target_and_tolerance() -> None:
    """Acceptance: each success criterion binds to a Bench metric with target +
    tolerance (and an explicit unit/direction), and the document validates."""
    doc = objective.load_objective(ANCHOR.read_text())
    criteria = doc.objective.success_criteria
    assert criteria, "anchor objective must declare success criteria"
    for c in criteria:
        b = c.binding
        assert b.metric, f"{c.id}: metric key required"
        assert b.unit, f"{c.id}: explicit unit required"
        assert isinstance(b.target, float)
        assert b.tolerance >= 0.0
        assert isinstance(b.direction, enums.MetricDirection)
    objective.validate_objective(doc)


def test_load_yaml_and_json_are_equivalent() -> None:
    doc = _minimal()
    assert objective.load_objective(json.dumps(doc)) == objective.load_objective(
        yaml.safe_dump(doc)
    )


def test_load_accepts_bytes() -> None:
    assert isinstance(
        objective.load_objective(json.dumps(_minimal()).encode()), objective.ObjectiveDocument
    )


# --- loud rejection --------------------------------------------------------------


def test_unknown_field_rejected() -> None:
    doc = _minimal()
    doc["objective"]["bogus"] = 1
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(doc)


def test_typod_field_rejected() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["targ"] = 1.0
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(doc)


def test_missing_required_binding_field_rejected() -> None:
    doc = _minimal()
    del doc["objective"]["success_criteria"][0]["binding"]["target"]
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(doc)


def test_empty_success_criteria_rejected() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"] = []
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(doc)


def test_bad_direction_enum_rejected() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["direction"] = "sideways"
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(doc)


def test_bad_aggregation_enum_rejected() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["aggregation"] = "stddev"
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(doc)


def test_negative_tolerance_rejected() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["tolerance"] = -1.0
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(doc)


def test_bad_objective_version_rejected() -> None:
    doc = _minimal()
    doc["objective_version"] = "0.2"
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(doc)


def test_empty_document_rejected() -> None:
    with pytest.raises(objective.ObjectiveValidationError):
        objective.load_objective("{}")


# --- semantic --------------------------------------------------------------------


def test_duplicate_criterion_ids_rejected() -> None:
    doc = _minimal()
    crit = doc["objective"]["success_criteria"][0]
    doc["objective"]["success_criteria"].append(json.loads(json.dumps(crit)))  # same id "c1"
    with pytest.raises(objective.ObjectiveValidationError, match="duplicate"):
        objective.validate_objective(doc)


def test_metric_direction_vocabulary_is_closed() -> None:
    assert {d.value for d in enums.MetricDirection} == {"higher_better", "lower_better"}
    for d in enums.MetricDirection:
        assert d.value == d.value.lower()


# --- timing: deadlines + evaluation windows --------------------------------------


def _with_criterion(**extra: object) -> dict[str, Any]:
    doc = _minimal()
    doc["objective"]["success_criteria"][0].update(extra)
    return doc


def test_deadline_and_rolling_window_validate() -> None:
    doc = _with_criterion(deadline_s=2.592e6)
    doc["objective"]["success_criteria"][0]["binding"]["evaluation_window"] = {
        "kind": "rolling",
        "duration_s": 2.551e6,
    }
    loaded = objective.load_objective(json.dumps(doc))
    crit = loaded.objective.success_criteria[0]
    assert crit.deadline_s == 2.592e6
    assert crit.binding.evaluation_window is not None
    assert crit.binding.evaluation_window.kind is enums.WindowKind.ROLLING


def test_anchor_example_exercises_timing() -> None:
    doc = objective.load_objective(ANCHOR.read_text())
    by_id = {c.id: c for c in doc.objective.success_criteria}
    assert by_id["water_produced"].binding.evaluation_window is not None
    assert by_id["resource_characterized"].deadline_s is not None


def test_zero_deadline_rejected() -> None:
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(_with_criterion(deadline_s=0.0))


def test_bad_window_kind_rejected() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["evaluation_window"] = {"kind": "yearly"}
    with pytest.raises(objective.ObjectiveValidationError):
        objective.validate_objective(doc)


def test_rolling_window_requires_duration() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["evaluation_window"] = {"kind": "rolling"}
    with pytest.raises(objective.ObjectiveValidationError, match=r"rolling.*requires duration"):
        objective.validate_objective(doc)


def test_cumulative_window_forbids_duration() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["evaluation_window"] = {
        "kind": "cumulative",
        "duration_s": 5.0,
    }
    with pytest.raises(objective.ObjectiveValidationError, match="must not set duration"):
        objective.validate_objective(doc)


def test_per_phase_window_forbids_duration() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["evaluation_window"] = {
        "kind": "per_phase",
        "duration_s": 5.0,
    }
    with pytest.raises(objective.ObjectiveValidationError, match="must not set duration"):
        objective.validate_objective(doc)


# --- objective->metric binding matured end-to-end (RM-P1-CORE-03) ----------------


def test_check_objective_accepts_the_anchor_end_to_end() -> None:
    """Acceptance: the anchor objective validates and round-trips author -> measure."""
    doc = objective.check_objective(ANCHOR.read_text())
    assert isinstance(doc, objective.ObjectiveDocument)


def test_check_objective_accepts_a_typed_document() -> None:
    doc = objective.load_objective(json.dumps(_minimal()))
    assert objective.check_objective(doc) == doc


def test_check_objective_accepts_bytes() -> None:
    assert isinstance(
        objective.check_objective(json.dumps(_minimal()).encode()), objective.ObjectiveDocument
    )


def test_check_objective_rejects_invalid_binding() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["direction"] = "sideways"
    with pytest.raises(objective.ObjectiveContractError):
        objective.check_objective(json.dumps(doc))


def test_metric_keys_are_distinct_and_ordered() -> None:
    doc = objective.load_objective(ANCHOR.read_text())
    assert doc.objective.metric_keys() == [
        "water_mass_produced",
        "information_gain",
        "nights_survived",
        "comms_robustness",
        "energy_per_kg",
        "psr_area_characterized",
    ]


def test_metric_keys_deduplicate_preserving_order() -> None:
    doc = _minimal()
    second = json.loads(json.dumps(doc["objective"]["success_criteria"][0]))
    second["id"] = "c2"  # distinct criterion, same metric key
    doc["objective"]["success_criteria"].append(second)
    loaded = objective.load_objective(json.dumps(doc))
    assert loaded.objective.metric_keys() == ["water_mass_produced"]


def test_blank_metric_rejected() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["metric"] = "   "
    with pytest.raises(objective.ObjectiveValidationError, match="metric"):
        objective.validate_objective(doc)


def test_blank_unit_rejected() -> None:
    doc = _minimal()
    doc["objective"]["success_criteria"][0]["binding"]["unit"] = ""
    with pytest.raises(objective.ObjectiveValidationError, match="unit"):
        objective.validate_objective(doc)
