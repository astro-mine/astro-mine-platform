"""MissionSpec v0.1 acceptance tests (RM-P1-CORE-04 / RFC-0001).

The mission-architecture hooks are **reserved schema, no mechanism**. These tests cover
the RFC-0001 acceptance: a single-`surface`-phase mission validates as a one-phase
MissionSpec with no author action; documents round-trip byte-stably through the Protobuf
wire form (proving proto3 round-trips without loss); the full multi-regime schema loads;
malformed documents are rejected loudly; and TrajectoryRef/Maneuver omit executable-
guidance channels by schema (the dual-use gate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from astro_mine.core import mission

EXAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "examples" / "mission").glob("*.mission.yaml")
)
SINGLE = next(p for p in EXAMPLES if "single-phase" in p.name)
MULTIPHASE = next(p for p in EXAMPLES if "multiphase" in p.name)


def _minimal() -> dict[str, Any]:
    """Smallest valid mission document — a single `surface` phase (a Phase-0 campaign)."""
    return {
        "mission_version": "0.1",
        "mission": {
            "id": "m",
            "name": "M",
            "phases": [{"id": "p1", "regime": "surface"}],
        },
    }


def test_examples_present() -> None:
    assert EXAMPLES, "expected at least one mission example"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_loads_and_validates(path: Path) -> None:
    doc = mission.load_mission(path.read_text())
    assert isinstance(doc, mission.MissionDocument)
    mission.validate_mission(doc)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_roundtrips_byte_stably(path: Path) -> None:
    """A mission round-trips to/from the proto wire form byte-stably (proto3 without loss)."""
    doc = mission.load_mission(path.read_text())
    wire = mission.to_wire(doc)
    restored = mission.from_wire(wire)
    assert restored == doc
    assert mission.to_wire(restored) == wire


def test_single_surface_phase_validates_with_no_author_action() -> None:
    # RFC-0001 §5: a single-`surface`-phase mission is exactly today's campaign.
    doc = mission.load_mission(json.dumps(_minimal()))
    assert [p.regime for p in doc.mission.phases] == [mission.Regime.SURFACE]
    doc_from_example = mission.load_mission(SINGLE.read_text())
    assert len(doc_from_example.mission.phases) == 1


def test_load_yaml_and_json_are_equivalent() -> None:
    import yaml

    doc = _minimal()
    assert mission.load_mission(json.dumps(doc)) == mission.load_mission(yaml.safe_dump(doc))


def test_load_accepts_bytes() -> None:
    assert isinstance(
        mission.load_mission(json.dumps(_minimal()).encode()), mission.MissionDocument
    )


def test_validate_mission_accepts_typed_text_and_dict() -> None:
    doc = mission.load_mission(json.dumps(_minimal()))
    mission.validate_mission(doc)  # typed
    mission.validate_mission(json.dumps(_minimal()))  # text
    mission.validate_mission(_minimal())  # dict


# --- the full multi-regime schema + descriptive trajectory artifacts -------------


def test_multiphase_example_carries_descriptive_trajectory_artifacts() -> None:
    doc = mission.load_mission(MULTIPHASE.read_text())
    phases = {p.id: p for p in doc.mission.phases}
    leg = phases["launch"].legs[0]
    assert leg.trajectory_ref is not None
    assert leg.trajectory_ref.frame == "ECLIPJ2000"
    assert leg.trajectory_ref.maneuvers[0].maneuver_type is mission.ManeuverType.LOW_THRUST_ARC
    assert leg.maneuver_budget is not None
    assert leg.maneuver_budget.total_delta_v_mps == 4200.0
    # the six-regime enum is exercised end to end
    assert {p.regime for p in doc.mission.phases} >= {
        mission.Regime.LAUNCH_ASCENT,
        mission.Regime.INTERPLANETARY_TRANSIT,
        mission.Regime.SURFACE,
        mission.Regime.EARTH_INTERFACE,
    }


def test_phase_transition_is_a_reserved_typed_event() -> None:
    t = mission.PhaseTransition(from_phase="surface_ops", to_phase="ascent")
    assert t.terminal_state_ref is None and t.initial_state_ref is None
    t2 = mission.PhaseTransition(
        from_phase="a", to_phase="b", terminal_state_ref="s1", initial_state_ref="s1"
    )
    assert t2.terminal_state_ref == "s1"


# --- dual-use: TrajectoryRef omits executable guidance by schema (RFC-0001 R3) ---


def test_trajectory_ref_omits_executable_guidance_fields_by_schema() -> None:
    schema = mission.load_schema()
    forbidden = (
        "actuator",
        "thruster",
        "gain",
        "guidance",
        "closed_loop",
        "flight_clock",
        "command",
    )
    for defname in ("TrajectoryRef", "Maneuver", "ReferenceState", "TrajectorySegment"):
        for prop in schema["$defs"][defname]["properties"]:
            for bad in forbidden:
                assert bad not in prop, f"{defname}.{prop} looks like executable guidance"
    # A Maneuver is descriptive only — epoch, Δv magnitude, direction, and a type, plus the
    # RFC-0007 information-preserving typed `epoch` sibling: the same design-time TDB instant
    # as epoch_tdb_s, no onboard-clock binding, no guidance capability (RFC-0007 R3; RFC-0001
    # R3). The forbidden-substring guardrail above still rejects any actuator/guidance field.
    assert set(schema["$defs"]["Maneuver"]["properties"]) == {
        "epoch_tdb_s",
        "delta_v_mps",
        "direction",
        "maneuver_type",
        "epoch",
    }


# --- loud rejection --------------------------------------------------------------


def test_unknown_field_rejected() -> None:
    doc = _minimal()
    doc["mission"]["bogus"] = 1
    with pytest.raises(mission.MissionValidationError):
        mission.validate_mission(doc)


def test_bad_regime_rejected() -> None:
    doc = _minimal()
    doc["mission"]["phases"][0]["regime"] = "hyperspace"
    with pytest.raises(mission.MissionValidationError):
        mission.validate_mission(doc)


def test_bad_maneuver_type_rejected() -> None:
    doc = _minimal()
    doc["mission"]["phases"][0]["legs"] = [
        {
            "id": "l",
            "trajectory_ref": {
                "id": "t",
                "frame": "J2000",
                "maneuvers": [
                    {
                        "epoch_tdb_s": 0.0,
                        "delta_v_mps": 1.0,
                        "direction": {"x": 0.0, "y": 0.0, "z": 1.0},
                        "maneuver_type": "warp",
                    }
                ],
            },
        }
    ]
    with pytest.raises(mission.MissionValidationError):
        mission.validate_mission(doc)


def test_missing_required_name_rejected() -> None:
    doc = _minimal()
    del doc["mission"]["name"]
    with pytest.raises(mission.MissionValidationError):
        mission.validate_mission(doc)


def test_bad_version_const_rejected() -> None:
    doc = _minimal()
    doc["mission_version"] = "0.2"
    with pytest.raises(mission.MissionValidationError):
        mission.validate_mission(doc)


def test_empty_document_rejected() -> None:
    with pytest.raises(mission.MissionValidationError):
        mission.load_mission("{}")


def test_non_mapping_rejected() -> None:
    with pytest.raises(mission.MissionValidationError, match="must be a YAML/JSON mapping"):
        mission.load_mission("[1, 2, 3]")


def test_validate_mission_rejects_unknown_type() -> None:
    with pytest.raises(mission.MissionValidationError, match="cannot validate object of type"):
        mission.validate_mission(object())  # type: ignore[arg-type]
