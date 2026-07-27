"""SADF v0.1 acceptance tests (RM-P0-CORE-01).

Covers the issue's acceptance criteria: hand-authored YAML validates against the
schema and round-trips to/from the Protobuf wire form byte-stably; the schema
rejects unknown/typo'd fields loudly; capability tags are a closed vocabulary and
the reserved ``operational_targeting`` tag is gated out of the open commons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from astro_mine.core import compat, sadf
from astro_mine.core.sadf import enums

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples" / "assets").glob("*.sadf.yaml"))


def _minimal() -> dict[str, Any]:
    """Smallest valid SADF document."""
    return {
        "sadf_version": "0.1",
        "asset": {
            "identity": {"id": "a", "name": "A", "version": "0.1.0", "kind": "rover"},
            "root_frame": "body",
            "frames": [{"name": "body"}],
        },
    }


def test_examples_present() -> None:
    assert EXAMPLES, "expected at least one SADF example asset"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_loads_and_validates(path: Path) -> None:
    doc = sadf.load_sadf(path.read_text())
    assert isinstance(doc, sadf.SadfDocument)
    sadf.validate_sadf(doc)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_roundtrips_byte_stably(path: Path) -> None:
    """Acceptance: a hand-authored asset round-trips to/from the proto wire form
    byte-stably."""
    doc = sadf.load_sadf(path.read_text())
    wire = sadf.to_wire(doc)
    restored = sadf.from_wire(wire)
    assert restored == doc
    assert sadf.to_wire(restored) == wire


def test_load_yaml_and_json_are_equivalent() -> None:
    doc = _minimal()
    assert sadf.load_sadf(json.dumps(doc)) == sadf.load_sadf(yaml.safe_dump(doc))


def test_load_accepts_bytes() -> None:
    assert isinstance(sadf.load_sadf(json.dumps(_minimal()).encode()), sadf.SadfDocument)


# --- loud rejection (acceptance criterion 2) -------------------------------------


def test_unknown_top_level_field_rejected() -> None:
    doc = _minimal()
    doc["nonsense"] = 1
    with pytest.raises(sadf.SadfValidationError):
        sadf.validate_sadf(doc)


def test_unknown_asset_field_rejected() -> None:
    doc = _minimal()
    doc["asset"]["bogus"] = 1
    with pytest.raises(sadf.SadfValidationError):
        sadf.validate_sadf(doc)


def test_typod_field_rejected() -> None:
    doc = _minimal()
    doc["asset"]["identity"]["nam"] = "typo-of-name"
    with pytest.raises(sadf.SadfValidationError):
        sadf.validate_sadf(doc)


def test_missing_required_field_rejected() -> None:
    doc = _minimal()
    del doc["asset"]["root_frame"]
    with pytest.raises(sadf.SadfValidationError):
        sadf.validate_sadf(doc)


def test_bad_enum_value_rejected() -> None:
    doc = _minimal()
    doc["asset"]["capabilities"] = ["mobility.teleport"]
    with pytest.raises(sadf.SadfValidationError):
        sadf.validate_sadf(doc)


def test_bad_sadf_version_rejected() -> None:
    doc = _minimal()
    doc["sadf_version"] = "0.2"
    with pytest.raises(sadf.SadfValidationError):
        sadf.validate_sadf(doc)


def test_wrong_scalar_type_rejected() -> None:
    doc = _minimal()
    doc["asset"]["bodies"] = [
        {
            "name": "b",
            "frame": "body",
            "mass_kg": {},  # object where a number is required
            "center_of_mass_m": {"x": 0, "y": 0, "z": 0},
            "inertia_kg_m2": {"ixx": 1, "iyy": 1, "izz": 1},
        }
    ]
    with pytest.raises(sadf.SadfValidationError):
        sadf.validate_sadf(doc)


def test_empty_document_rejected() -> None:
    with pytest.raises(sadf.SadfValidationError):
        sadf.load_sadf("{}")


# --- semantic / dual-use gate ----------------------------------------------------


def test_operational_targeting_is_gated() -> None:
    doc = _minimal()
    doc["asset"]["capabilities"] = ["operational_targeting"]
    with pytest.raises(sadf.SadfValidationError, match="operational_targeting"):
        sadf.validate_sadf(doc)


def test_ground_truth_access_is_gated() -> None:
    doc = _minimal()
    doc["asset"]["capabilities"] = ["ground_truth_access"]
    with pytest.raises(sadf.SadfValidationError, match="ground_truth_access"):
        sadf.validate_sadf(doc)


def test_live_mission_link_prediction_is_gated() -> None:
    doc = _minimal()
    doc["asset"]["capabilities"] = ["comms.live_mission_link_prediction"]
    with pytest.raises(sadf.SadfValidationError, match=r"comms\.live_mission_link_prediction"):
        sadf.validate_sadf(doc)


def test_root_frame_must_resolve() -> None:
    doc = _minimal()
    doc["asset"]["root_frame"] = "does-not-exist"
    with pytest.raises(sadf.SadfValidationError, match="root_frame"):
        sadf.validate_sadf(doc)


# --- capability vocabulary -------------------------------------------------------


def test_capability_vocabulary_is_closed_and_gated() -> None:
    assert enums.CapabilityTag.OPERATIONAL_TARGETING in enums.GATED_CAPABILITY_TAGS
    assert enums.CapabilityTag.GROUND_TRUTH_ACCESS in enums.GATED_CAPABILITY_TAGS
    # Append-only Wave-10 addition: live-mission link prediction is dual-use operational
    # intelligence, so it is reserved + gated (RFC-0003 / RM-P1-LINK-13).
    assert enums.CapabilityTag.COMMS_LIVE_MISSION_LINK_PREDICTION in enums.GATED_CAPABILITY_TAGS
    assert enums.GATED_CAPABILITY_TAGS.issubset(set(enums.CapabilityTag))
    # Tag values use the documented dotted-namespace form (or a bare reserved tag).
    for tag in enums.CapabilityTag:
        assert tag.value == tag.value.lower()


# --- core_interface_versions is a mapping (B2) -----------------------------------


def test_core_interface_versions_is_a_mapping() -> None:
    doc = _minimal()
    doc["asset"]["core_interface_versions"] = {"sadf": "0.1.0", "messages": "0.1.0"}
    loaded = sadf.load_sadf(json.dumps(doc))
    assert loaded.asset.core_interface_versions == {"sadf": "0.1.0", "messages": "0.1.0"}


def test_old_list_shape_is_rejected() -> None:
    # The pre-B2 array-of-string shape no longer validates (schema + Pydantic agree).
    doc = _minimal()
    doc["asset"]["core_interface_versions"] = ["0.1"]
    with pytest.raises(sadf.SadfValidationError):
        sadf.validate_sadf(doc)


def test_malformed_core_interface_version_is_refused() -> None:
    doc = _minimal()
    doc["asset"]["core_interface_versions"] = {"sadf": "0.1"}  # not MAJOR.MINOR.PATCH
    with pytest.raises(sadf.SadfValidationError, match=r"MAJOR\.MINOR\.PATCH"):
        sadf.validate_sadf(doc)


def test_assert_core_compatible_bridge_passes() -> None:
    doc = _minimal()
    doc["asset"]["core_interface_versions"] = {"sadf": "0.1.0"}
    asset = sadf.load_sadf(json.dumps(doc)).asset
    asset.assert_core_compatible()  # satisfied by this Core


def test_assert_core_compatible_bridge_refuses_unsatisfied() -> None:
    doc = _minimal()
    doc["asset"]["core_interface_versions"] = {"sadf": "9.9.9"}
    asset = sadf.load_sadf(json.dumps(doc)).asset
    with pytest.raises(compat.IncompatibleCoreInterface, match="sadf"):
        asset.assert_core_compatible()
    # an explicit `provided` override negotiates against a hypothetical future Core
    asset.assert_core_compatible(provided={"sadf": "9.9.0"})


# --- kinematic-graph referential closure (RM-P1-CORE-05) -------------------------


def _articulated() -> dict[str, Any]:
    """A valid, referentially-closed articulated asset exercising every cross-reference:
    a chassis + wheel body joined by an actuated joint, plus geometry/sensor/comms/
    payload/sub-assembly frame references — each resolving to a declared entity."""
    return {
        "sadf_version": "0.1",
        "asset": {
            "identity": {"id": "art", "name": "Art", "version": "0.1.0", "kind": "rover"},
            "root_frame": "body",
            "frames": [{"name": "body"}, {"name": "wheel", "parent": "body"}],
            "geometry": [{"role": "visual", "format": "usd", "uri": "a.usd", "frame": "body"}],
            "bodies": [
                {
                    "name": "chassis",
                    "frame": "body",
                    "mass_kg": 10.0,
                    "center_of_mass_m": {"x": 0, "y": 0, "z": 0},
                    "inertia_kg_m2": {"ixx": 1, "iyy": 1, "izz": 1},
                },
                {
                    "name": "wheel",
                    "frame": "wheel",
                    "mass_kg": 1.0,
                    "center_of_mass_m": {"x": 0, "y": 0, "z": 0},
                    "inertia_kg_m2": {"ixx": 0.1, "iyy": 0.1, "izz": 0.1},
                },
            ],
            "joints": [
                {
                    "name": "hub",
                    "type": "continuous",
                    "parent_body": "chassis",
                    "child_body": "wheel",
                    "axis": {"x": 0, "y": 1, "z": 0},
                }
            ],
            "actuators": [{"name": "drive", "target_joint": "hub", "torque_nm": 5.0}],
            "sensors": [{"name": "cam", "kind": "imaging", "frame": "wheel"}],
            "comms": [
                {
                    "name": "radio",
                    "band": "uhf",
                    "antenna": {"gain_dbi": 3.0, "boresight_frame": "body"},
                }
            ],
            "payload": {"slots": [{"name": "deck", "frame": "body"}]},
            "subassemblies": [{"ref": "child-asset", "mount_frame": "body"}],
        },
    }


def test_articulated_asset_is_referentially_closed() -> None:
    doc = _articulated()
    assert isinstance(sadf.load_sadf(json.dumps(doc)), sadf.SadfDocument)
    sadf.validate_sadf(doc)


def test_resource_storage_sensor_kind_is_declarable() -> None:
    """The append-only ``resource_storage`` sensor kind (RFC-0003 / RM-P1-SIM-02) —
    an ISRU stored-mass gauge — declares and validates like any other sensor kind, so
    Sim can render cumulative extracted water (kg) into the Observation stream Bench
    scores. Append-only: the pre-existing kinds are untouched."""
    assert enums.SensorKind.RESOURCE_STORAGE.value == "resource_storage"
    doc = _articulated()
    doc["asset"]["sensors"].append(
        {"name": "isru_tank", "kind": "resource_storage", "frame": "body"}
    )
    sadf.validate_sadf(doc)
    parsed = sadf.load_sadf(json.dumps(doc))
    assert parsed.asset.sensors[-1].kind is enums.SensorKind.RESOURCE_STORAGE


def test_joint_child_body_must_be_declared() -> None:
    # The astro-mine-fleet#15 defect: a joint child_body that was never declared.
    doc = _articulated()
    doc["asset"]["joints"][0]["child_body"] = "ghost_wheel"
    with pytest.raises(
        sadf.SadfValidationError, match=r"child_body references undeclared body 'ghost_wheel'"
    ):
        sadf.validate_sadf(doc)


def test_joint_parent_body_must_be_declared() -> None:
    doc = _articulated()
    doc["asset"]["joints"][0]["parent_body"] = "ghost_chassis"
    with pytest.raises(
        sadf.SadfValidationError, match=r"parent_body references undeclared body 'ghost_chassis'"
    ):
        sadf.validate_sadf(doc)


def test_actuator_target_joint_must_be_declared() -> None:
    doc = _articulated()
    doc["asset"]["actuators"][0]["target_joint"] = "ghost_joint"
    with pytest.raises(
        sadf.SadfValidationError, match=r"target_joint references undeclared joint 'ghost_joint'"
    ):
        sadf.validate_sadf(doc)


def test_frame_parent_must_be_declared() -> None:
    doc = _articulated()
    doc["asset"]["frames"][1]["parent"] = "ghost_frame"
    with pytest.raises(
        sadf.SadfValidationError,
        match=r"frame 'wheel' parent references undeclared frame 'ghost_frame'",
    ):
        sadf.validate_sadf(doc)


def test_body_frame_must_be_declared() -> None:
    doc = _articulated()
    doc["asset"]["bodies"][1]["frame"] = "ghost_frame"
    with pytest.raises(
        sadf.SadfValidationError, match=r"body 'wheel' references undeclared frame 'ghost_frame'"
    ):
        sadf.validate_sadf(doc)


def test_geometry_frame_must_be_declared() -> None:
    doc = _articulated()
    doc["asset"]["geometry"][0]["frame"] = "ghost_frame"
    with pytest.raises(
        sadf.SadfValidationError,
        match=r"geometry 'a.usd' references undeclared frame 'ghost_frame'",
    ):
        sadf.validate_sadf(doc)


def test_sensor_frame_must_be_declared() -> None:
    doc = _articulated()
    doc["asset"]["sensors"][0]["frame"] = "ghost_frame"
    with pytest.raises(
        sadf.SadfValidationError, match=r"sensor 'cam' references undeclared frame 'ghost_frame'"
    ):
        sadf.validate_sadf(doc)


def test_comms_boresight_frame_must_be_declared() -> None:
    doc = _articulated()
    doc["asset"]["comms"][0]["antenna"]["boresight_frame"] = "ghost_frame"
    with pytest.raises(
        sadf.SadfValidationError, match=r"boresight_frame references undeclared frame 'ghost_frame'"
    ):
        sadf.validate_sadf(doc)


def test_payload_slot_frame_must_be_declared() -> None:
    doc = _articulated()
    doc["asset"]["payload"]["slots"][0]["frame"] = "ghost_frame"
    with pytest.raises(
        sadf.SadfValidationError,
        match=r"payload slot 'deck' references undeclared frame 'ghost_frame'",
    ):
        sadf.validate_sadf(doc)


def test_subassembly_mount_frame_must_be_declared() -> None:
    doc = _articulated()
    doc["asset"]["subassemblies"][0]["mount_frame"] = "ghost_frame"
    with pytest.raises(
        sadf.SadfValidationError,
        match=r"subassembly 'child-asset' mount_frame references undeclared frame 'ghost_frame'",
    ):
        sadf.validate_sadf(doc)


def test_all_dangling_references_reported_together() -> None:
    """Fail-loud reports every dangling reference at once, not just the first."""
    doc = _articulated()
    doc["asset"]["joints"][0]["child_body"] = "ghost_wheel"
    doc["asset"]["actuators"][0]["target_joint"] = "ghost_joint"
    with pytest.raises(sadf.SadfValidationError) as exc:
        sadf.validate_sadf(doc)
    msg = str(exc.value)
    assert "ghost_wheel" in msg
    assert "ghost_joint" in msg
