"""Drift guards between the canonical mission JSON Schema and the Pydantic models.

Mirrors tests/test_objective_consistency.py: (1) every enum ``$def`` matches the Python
enum (including the SADF-owned :class:`Regime` the mission schema reuses); (2) a corpus of
valid/invalid documents gets the *same* structural verdict from the JSON Schema validator
and from Pydantic. The mission hooks are reserved schema with no semantic checks, so the
two are a single structural contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.mission import enums
from astro_mine.core.mission.loader import load_schema
from astro_mine.core.mission.model import MissionDocument
from astro_mine.core.schemas import schema_registry

EXAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "examples" / "mission").glob("*.mission.yaml")
)

ENUM_DEFS = {
    "Regime": enums.Regime,
    "ManeuverType": enums.ManeuverType,
}


@pytest.mark.parametrize("name", list(ENUM_DEFS), ids=list(ENUM_DEFS))
def test_schema_enum_matches_python_enum(name: str) -> None:
    schema = load_schema()
    assert name in schema["$defs"], f"{name} missing from schema $defs"
    schema_values = set(schema["$defs"][name]["enum"])
    python_values = {member.value for member in ENUM_DEFS[name]}
    assert schema_values == python_values


def _jsonschema_ok(data: Any) -> bool:
    schema = load_schema()
    return not list(
        Draft202012Validator(schema, registry=schema_registry(schema)).iter_errors(data)
    )


def _pydantic_ok(data: Any) -> bool:
    try:
        MissionDocument.model_validate(data)
    except ValidationError:
        return False
    return True


def _base() -> dict[str, Any]:
    return {
        "mission_version": "0.1",
        "mission": {"id": "m", "name": "M", "phases": [{"id": "p", "regime": "surface"}]},
    }


def _corpus() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [("minimal-valid", _base())]

    unknown_top = _base()
    unknown_top["extra"] = 1
    cases.append(("unknown-top-level", unknown_top))

    unknown_field = _base()
    unknown_field["mission"]["bogus"] = 1
    cases.append(("unknown-mission-field", unknown_field))

    bad_regime = _base()
    bad_regime["mission"]["phases"][0]["regime"] = "hyperspace"
    cases.append(("bad-regime", bad_regime))

    missing_mission = {"mission_version": "0.1"}
    cases.append(("missing-mission", missing_mission))

    missing_name = _base()
    del missing_name["mission"]["name"]
    cases.append(("missing-name", missing_name))

    bad_const = _base()
    bad_const["mission_version"] = "0.2"
    cases.append(("bad-version-const", bad_const))

    bad_maneuver = _base()
    bad_maneuver["mission"]["phases"][0]["legs"] = [
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
    cases.append(("bad-maneuver-type", bad_maneuver))

    null_optionals = _base()
    null_optionals["mission"].update(
        {"description": None, "objective_ref": None, "constraints": None, "provenance": None}
    )
    cases.append(("null-optionals", null_optionals))

    phase_boundary = _base()
    phase_boundary["mission"]["phases"][0].update(
        {"entry": None, "exit": {"condition": None, "state_ref": "s"}}
    )
    cases.append(("phase-boundary", phase_boundary))

    # RFC-0007 typed epoch/frame siblings on Maneuver / TrajectoryRef, resolved via the
    # cross-file $ref to units.schema.json.
    typed_wire = _base()
    typed_wire["mission"]["phases"][0]["legs"] = [
        {
            "id": "l",
            "trajectory_ref": {
                "id": "t",
                "frame": "J2000",
                "frame_ref": {"name": "J2000", "frame_class": "inertial"},
                "maneuvers": [
                    {
                        "epoch_tdb_s": 0.0,
                        "delta_v_mps": 1.0,
                        "direction": {"x": 0.0, "y": 0.0, "z": 1.0},
                        "maneuver_type": "impulsive",
                        "epoch": {"tdb_seconds": 0.0, "scale": "et"},
                    }
                ],
            },
        }
    ]
    cases.append(("typed-wire-siblings", typed_wire))

    bad_epoch = _base()
    bad_epoch["mission"]["phases"][0]["legs"] = [
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
                        "maneuver_type": "impulsive",
                        "epoch": {"tdb_seconds": 0.0, "scale": "utc"},  # bad scale -> both reject
                    }
                ],
            },
        }
    ]
    cases.append(("bad-maneuver-epoch-scale", bad_epoch))

    for path in EXAMPLES:
        cases.append((path.name, yaml.safe_load(path.read_text())))

    return cases


@pytest.mark.parametrize("case", _corpus(), ids=lambda c: c[0])
def test_jsonschema_and_pydantic_agree(case: tuple[str, dict[str, Any]]) -> None:
    _, data = case
    assert _jsonschema_ok(data) == _pydantic_ok(data)


def test_schema_is_self_consistent_json() -> None:
    raw = json.dumps(load_schema())
    Draft202012Validator.check_schema(json.loads(raw))
