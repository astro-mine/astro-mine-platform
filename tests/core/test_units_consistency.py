"""Drift guards between the canonical units JSON Schema and the Pydantic models.

Mirrors ``tests/test_messages_consistency.py``: (1) every enum ``$def`` matches its
Python enum; (2) a corpus of valid/invalid documents gets the same *structural* verdict
from the JSON Schema validator and from Pydantic, per type.

The **semantic** guard rules (``require_frame``/``require_crs``, ``EpochWindow`` ordering,
``ET`` == ``TDB``, and the Earth-CRS body/datum consistency rule) are deliberately NOT
structural — JSON Schema cannot express them and the Pydantic *models* do not enforce all
of them either. They are the conformance-vector contract (RM-P1-CORE-08; conventions.md
§5), tested against ``units.validate``, not here. So a MOON CRS carrying a WGS84 datum is
valid at *this* (structural) layer and both validators agree it is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

import astro_mine.core.units as units_pkg
from astro_mine.core.units import enums, model

SCHEMA_PATH = Path(units_pkg.__file__).resolve().parent / "schema" / "units.schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


ENUM_DEFS = {
    "TimeScale": enums.TimeScale,
    "FrameClass": enums.FrameClass,
}


@pytest.mark.parametrize("name", list(ENUM_DEFS), ids=list(ENUM_DEFS))
def test_schema_enum_matches_python_enum(name: str) -> None:
    schema = load_schema()
    assert name in schema["$defs"], f"{name} missing from schema $defs"
    schema_values = set(schema["$defs"][name]["enum"])
    python_values = {member.value for member in ENUM_DEFS[name]}
    assert schema_values == python_values


def _validator(root: str) -> Draft202012Validator:
    schema = load_schema()
    return Draft202012Validator({"$ref": f"#/$defs/{root}", "$defs": schema["$defs"]})


def _jsonschema_ok(data: Any, root: str) -> bool:
    return not list(_validator(root).iter_errors(data))


def _pydantic_ok(data: Any, cls: type[BaseModel]) -> bool:
    try:
        cls.model_validate(data)
    except ValidationError:
        return False
    return True


# (id, root $def, model, data) — the JSON Schema validator and Pydantic MUST return the
# same structural verdict for each. Non-agreeing (semantic-guard) cases belong in the
# conformance vectors (RM-P1-CORE-08), not here.
CASES: list[tuple[str, str, type[BaseModel], dict[str, Any]]] = [
    # ReferenceFrame
    (
        "frame-ok",
        "ReferenceFrame",
        model.ReferenceFrame,
        {"name": "MOON_ME", "frame_class": "body_fixed", "center": "MOON"},
    ),
    (
        "frame-no-center",
        "ReferenceFrame",
        model.ReferenceFrame,
        {"name": "J2000", "frame_class": "inertial"},
    ),
    (
        "frame-null-center",
        "ReferenceFrame",
        model.ReferenceFrame,
        {"name": "J2000", "frame_class": "inertial", "center": None},
    ),
    (
        "frame-empty-name",
        "ReferenceFrame",
        model.ReferenceFrame,
        {"name": "", "frame_class": "inertial"},
    ),
    (
        "frame-whitespace-name",
        "ReferenceFrame",
        model.ReferenceFrame,
        {"name": "MOON ME", "frame_class": "body_fixed"},
    ),
    (
        "frame-padded-name",
        "ReferenceFrame",
        model.ReferenceFrame,
        {"name": " J2000 ", "frame_class": "inertial"},
    ),
    (
        "frame-bad-class",
        "ReferenceFrame",
        model.ReferenceFrame,
        {"name": "J2000", "frame_class": "galactic"},
    ),
    (
        "frame-unknown-field",
        "ReferenceFrame",
        model.ReferenceFrame,
        {"name": "J2000", "frame_class": "inertial", "bogus": 1},
    ),
    ("frame-missing-class", "ReferenceFrame", model.ReferenceFrame, {"name": "J2000"}),
    # PlanetaryCRS
    (
        "crs-ok",
        "PlanetaryCRS",
        model.PlanetaryCRS,
        {"body": "MOON", "body_fixed_frame": "MOON_ME", "reference_radius_m": 1737400.0},
    ),
    (
        "crs-projected-spaces",
        "PlanetaryCRS",
        model.PlanetaryCRS,
        {
            "body": "MOON",
            "body_fixed_frame": "MOON_ME",
            "reference_radius_m": 1737400.0,
            "projection": "+proj=stere +lat_0=-90 +R=1737400",
            "datum": None,
        },
    ),
    # Earth marker + non-Earth body is a guard-layer reject (rule 6), but structurally valid
    # here — both validators agree it passes at the waist.
    (
        "crs-moon-wgs84-valid-at-waist",
        "PlanetaryCRS",
        model.PlanetaryCRS,
        {
            "body": "MOON",
            "body_fixed_frame": "MOON_ME",
            "reference_radius_m": 1737400.0,
            "projection": "+proj=longlat +datum=WGS84",
        },
    ),
    (
        "crs-zero-radius",
        "PlanetaryCRS",
        model.PlanetaryCRS,
        {"body": "MOON", "body_fixed_frame": "MOON_ME", "reference_radius_m": 0.0},
    ),
    (
        "crs-negative-radius",
        "PlanetaryCRS",
        model.PlanetaryCRS,
        {"body": "MOON", "body_fixed_frame": "MOON_ME", "reference_radius_m": -1.0},
    ),
    (
        "crs-empty-body",
        "PlanetaryCRS",
        model.PlanetaryCRS,
        {"body": "", "body_fixed_frame": "MOON_ME", "reference_radius_m": 1.0},
    ),
    (
        "crs-missing-radius",
        "PlanetaryCRS",
        model.PlanetaryCRS,
        {"body": "MOON", "body_fixed_frame": "MOON_ME"},
    ),
    # Epoch
    ("epoch-ok", "Epoch", model.Epoch, {"tdb_seconds": 0.0, "scale": "tdb"}),
    ("epoch-et", "Epoch", model.Epoch, {"tdb_seconds": 100.0, "scale": "et"}),
    ("epoch-missing-scale", "Epoch", model.Epoch, {"tdb_seconds": 0.0}),
    ("epoch-bad-scale", "Epoch", model.Epoch, {"tdb_seconds": 0.0, "scale": "utc"}),
    # EpochWindow (ordering is a semantic guard, not structural — no out-of-order case here)
    (
        "window-ok",
        "EpochWindow",
        model.EpochWindow,
        {
            "start": {"tdb_seconds": 0.0, "scale": "tdb"},
            "end": {"tdb_seconds": 1.0, "scale": "tdb"},
        },
    ),
    (
        "window-missing-end",
        "EpochWindow",
        model.EpochWindow,
        {"start": {"tdb_seconds": 0.0, "scale": "tdb"}},
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_jsonschema_and_pydantic_agree(
    case: tuple[str, str, type[BaseModel], dict[str, Any]],
) -> None:
    _, root, cls, data = case
    assert _jsonschema_ok(data, root) == _pydantic_ok(data, cls)


def test_schema_is_self_consistent_json() -> None:
    Draft202012Validator.check_schema(load_schema())
