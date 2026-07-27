"""Drift guards between the canonical SafetySpec JSON Schema and the Pydantic models.

The JSON Schema is canonical; the Pydantic models mirror it (until a generator makes one from
the other). Two guards keep them aligned, mirroring Core's ``test_objective_consistency``:

1. every enum ``$def`` in the schema has exactly the members of the matching Python enum;
2. a corpus of valid/invalid documents gets the *same* structural verdict from the JSON Schema
   validator and from Pydantic (fail-safe *semantics* live in the loader, so the two are a
   single structural contract here).
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core import schema_registry
from astro_mine.guard.spec import enums
from astro_mine.guard.spec.loader import load_schema
from astro_mine.guard.spec.model import SafetyDocument

# Guard's shipped reference safety specs (package data). The anchor is the one authored spec today;
# any spec added under reference/safety_specs/ is checked for schema/Pydantic agreement too.
EXAMPLES = sorted(
    (
        entry
        for entry in resources.files("astro_mine.guard.reference")
        .joinpath("safety_specs")
        .iterdir()
        if entry.name.endswith(".safety.yaml")
    ),
    key=lambda entry: entry.name,
)

ENUM_DEFS = {
    "ConstraintKind": enums.ConstraintKind,
    "PredicateOp": enums.PredicateOp,
    "TemporalOp": enums.TemporalOp,
    "OnUncertain": enums.OnUncertain,
    "SignalSource": enums.SignalSource,
    "GeometryKind": enums.GeometryKind,
}


@pytest.mark.parametrize("name", list(ENUM_DEFS), ids=list(ENUM_DEFS))
def test_schema_enum_matches_python_enum(name: str) -> None:
    schema = load_schema()
    assert name in schema["$defs"], f"{name} missing from schema $defs"
    schema_values = set(schema["$defs"][name]["enum"])
    python_values = {member.value for member in ENUM_DEFS[name]}
    assert schema_values == python_values


def test_on_uncertain_has_no_passthrough() -> None:
    # Fail-safe by vocabulary: the schema must never admit a "passthrough" resolution.
    assert "passthrough" not in load_schema()["$defs"]["OnUncertain"]["enum"]


def _jsonschema_ok(data: Any) -> bool:
    # frame_ref fields $ref Core's units.schema.json by its absolute $id (RFC-0009 §1); resolve
    # them offline through Core's public schema_registry, exactly as the loader's validator does.
    schema = load_schema()
    validator = Draft202012Validator(schema, registry=schema_registry(schema))
    return not list(validator.iter_errors(data))


def _pydantic_ok(data: Any) -> bool:
    try:
        SafetyDocument.model_validate(data)
    except ValidationError:
        return False
    return True


def _base() -> dict[str, Any]:
    return {
        "safety_version": "0.1",
        "safety": {
            "id": "s",
            "name": "S",
            "signals": [{"key": "soc", "unit": "J", "source": "observation"}],
            "constraints": [
                {
                    "kind": "energy_floor",
                    "id": "c1",
                    "energy_floor": {"signal": "soc", "floor_j": 1.0},
                }
            ],
        },
    }


def _corpus() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [("minimal-valid", _base())]

    unknown_top = _base()
    unknown_top["extra"] = 1
    cases.append(("unknown-top-level", unknown_top))

    unknown_spec = _base()
    unknown_spec["safety"]["bogus"] = 1
    cases.append(("unknown-spec-field", unknown_spec))

    typo = _base()
    typo["safety"]["constraints"][0]["energy_floor"]["floo_j"] = 1.0
    cases.append(("payload-typo", typo))

    missing_safety = {"safety_version": "0.1"}
    cases.append(("missing-safety", missing_safety))

    missing_version = {"safety": _base()["safety"]}
    cases.append(("missing-version", missing_version))

    bad_const = _base()
    bad_const["safety_version"] = "0.2"
    cases.append(("bad-version-const", bad_const))

    empty_constraints = _base()
    empty_constraints["safety"]["constraints"] = []
    cases.append(("empty-constraints", empty_constraints))

    bad_kind = _base()
    bad_kind["safety"]["constraints"][0]["kind"] = "teleport"
    cases.append(("bad-constraint-kind", bad_kind))

    bad_source = _base()
    bad_source["safety"]["signals"][0]["source"] = "telepathy"
    cases.append(("bad-signal-source", bad_source))

    bad_on_uncertain = _base()
    bad_on_uncertain["safety"]["constraints"][0]["on_uncertain"] = "passthrough"
    cases.append(("passthrough-rejected", bad_on_uncertain))

    neg_margin = _base()
    neg_margin["safety"]["constraints"][0] = {
        "kind": "keep_out",
        "id": "k",
        "keep_out": {
            "margin_m": -1.0,
            "volume": {
                "shape": "sphere",
                "sphere": {"frame": "F", "center_m": {"x": 0, "y": 0, "z": 0}, "radius_m": 1.0},
            },
        },
    }
    cases.append(("negative-margin", neg_margin))

    bad_cmp = _base()
    bad_cmp["safety"]["constraints"][0] = {
        "kind": "temporal",
        "id": "t",
        "temporal": {
            "formula": {
                "op": "always",
                "interval_s": {"lo": 0.0, "hi": 10.0},
                "args": [{"op": "predicate", "signal": "soc", "cmp": "approx", "threshold": 1.0}],
            }
        },
    }
    cases.append(("bad-predicate-op", bad_cmp))

    full_valid = _base()
    full_valid["safety"]["description"] = "d"
    full_valid["safety"]["scenario_ref"] = "sha256:ab"
    full_valid["safety"]["provenance"] = {"seed": 1, "input_hashes": ["sha256:00"]}
    full_valid["safety"]["constraints"][0]["description"] = "d"
    full_valid["safety"]["constraints"][0]["on_uncertain"] = "hold"
    cases.append(("full-valid", full_valid))

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
