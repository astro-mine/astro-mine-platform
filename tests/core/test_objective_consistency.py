"""Drift guards between the canonical ObjectiveSpec JSON Schema and the Pydantic models.

The JSON Schema is canonical; the Pydantic models are hand-written to mirror it
(until RM-P0-CORE-07 generates one from the other). Two guards keep them aligned,
mirroring ``tests/test_sadf_consistency.py``:

1. every enum ``$def`` in the schema has exactly the members of the matching Python
   enum;
2. a corpus of valid/invalid documents gets the *same* structural verdict from the
   JSON Schema validator and from Pydantic (semantics live in the loader, so the two
   are a single structural contract here).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.objective import enums
from astro_mine.core.objective.loader import load_schema
from astro_mine.core.objective.model import ObjectiveDocument

EXAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "examples" / "objectives").glob("*.objective.yaml")
)

ENUM_DEFS = {
    "MetricDirection": enums.MetricDirection,
    "MetricAggregation": enums.MetricAggregation,
    "WindowKind": enums.WindowKind,
}


@pytest.mark.parametrize("name", list(ENUM_DEFS), ids=list(ENUM_DEFS))
def test_schema_enum_matches_python_enum(name: str) -> None:
    schema = load_schema()
    assert name in schema["$defs"], f"{name} missing from schema $defs"
    schema_values = set(schema["$defs"][name]["enum"])
    python_values = {member.value for member in ENUM_DEFS[name]}
    assert schema_values == python_values


def _jsonschema_ok(data: Any) -> bool:
    return not list(Draft202012Validator(load_schema()).iter_errors(data))


def _pydantic_ok(data: Any) -> bool:
    try:
        ObjectiveDocument.model_validate(data)
    except ValidationError:
        return False
    return True


def _base() -> dict[str, Any]:
    return {
        "objective_version": "0.1",
        "objective": {
            "id": "obj",
            "name": "Obj",
            "success_criteria": [
                {
                    "id": "c1",
                    "binding": {
                        "metric": "m",
                        "unit": "kg",
                        "direction": "higher_better",
                        "target": 1.0,
                        "tolerance": 0.1,
                    },
                }
            ],
        },
    }


def _corpus() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [("minimal-valid", _base())]

    unknown_top = _base()
    unknown_top["extra"] = 1
    cases.append(("unknown-top-level", unknown_top))

    unknown_obj = _base()
    unknown_obj["objective"]["bogus"] = 1
    cases.append(("unknown-objective-field", unknown_obj))

    typo = _base()
    typo["objective"]["success_criteria"][0]["binding"]["targ"] = 1.0
    cases.append(("binding-typo", typo))

    missing_obj = {"objective_version": "0.1"}
    cases.append(("missing-objective", missing_obj))

    missing_version = {"objective": _base()["objective"]}
    cases.append(("missing-version", missing_version))

    bad_const = _base()
    bad_const["objective_version"] = "0.2"
    cases.append(("bad-version-const", bad_const))

    empty_criteria = _base()
    empty_criteria["objective"]["success_criteria"] = []
    cases.append(("empty-criteria", empty_criteria))

    bad_dir = _base()
    bad_dir["objective"]["success_criteria"][0]["binding"]["direction"] = "sideways"
    cases.append(("bad-direction", bad_dir))

    bad_agg = _base()
    bad_agg["objective"]["success_criteria"][0]["binding"]["aggregation"] = "stddev"
    cases.append(("bad-aggregation", bad_agg))

    neg_tol = _base()
    neg_tol["objective"]["success_criteria"][0]["binding"]["tolerance"] = -1.0
    cases.append(("negative-tolerance", neg_tol))

    full_valid = _base()
    full_valid["objective"]["success_criteria"][0]["weight"] = 0.5
    full_valid["objective"]["success_criteria"][0]["required"] = False
    full_valid["objective"]["success_criteria"][0]["deadline_s"] = 2.5e6
    full_valid["objective"]["success_criteria"][0]["binding"]["threshold"] = 0.8
    full_valid["objective"]["success_criteria"][0]["binding"]["evaluation_window"] = {
        "kind": "rolling",
        "duration_s": 2.551e6,
    }
    full_valid["objective"]["provenance"] = {"seed": 1, "input_hashes": ["sha256:00"]}
    cases.append(("full-valid", full_valid))

    null_window = _base()
    null_window["objective"]["success_criteria"][0]["binding"]["evaluation_window"] = None
    null_window["objective"]["success_criteria"][0]["deadline_s"] = None
    cases.append(("null-timing", null_window))

    bad_window = _base()
    bad_window["objective"]["success_criteria"][0]["binding"]["evaluation_window"] = {
        "kind": "yearly"
    }
    cases.append(("bad-window-kind", bad_window))

    zero_deadline = _base()
    zero_deadline["objective"]["success_criteria"][0]["deadline_s"] = 0.0
    cases.append(("zero-deadline", zero_deadline))

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
