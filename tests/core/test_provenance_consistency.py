"""Drift guard between the canonical RunProvenance JSON Schema and the Pydantic models.

The JSON Schema is canonical; the Pydantic models are hand-written to mirror it. As with
``tests/test_objective_consistency.py``, a corpus of valid/invalid documents must get the
*same* structural verdict from the JSON Schema validator and from Pydantic (run provenance
has no closed enum vocabulary, so there is no enum-parity guard here). The field-set drift
guard lives in ``scripts/check_model_drift.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.provenance.loader import load_schema
from astro_mine.core.provenance.model import RunProvenanceDocument

EXAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "examples" / "run-provenance").glob(
        "*.run-provenance.yaml"
    )
)


def _jsonschema_ok(data: Any) -> bool:
    return not list(Draft202012Validator(load_schema()).iter_errors(data))


def _pydantic_ok(data: Any) -> bool:
    try:
        RunProvenanceDocument.model_validate(data)
    except ValidationError:
        return False
    return True


def _base() -> dict[str, Any]:
    return {"run_provenance_version": "0.1", "run_provenance": {}}


def _corpus() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [("minimal-valid", _base())]

    unknown_top = _base()
    unknown_top["extra"] = 1
    cases.append(("unknown-top-level", unknown_top))

    unknown_field = _base()
    unknown_field["run_provenance"]["bogus"] = 1
    cases.append(("unknown-provenance-field", unknown_field))

    missing_payload = {"run_provenance_version": "0.1"}
    cases.append(("missing-payload", missing_payload))

    missing_version = {"run_provenance": {}}
    cases.append(("missing-version", missing_version))

    bad_const = _base()
    bad_const["run_provenance_version"] = "0.2"
    cases.append(("bad-version-const", bad_const))

    bad_seed = _base()
    bad_seed["run_provenance"]["seed"] = "not-an-int"
    cases.append(("bad-seed-type", bad_seed))

    bad_seeds_value = _base()
    bad_seeds_value["run_provenance"]["seeds"] = {"episode": "x"}
    cases.append(("bad-seeds-value", bad_seeds_value))

    bad_engine_value = _base()
    bad_engine_value["run_provenance"]["engine_versions"] = {"orbital": 3}
    cases.append(("bad-engine-version-value", bad_engine_value))

    outcome_missing_verdict = _base()
    outcome_missing_verdict["run_provenance"]["error_budget_outcomes"] = [{"name": "x"}]
    cases.append(("outcome-missing-verdict", outcome_missing_verdict))

    outcome_unknown_field = _base()
    outcome_unknown_field["run_provenance"]["error_budget_outcomes"] = [
        {"name": "x", "within_budget": True, "oops": 1}
    ]
    cases.append(("outcome-unknown-field", outcome_unknown_field))

    full_valid = _base()
    full_valid["run_provenance"] = {
        "run_id": "r1",
        "input_hashes": ["sha256:00"],
        "engine_versions": {"orbital": "basilisk-2.2.0"},
        "fidelity_tiers": {"orbital": "kinematic"},
        "seed": 42,
        "seeds": {"episode": 42},
        "code_version": "astro-mine-sim 0.1.0",
        "toolchain_version": "python-3.12.4",
        "env_lockfile": "sha256:22",
        "error_budget_outcomes": [
            {
                "name": "orbital_position",
                "within_budget": True,
                "tier": "kinematic",
                "metric": "position_rms_m",
                "value": 12.5,
                "tolerance": 50.0,
            }
        ],
    }
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
