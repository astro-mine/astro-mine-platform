"""Drift guards between the canonical JSON Schema and the Pydantic models.

The JSON Schema is canonical; the Pydantic models are hand-written to mirror it
(until RM-P0-CORE-07 generates one from the other). Two guards keep them aligned:

1. every enum ``$def`` in the schema has exactly the members of the matching Python
   enum;
2. a corpus of valid/invalid documents gets the *same* structural verdict from the
   JSON Schema validator and from Pydantic. (Pydantic carries no cross-reference or
   dual-use semantics — those live in the loader — so the two are a single
   structural contract here.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.sadf import enums
from astro_mine.core.sadf.loader import load_schema
from astro_mine.core.sadf.model import SadfDocument

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples" / "assets").glob("*.sadf.yaml"))

ENUM_DEFS = {
    "CapabilityTag": enums.CapabilityTag,
    "Regime": enums.Regime,
    "PropulsionKind": enums.PropulsionKind,
    "PropellantType": enums.PropellantType,
    "PowerSourceKind": enums.PowerSourceKind,
    "SensorKind": enums.SensorKind,
    "CommsBand": enums.CommsBand,
    "CommsProtocol": enums.CommsProtocol,
    "NodeRole": enums.NodeRole,
    "JointType": enums.JointType,
    "ContactElementKind": enums.ContactElementKind,
    "GeometryRole": enums.GeometryRole,
    "GeometryFormat": enums.GeometryFormat,
    "FidelityTier": enums.FidelityTier,
    "SurrogatePhysicsDomain": enums.SurrogatePhysicsDomain,
    "DeterminismClass": enums.DeterminismClass,
    "ReturnKind": enums.ReturnKind,
    "EarthInterfaceMode": enums.EarthInterfaceMode,
    "FlightStack": enums.FlightStack,
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
        SadfDocument.model_validate(data)
    except ValidationError:
        return False
    return True


def _base() -> dict[str, Any]:
    return {
        "sadf_version": "0.1",
        "asset": {
            "identity": {"id": "a", "name": "A", "version": "0.1.0", "kind": "rover"},
            "root_frame": "body",
            "frames": [{"name": "body"}],
        },
    }


def _corpus() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [("minimal-valid", _base())]

    unknown_top = _base()
    unknown_top["extra"] = 1
    cases.append(("unknown-top-level", unknown_top))

    unknown_asset = _base()
    unknown_asset["asset"]["bogus"] = 1
    cases.append(("unknown-asset-field", unknown_asset))

    typo = _base()
    typo["asset"]["identity"]["nam"] = "x"
    cases.append(("identity-typo", typo))

    missing_asset = {"sadf_version": "0.1"}
    cases.append(("missing-asset", missing_asset))

    missing_version = {"asset": _base()["asset"]}
    cases.append(("missing-version", missing_version))

    bad_const = _base()
    bad_const["sadf_version"] = "0.2"
    cases.append(("bad-version-const", bad_const))

    bad_cap = _base()
    bad_cap["asset"]["capabilities"] = ["mobility.teleport"]
    cases.append(("bad-capability", bad_cap))

    bad_prop_kind = _base()
    bad_prop_kind["asset"]["propulsion"] = {"systems": [{"kind": "antimatter"}]}
    cases.append(("bad-propulsion-kind", bad_prop_kind))

    wrong_type = _base()
    wrong_type["asset"]["geometry"] = [
        {"role": "visual", "format": "usd", "uri": [], "frame": "body"}
    ]
    cases.append(("wrong-scalar-type", wrong_type))

    old_civ = _base()
    old_civ["asset"]["core_interface_versions"] = ["0.1"]  # pre-B2 list shape, now invalid
    cases.append(("old-list-core-interface", old_civ))

    dict_civ = _base()
    dict_civ["asset"]["core_interface_versions"] = {"sadf": "0.1.0"}
    cases.append(("dict-core-interface", dict_civ))

    full_valid = _base()
    full_valid["asset"]["propulsion"] = {
        "systems": [{"kind": "electric_ion", "isp_s": 3000.0}],
        "delta_v_budget_mps": 1500.0,
    }
    full_valid["asset"]["return"] = {"capability": "sample_canister"}
    cases.append(("full-valid-with-return", full_valid))

    for path in EXAMPLES:
        cases.append((path.name, yaml.safe_load(path.read_text())))

    return cases


@pytest.mark.parametrize("case", _corpus(), ids=lambda c: c[0])
def test_jsonschema_and_pydantic_agree(case: tuple[str, dict[str, Any]]) -> None:
    _, data = case
    assert _jsonschema_ok(data) == _pydantic_ok(data)


def test_schema_is_self_consistent_json() -> None:
    # The shipped schema parses and is a valid Draft 2020-12 document.
    raw = json.dumps(load_schema())
    Draft202012Validator.check_schema(json.loads(raw))
