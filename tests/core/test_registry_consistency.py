"""Drift guards between the canonical plugin-manifest JSON Schema and the Pydantic models.

The JSON Schema is canonical; the Pydantic models mirror it (until RM-P0-CORE-07
generates one from the other). Two guards keep them aligned, mirroring
``tests/test_objective_consistency.py``:

1. every enum ``$def`` in the schema has exactly the members of the matching Python enum
   (including the SADF-owned vocabularies the manifest reuses);
2. a corpus of valid/invalid documents gets the *same* structural verdict from the JSON
   Schema validator and from Pydantic (semantics — the gated-tag and SemVer gates — live
   in the loader, so the two are a single structural contract here).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.registry import enums
from astro_mine.core.registry.loader import load_schema
from astro_mine.core.registry.model import ManifestDocument

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples" / "plugins").glob("*.yaml"))

ENUM_DEFS = {
    "PluginKind": enums.PluginKind,
    "SignatureScheme": enums.SignatureScheme,
    "SignatureKind": enums.SignatureKind,
    "DeterminismClass": enums.DeterminismClass,
    "Regime": enums.Regime,
    "CapabilityTag": enums.CapabilityTag,
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
        ManifestDocument.model_validate(data)
    except ValidationError:
        return False
    return True


def _base() -> dict[str, Any]:
    return {
        "manifest_version": "0.1",
        "manifest": {
            "name": "engine",
            "version": "0.1.0",
            "kind": "regime_engine",
        },
    }


def _corpus() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [("minimal-valid", _base())]

    unknown_top = _base()
    unknown_top["extra"] = 1
    cases.append(("unknown-top-level", unknown_top))

    unknown_field = _base()
    unknown_field["manifest"]["bogus"] = 1
    cases.append(("unknown-manifest-field", unknown_field))

    bad_kind = _base()
    bad_kind["manifest"]["kind"] = "warp_drive"
    cases.append(("bad-kind", bad_kind))

    missing_manifest = {"manifest_version": "0.1"}
    cases.append(("missing-manifest", missing_manifest))

    missing_version = {"manifest": _base()["manifest"]}
    cases.append(("missing-version", missing_version))

    bad_const = _base()
    bad_const["manifest_version"] = "0.2"
    cases.append(("bad-version-const", bad_const))

    missing_name = _base()
    del missing_name["manifest"]["name"]
    cases.append(("missing-name", missing_name))

    bad_cap = _base()
    bad_cap["manifest"]["capability_tags"] = ["mobility.wheeled", "not.a.tag"]
    cases.append(("bad-capability-tag", bad_cap))

    bad_determinism = _base()
    bad_determinism["manifest"]["determinism_class"] = "perfect"
    cases.append(("bad-determinism", bad_determinism))

    bad_regime = _base()
    bad_regime["manifest"]["regimes"] = ["hyperspace"]
    cases.append(("bad-regime", bad_regime))

    bad_sig_scheme = _base()
    bad_sig_scheme["manifest"]["signature"] = {"scheme": "pgp"}
    cases.append(("bad-signature-scheme", bad_sig_scheme))

    sig_missing_scheme = _base()
    sig_missing_scheme["manifest"]["signature"] = {"value": "x"}
    cases.append(("signature-missing-scheme", sig_missing_scheme))

    bad_core_interfaces = _base()
    bad_core_interfaces["manifest"]["core_interfaces"] = "not-a-map"
    cases.append(("bad-core-interfaces", bad_core_interfaces))

    full_valid = _base()
    full_valid["manifest"].update(
        {
            "core_interfaces": {"env": "0.1.0", "sadf": "0.1.0"},
            "inputs": ["StateSample", "ActionBatch"],
            "outputs": ["StateSample"],
            "capability_tags": ["mobility.wheeled", "sensing.odometry"],
            "determinism_class": "tolerance",
            "regimes": ["surface"],
            "description": "A reduced-order terramechanics engine.",
            "provenance": {
                "input_hashes": ["sha256:00"],
                "code_version": "0.1.0",
                "toolchain_version": "python-3.12",
                "env_lockfile": "uv.lock",
                "seed": 7,
                "source_content_hashes": {"cad": "sha256:aa"},
                "builder_version": "conv-1",
                "digest": "sha256:bb",
            },
            "signature": {
                "scheme": "sigstore_cosign",
                "value": "MEUC",
                "payload": "eyJ9",
                "certificate": "-----BEGIN CERTIFICATE-----",
            },
            "attributes": {"fidelity_tier": "reduced_order", "error_budget": 0.1},
        }
    )
    cases.append(("full-valid", full_valid))

    null_optionals = _base()
    null_optionals["manifest"].update(
        {
            "description": None,
            "determinism_class": None,
            "provenance": None,
            "signature": None,
        }
    )
    cases.append(("null-optionals", null_optionals))

    license_and_signatures = _base()
    license_and_signatures["manifest"].update(
        {
            "license": "Apache-2.0",
            "signatures": [
                {"scheme": "sigstore_cosign", "kind": "cosign_signature", "digest": "sha256:cc"},
                {"scheme": "sigstore_cosign", "kind": "slsa_provenance", "digest": "sha256:dd"},
                {"scheme": "sigstore_cosign", "kind": "sbom", "digest": "sha256:ee"},
            ],
        }
    )
    cases.append(("license-and-signatures", license_and_signatures))

    bad_sig_kind = _base()
    bad_sig_kind["manifest"]["signatures"] = [{"scheme": "sigstore_cosign", "kind": "pgp"}]
    cases.append(("bad-signature-kind", bad_sig_kind))

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
