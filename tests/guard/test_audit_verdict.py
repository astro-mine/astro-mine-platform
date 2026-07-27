"""SafetyVerdict — the Core-catalogued audit record: model, schema, wire, provenance (GUARD-06).

Proves the record mirrors its canonical JSON Schema, round-trips byte-stably through its additive
Protobuf wire form (including the ``+inf`` barrier margin of a fallback tick), and that its
reproducibility provenance excludes the wall-clock latency — so the same seeded run reproduces the
same verdict provenance across machines (guard.md §5, §6).
"""

from __future__ import annotations

import math

import pytest
from jsonschema import Draft202012Validator

from astro_mine.core.compat import CORE_INTERFACE_VERSIONS
from astro_mine.core.registry import PluginKind, PluginRegistry
from astro_mine.core.registry.registry import IncompatibleManifest
from astro_mine.guard.audit import (
    SAFETY_VERDICT_INTERFACE_VERSIONS,
    SAFETY_VERDICT_OUTPUT,
    build_verdict_manifest,
    load_schema,
    register_verdict_schema,
    verdict_from_wire,
    verdict_schema_content_hash,
    verdict_to_wire,
)
from astro_mine.guard.audit.model import VERDICT_VERSION
from tests.guard.conftest import make_verdict


def test_version_constant() -> None:
    assert VERDICT_VERSION == "0.1"
    assert make_verdict().verdict_version == "0.1"


def test_model_dump_validates_against_schema() -> None:
    verdict = make_verdict(constraint_ids=["c_anchor_torque"], backup_kind="brake_to_stop")
    Draft202012Validator(load_schema()).validate(verdict.model_dump(mode="json"))


def test_provenance_excludes_latency() -> None:
    a = make_verdict(shield_latency_us=3.0)
    b = make_verdict(shield_latency_us=999.0)
    assert a.provenance() == b.provenance()  # latency is not in the deterministic view
    assert a.content_hash() == b.content_hash()
    assert "shield_latency_us" not in a.provenance()


def test_content_hash_is_sha256() -> None:
    assert make_verdict().content_hash().startswith("sha256:")


def test_wire_roundtrip_finite() -> None:
    verdict = make_verdict(
        constraint_ids=["c_x", "c_y"],
        certified_action=[1.0, -2.0, 3.5],
        layer="shield",
        intervention="modified",
        reason="shield_corrected",
    )
    assert verdict_from_wire(verdict_to_wire(verdict)).provenance() == verdict.provenance()


def test_wire_roundtrip_infinity_margin() -> None:
    # A fallback tick legitimately has min_barrier_margin = +inf.
    verdict = make_verdict(
        min_barrier_margin=math.inf,
        backup_kind="brake_to_stop",
        reason="bad_input",
        layer="backup",
        intervention="fallback",
    )
    back = verdict_from_wire(verdict_to_wire(verdict))
    assert math.isinf(back.min_barrier_margin)
    assert back.provenance() == verdict.provenance()


def test_wire_roundtrip_nan_margin() -> None:
    # NaN is representable on the wire via the proto3 "NaN" token (the encode/decode fixup).
    verdict = make_verdict(min_barrier_margin=math.nan)
    back = verdict_from_wire(verdict_to_wire(verdict))
    assert math.isnan(back.min_barrier_margin)


def test_wire_roundtrip_backup_kind_none() -> None:
    verdict = make_verdict(backup_kind=None)
    back = verdict_from_wire(verdict_to_wire(verdict))
    assert back.backup_kind is None
    assert back.provenance() == verdict.provenance()


def test_wire_is_byte_stable() -> None:
    verdict = make_verdict(certified_action=[1.0, 2.0, 3.0])
    assert verdict_to_wire(verdict) == verdict_to_wire(verdict)


def test_load_schema_is_the_canonical_verdict() -> None:
    schema = load_schema()
    assert schema["title"] == "Astro-Mine SafetyVerdict v0.1"
    assert "shield_latency_us" in schema["properties"]


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValueError, match="extra"):
        make_verdict(surprise="x")


# --- Core-catalog registration ---------------------------------------------------------------


def test_verdict_manifest_shape() -> None:
    manifest = build_verdict_manifest()
    assert manifest.kind == PluginKind.POLICY
    assert manifest.license == "Apache-2.0"
    assert manifest.outputs == [SAFETY_VERDICT_OUTPUT]
    assert manifest.capability_tags == []  # open-commons: no gated tags
    assert manifest.provenance is not None
    assert manifest.provenance.digest == verdict_schema_content_hash()


def test_declared_interfaces_known_to_core() -> None:
    for interface in SAFETY_VERDICT_INTERFACE_VERSIONS:
        assert interface in CORE_INTERFACE_VERSIONS


def test_registers_through_core_registry() -> None:
    reg = PluginRegistry(require_signature=False)
    registered = register_verdict_schema(registry=reg)
    assert registered.name in reg


def test_default_registry_registers_unsigned() -> None:
    assert register_verdict_schema().provenance is not None


def test_negotiation_rejects_incompatible_core() -> None:
    incompatible = dict(CORE_INTERFACE_VERSIONS)
    incompatible["messages"] = "0.2.0"
    reg = PluginRegistry(require_signature=False, provided=incompatible)
    with pytest.raises(IncompatibleManifest):
        register_verdict_schema(registry=reg)
