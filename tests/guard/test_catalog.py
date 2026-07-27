"""Core-catalog registration: the SafetySpec manifest registers and negotiates Core versions."""

from __future__ import annotations

import pytest

from astro_mine.core.compat import CORE_INTERFACE_VERSIONS
from astro_mine.core.registry import PluginKind, PluginRegistry
from astro_mine.core.registry.registry import IncompatibleManifest
from astro_mine.guard.spec import (
    COMPILED_MODEL_OUTPUT,
    SAFETY_SPEC_INTERFACE_VERSIONS,
    SAFETY_SPEC_OUTPUT,
    SafetyDocument,
    build_safety_manifest,
    register_safety_spec,
)


def test_manifest_shape(anchor_document: SafetyDocument) -> None:
    man = build_safety_manifest(anchor_document)
    assert man.name == "anchor-lunar-polar-v0"
    assert man.kind == PluginKind.POLICY
    assert man.license == "Apache-2.0"
    assert man.outputs == [SAFETY_SPEC_OUTPUT, COMPILED_MODEL_OUTPUT]
    # provenance pins the spec by content hash
    assert man.provenance is not None
    assert man.provenance.digest == anchor_document.content_hash()
    # open-commons: no gated capability tags
    assert man.capability_tags == []


def test_declared_interfaces_are_known_to_core() -> None:
    # every interface the manifest is built against exists in this Core.
    for interface in SAFETY_SPEC_INTERFACE_VERSIONS:
        assert interface in CORE_INTERFACE_VERSIONS


def test_registers_through_core_registry(anchor_document: SafetyDocument) -> None:
    reg = PluginRegistry(require_signature=False)
    registered = register_safety_spec(anchor_document, registry=reg)
    assert registered.name in reg
    assert reg.resolve("anchor-lunar-polar-v0").kind == PluginKind.POLICY


def test_default_registry_is_unsigned_dev(anchor_document: SafetyDocument) -> None:
    # No registry passed -> a fresh require_signature=False registry (signed loading is GUARD-05).
    man = register_safety_spec(anchor_document)
    assert man.provenance is not None


def test_negotiation_rejects_incompatible_core(anchor_document: SafetyDocument) -> None:
    # A Core that provides an incompatible messages minor refuses the load (fail loud).
    incompatible = dict(CORE_INTERFACE_VERSIONS)
    incompatible["messages"] = "0.2.0"
    reg = PluginRegistry(require_signature=False, provided=incompatible)
    with pytest.raises(IncompatibleManifest):
        register_safety_spec(anchor_document, registry=reg)


def test_catalog_record_projection(anchor_document: SafetyDocument) -> None:
    man = build_safety_manifest(anchor_document)
    record = man.to_catalog_record()
    assert record.outputs == [SAFETY_SPEC_OUTPUT, COMPILED_MODEL_OUTPUT]
    versions = {c.interface: c.version for c in record.core_interface_versions}
    assert versions == SAFETY_SPEC_INTERFACE_VERSIONS
