"""The allocation Core manifest + registry integration (RM-P1-ALLOC-01).

Proves Allocate publishes itself as a valid Core ``PluginManifest`` — a Policy/Planner
allocation sub-interface (``kind=policy``, no new PluginKind, no Core RFC) — that passes the
registry's validity + version-negotiation gates, and that an incompatible Core-interface bump
is refused.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.allocate import IR_VERSION, build_allocation_manifest
from astro_mine.allocate.api.manifest import AllocationAttributes
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.registry import IncompatibleManifest, PluginKind, PluginRegistry

_DIGEST = "sha256:" + "cd" * 32


def test_allocation_plugin_publishes_as_a_policy_sub_interface() -> None:
    manifest = build_allocation_manifest(
        name="cp-sat-allocator", version="0.1.0", artifact_digest=_DIGEST
    )
    assert manifest.kind is PluginKind.POLICY
    assert manifest.core_interfaces["policy"] == "0.1.0"
    assert manifest.core_interfaces["registry"] == "0.1.0"
    # Consumed contracts are declared too (TaskDirective/ActionBatch out; ObjectiveSpec in).
    assert manifest.core_interfaces["messages"] == "0.1.0"
    assert manifest.core_interfaces["objective"] == "0.1.0"
    assert manifest.capability_tags == []  # open commons — no gated tag
    assert manifest.license == "Apache-2.0"
    assert manifest.provenance is not None
    assert manifest.provenance.digest == _DIGEST


def test_attributes_carry_the_ir_version_backends_and_supported_kinds() -> None:
    manifest = build_allocation_manifest(
        name="cp-sat-allocator",
        version="0.1.0",
        artifact_digest=_DIGEST,
        backends=["cp-sat", "highs"],
        supported_task_kinds=[TaskKind.PROSPECT, TaskKind.EXCAVATE],
        seed=7,
        input_hashes=["sha256:" + "00" * 32],
    )
    attrs = manifest.attributes
    assert attrs["ir_version"] == IR_VERSION
    assert attrs["backends"] == ["cp-sat", "highs"]
    assert attrs["supported_task_kinds"] == ["prospect", "excavate"]
    assert attrs["deterministic"] is True
    assert manifest.provenance is not None
    assert manifest.provenance.seed == 7
    assert manifest.provenance.input_hashes == ["sha256:" + "00" * 32]


def test_default_supported_kinds_cover_the_schedulable_task_vocabulary() -> None:
    manifest = build_allocation_manifest(
        name="cp-sat-allocator", version="0.1.0", artifact_digest=_DIGEST
    )
    assert "prospect" in manifest.attributes["supported_task_kinds"]
    assert manifest.attributes["backends"] == ["trivial-stub"]


def test_manifest_passes_the_core_registry_validity_and_negotiation_gates() -> None:
    manifest = build_allocation_manifest(
        name="cp-sat-allocator", version="0.1.0", artifact_digest=_DIGEST
    )
    registry = PluginRegistry(require_signature=False)  # signing is exercised separately
    registered = registry.register(manifest)
    assert registered.name == "cp-sat-allocator"
    assert registry.by_kind(PluginKind.POLICY) == [registered]


def test_registry_refuses_an_incompatible_core_interface_version() -> None:
    manifest = build_allocation_manifest(
        name="cp-sat-allocator", version="0.1.0", artifact_digest=_DIGEST
    )
    # A 0.y minor bump may break: policy 0.2.0 is not satisfied by this Core's 0.1.0.
    bad = manifest.model_copy(update={"core_interfaces": {"policy": "0.2.0"}})
    with pytest.raises(IncompatibleManifest):
        PluginRegistry(require_signature=False).register(bad)


def test_allocation_attributes_are_frozen_and_reject_unknown_fields() -> None:
    attrs = AllocationAttributes(
        ir_version=IR_VERSION,
        supported_task_kinds=[TaskKind.PROSPECT],
        backends=["cp-sat"],
    )
    with pytest.raises(ValidationError):
        attrs.deterministic = False
    with pytest.raises(ValidationError):
        AllocationAttributes(  # type: ignore[call-arg]
            ir_version=IR_VERSION,
            supported_task_kinds=[TaskKind.PROSPECT],
            backends=["cp-sat"],
            surprise="nope",
        )
