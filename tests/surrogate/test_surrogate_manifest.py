"""SurrogateManifest builder + Core registry integration (RM-P1-SURR-01).

Proves a surrogate publishes itself as a valid Core PluginManifest that passes the
registry's validity + version-negotiation gates, and that the *same* builder serves both
consumers — a granular step surrogate (regime_engine / env, Sim) and a learned
illumination field surrogate (field_model / world_provider, Worlds).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.core.registry import IncompatibleManifest, PluginKind, PluginRegistry
from astro_mine.surrogate import ServedBackend, build_surrogate_manifest
from astro_mine.surrogate.manifest import SurrogateAttributes
from tests.surrogate.factories import granular_report, illumination_report

_DIGEST = "sha256:" + "cd" * 32


def test_granular_surrogate_publishes_as_a_sim_regime_engine() -> None:
    report = granular_report()
    manifest = build_surrogate_manifest(
        name="excavation-gnn", version="0.1.0", report=report, artifact_digest=_DIGEST
    )
    assert manifest.kind is PluginKind.REGIME_ENGINE
    assert manifest.core_interfaces == {"env": "0.1.0"}
    assert manifest.provenance is not None
    assert manifest.provenance.digest == _DIGEST
    assert manifest.provenance.input_hashes == [report.validation_dataset_hash]
    assert manifest.license == "Apache-2.0"
    assert manifest.capability_tags == []  # open commons — no gated tag


def test_illumination_surrogate_publishes_as_a_worlds_field_model() -> None:
    manifest = build_surrogate_manifest(
        name="illumination-surrogate",
        version="0.1.0",
        report=illumination_report(),
        artifact_digest=_DIGEST,
    )
    assert manifest.kind is PluginKind.FIELD_MODEL
    assert manifest.core_interfaces == {"world_provider": "0.1.0"}


def test_attributes_carry_the_surrogate_facets_and_error_report_digest() -> None:
    report = granular_report()
    manifest = build_surrogate_manifest(
        name="excavation-gnn",
        version="0.1.0",
        report=report,
        artifact_digest=_DIGEST,
        served_backend=ServedBackend.NATIVE_GRAPH,
        native_graph_fallback=True,
    )
    attrs = manifest.attributes
    assert attrs["domain"] == "granular_excavation"
    assert attrs["served_backend"] == "native_graph"
    assert attrs["native_graph_fallback"] is True
    # The manifest commits to the exact bound by referencing the ErrorReport by content hash.
    assert attrs["error_report_digest"] == report.content_hash()
    # The admission budget is projected from the report so a scheduler can decide from the manifest.
    assert attrs["recommended_error_budget"] == dict(
        report.substitution_policy.recommended_error_budget
    )
    assert set(attrs["output_channels"]) == {"reaction_force_n"}
    assert set(attrs["input_channels"]) == set(report.trust_region.bounds)
    assert "bounds" in attrs["trust_region"]


def test_manifest_passes_the_core_registry_validity_and_negotiation_gates() -> None:
    manifest = build_surrogate_manifest(
        name="excavation-gnn", version="0.1.0", report=granular_report(), artifact_digest=_DIGEST
    )
    registry = PluginRegistry(require_signature=False)  # signing is exercised separately
    registered = registry.register(manifest)
    assert registered.name == "excavation-gnn"
    assert registry.by_kind(PluginKind.REGIME_ENGINE) == [registered]


def test_registry_refuses_an_incompatible_core_interface_version() -> None:
    manifest = build_surrogate_manifest(
        name="excavation-gnn", version="0.1.0", report=granular_report(), artifact_digest=_DIGEST
    )
    # A 0.y minor bump may break: env 0.2.0 is not satisfied by this Core's 0.1.0.
    bad = manifest.model_copy(update={"core_interfaces": {"env": "0.2.0"}})
    with pytest.raises(IncompatibleManifest):
        PluginRegistry(require_signature=False).register(bad)


def test_surrogate_attributes_are_frozen_and_reject_unknown_fields() -> None:
    report = granular_report()
    attrs = SurrogateAttributes(
        domain=report.domain,
        input_channels=list(report.trust_region.bounds),
        output_channels=[c.channel for c in report.channels],
        trust_region=report.trust_region,
        recommended_error_budget=dict(report.substitution_policy.recommended_error_budget),
        served_backend=ServedBackend.ONNX,
        native_graph_fallback=False,
        error_report_digest=report.content_hash(),
    )
    with pytest.raises(ValidationError):
        attrs.domain = report.domain  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SurrogateAttributes(  # type: ignore[call-arg]
            domain=report.domain,
            input_channels=["x"],
            output_channels=["y"],
            trust_region=report.trust_region,
            served_backend=ServedBackend.ONNX,
            native_graph_fallback=False,
            error_report_digest="sha256:x",
            surprise="nope",
        )
