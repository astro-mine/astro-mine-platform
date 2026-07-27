"""Cosign-verified-images-only admission -- the supply-chain trust boundary."""

from __future__ import annotations

from astro_mine.cloud.tenancy.admission import (
    ImageAttestation,
    admit,
    cosign_cluster_policy,
)

SIGNED = ImageAttestation(
    image="ghcr.io/astro-mine/x@sha256:" + "a" * 64,
    cosign_verified=True,
    slsa_provenance=True,
    sbom=True,
    core_interface_version="0.1.0",
)


def test_fully_attested_image_is_admitted() -> None:
    decision = admit(SIGNED)
    assert decision.admitted is True
    assert decision.reasons == []


def test_unsigned_image_is_refused() -> None:
    decision = admit(SIGNED.model_copy(update={"cosign_verified": False}))
    assert decision.admitted is False
    assert "not cosign-verified" in " ".join(decision.reasons)


def test_incompatible_core_version_is_refused() -> None:
    decision = admit(SIGNED.model_copy(update={"core_interface_version": "2.0.0"}))
    assert decision.admitted is False
    assert any("Core interface version" in r for r in decision.reasons)


def test_undeclared_core_version_is_refused_by_default() -> None:
    decision = admit(SIGNED.model_copy(update={"core_interface_version": None}))
    assert decision.admitted is False
    assert any("no declared Core interface version" in r for r in decision.reasons)


def test_unpinned_image_is_refused() -> None:
    decision = admit(SIGNED.model_copy(update={"image": "ghcr.io/astro-mine/x:latest"}))
    assert decision.admitted is False
    assert any("not digest-pinned" in r for r in decision.reasons)


def test_missing_provenance_or_sbom_is_refused() -> None:
    no_slsa = admit(SIGNED.model_copy(update={"slsa_provenance": False}))
    assert "missing SLSA provenance" in " ".join(no_slsa.reasons)
    no_sbom = admit(SIGNED.model_copy(update={"sbom": False}))
    assert "missing SBOM" in " ".join(no_sbom.reasons)


def test_requirements_can_be_relaxed() -> None:
    lax = ImageAttestation(image="ghcr.io/astro-mine/x@sha256:" + "a" * 64, cosign_verified=True)
    decision = admit(lax, require_slsa=False, require_sbom=False, require_core_version=False)
    assert decision.admitted is True


def test_cosign_cluster_policy_enforces_signatures_and_attestations() -> None:
    policy = cosign_cluster_policy()
    assert policy["kind"] == "ClusterPolicy"
    assert policy["spec"]["validationFailureAction"] == "Enforce"
    rule = policy["spec"]["rules"][0]["verifyImages"][0]
    predicate_types = {a["predicateType"] for a in rule["attestations"]}
    assert any("slsa.dev/provenance" in p for p in predicate_types)
    assert any("cyclonedx.org/bom" in p for p in predicate_types)
