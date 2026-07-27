"""Fail-closed load of a served surrogate + Hub round-trip (RM-P1-SURR-04; surrogate.md §9).

The signed manifest is verified before the ONNX artifact is trusted. This exercises the real gate —
Core's ``PluginRegistry`` signature check plus the bundle-hash and ErrorReport-hash bindings — and
the publish → resolve → verify → load round trip through a local Hub registry. Everything fails
closed: unsigned, tampered artifact, tampered report, and untrusted key each refuse the load.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from astro_mine.core.registry import UnsignedManifest
from astro_mine.hub.registry import Registry
from astro_mine.hub.supply_chain import SignatureError, make_verifier, sign_digest
from astro_mine.surrogate.enums import ServedBackend
from astro_mine.surrogate.manifest import build_surrogate_manifest
from astro_mine.surrogate.serve import (
    OnnxServedSurrogate,
    ServedIntegrityError,
    load_served_surrogate,
    publish_served_surrogate,
    resolve_and_load,
)


def _signed_manifest(bundle, private_pem, *, artifact_digest=None):
    manifest = build_surrogate_manifest(
        name="excavation-gns",
        version="0.1.0",
        report=bundle.error_report,
        artifact_digest=artifact_digest or bundle.content_hash(),
        served_backend=ServedBackend.ONNX,
    )
    return manifest.model_copy(
        update={"signature": sign_digest(manifest.provenance.digest, private_pem)}
    )


def test_publish_resolve_load_round_trip(served_bundle, served_query, keypair) -> None:
    private_pem, public_pem = keypair
    with tempfile.TemporaryDirectory() as path:
        registry = Registry(path)
        published = publish_served_surrogate(
            served_bundle,
            registry,
            name="excavation-gns",
            version="0.1.0",
            private_key_pem=private_pem,
        )
        assert published.artifact_digest == served_bundle.content_hash()
        loaded = resolve_and_load(
            registry,
            "excavation-gns:0.1.0",
            verifier=make_verifier(trusted_public_key_pem=public_pem),
        )
        # The resolved-and-verified tier predicts identically to the local bundle.
        local = OnnxServedSurrogate(served_bundle)
        assert np.array_equal(
            loaded.predict(served_query).fields["position"],
            local.predict(served_query).fields["position"],
        )
        assert loaded.error_report.content_hash() == served_bundle.error_report.content_hash()


def test_published_surrogate_carries_verifiable_attestations(served_bundle, keypair) -> None:
    """A published tier must be verifiable by Hub, not merely signed inside its own manifest.

    Publishing used to go straight to `registry.publish`, which only stores bytes — so no cosign
    signature, SLSA provenance, or SBOM referrer was ever attached. The artifact read as signed to
    a human (the manifest carries a `signature`) and as *unsigned* to the verifier, could not pass
    Hub's admission gate, and failed every default pull (astro-mine-surrogate#28).
    """
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.supply_chain import (
        ARTIFACT_TYPE_SBOM,
        ARTIFACT_TYPE_SIGNATURE,
        ARTIFACT_TYPE_SLSA,
    )
    from astro_mine.hub.supply_chain import verify as supply_verify

    private_pem, public_pem = keypair
    with tempfile.TemporaryDirectory() as path:
        registry = Registry(path)
        published = publish_served_surrogate(
            served_bundle,
            registry,
            name="excavation-gns",
            version="0.1.0",
            private_key_pem=private_pem,
            train_dataset_hash="sha256:" + "d" * 64,
            sampling_policy_hash="sha256:" + "5" * 64,
        )

        # The evidence a verifier actually reads is attached.
        kinds = {d.artifact_type for d in registry.referrers(published.manifest_digest)}
        assert {ARTIFACT_TYPE_SIGNATURE, ARTIFACT_TYPE_SLSA, ARTIFACT_TYPE_SBOM} <= kinds

        # And it verifies fail-closed under the default requirement set — no `require=()` escape.
        supply_verify(registry, published.manifest_digest, trusted_public_key_pem=public_pem)

        # A default pull returns bytes rather than raising.
        assert HubClient(registry, trusted_public_key_pem=public_pem).pull("excavation-gns:0.1.0")


def test_unsigned_manifest_is_refused(served_bundle, keypair) -> None:
    _, public_pem = keypair
    unsigned = build_surrogate_manifest(
        name="excavation-gns",
        version="0.1.0",
        report=served_bundle.error_report,
        artifact_digest=served_bundle.content_hash(),
        served_backend=ServedBackend.ONNX,
    )
    with pytest.raises(UnsignedManifest):
        load_served_surrogate(
            bundle_bytes=served_bundle.serialize(),
            manifest=unsigned,
            verifier=make_verifier(trusted_public_key_pem=public_pem),
        )


def test_untrusted_key_fails_closed(served_bundle, keypair) -> None:
    private_pem, _ = keypair
    from astro_mine.hub.supply_chain import generate_keypair

    _, other_public = generate_keypair()
    signed = _signed_manifest(served_bundle, private_pem)
    with pytest.raises(SignatureError):
        load_served_surrogate(
            bundle_bytes=served_bundle.serialize(),
            manifest=signed,
            verifier=make_verifier(trusted_public_key_pem=other_public),
        )


def test_tampered_bundle_fails_closed(served_bundle, keypair) -> None:
    private_pem, public_pem = keypair
    signed = _signed_manifest(served_bundle, private_pem)
    with pytest.raises(ServedIntegrityError, match="bundle hash"):
        load_served_surrogate(
            bundle_bytes=served_bundle.serialize() + b"tamper",
            manifest=signed,
            verifier=make_verifier(trusted_public_key_pem=public_pem),
        )


def test_manifest_referencing_a_different_report_fails_closed(served_bundle, keypair) -> None:
    """A signed manifest whose error_report_digest does not match the bundle's report is refused."""
    private_pem, public_pem = keypair
    from tests.surrogate.factories import granular_report

    # Sign a manifest built from a *different* report but pointing at this bundle's bytes.
    other = build_surrogate_manifest(
        name="excavation-gns",
        version="0.1.0",
        report=granular_report(),
        artifact_digest=served_bundle.content_hash(),
        served_backend=ServedBackend.ONNX,
    )
    signed = other.model_copy(
        update={"signature": sign_digest(other.provenance.digest, private_pem)}
    )
    with pytest.raises(ServedIntegrityError, match="ErrorReport"):
        load_served_surrogate(
            bundle_bytes=served_bundle.serialize(),
            manifest=signed,
            verifier=make_verifier(trusted_public_key_pem=public_pem),
        )
