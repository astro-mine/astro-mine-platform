"""Signed-manifest load gate (RM-P1-SURR-01 acceptance).

A SurrogateManifest is signed and verified before Sim would load it (surrogate.md §9,
mirrors Core's manifest-signing rule). This exercises the real path: Core's
``PluginRegistry`` signature gate + astro-mine-hub's keyed ECDSA P-256 verifier. Everything
fails closed — unsigned, tampered, or wrong-key all refuse the load.
"""

from __future__ import annotations

import pytest

from astro_mine.core.registry import PluginRegistry, UnsignedManifest
from astro_mine.hub.supply_chain import (
    SignatureError,
    generate_keypair,
    make_verifier,
    sign_digest,
)
from astro_mine.surrogate import build_surrogate_manifest
from tests.surrogate.factories import granular_report

_DIGEST = "sha256:" + "cd" * 32


def _manifest():
    return build_surrogate_manifest(
        name="excavation-gnn", version="0.1.0", report=granular_report(), artifact_digest=_DIGEST
    )


def test_a_signature_requiring_registry_refuses_an_unsigned_manifest() -> None:
    registry = PluginRegistry(require_signature=True)
    with pytest.raises(UnsignedManifest):
        registry.register(_manifest())


def test_a_signed_manifest_verifies_and_loads() -> None:
    private_pem, public_pem = generate_keypair()
    manifest = _manifest()
    signed = manifest.model_copy(
        update={"signature": sign_digest(manifest.provenance.digest, private_pem)}
    )
    registry = PluginRegistry(
        require_signature=True, verifier=make_verifier(trusted_public_key_pem=public_pem)
    )
    assert registry.register(signed).name == "excavation-gnn"


def test_a_tampered_digest_fails_closed() -> None:
    private_pem, public_pem = generate_keypair()
    manifest = _manifest()
    signed = manifest.model_copy(
        update={"signature": sign_digest(manifest.provenance.digest, private_pem)}
    )
    # Re-point the artifact digest after signing: the signature no longer covers it.
    tampered = signed.model_copy(
        update={
            "provenance": signed.provenance.model_copy(update={"digest": "sha256:" + "00" * 32})
        }
    )
    registry = PluginRegistry(
        require_signature=True, verifier=make_verifier(trusted_public_key_pem=public_pem)
    )
    with pytest.raises(SignatureError):
        registry.register(tampered)


def test_a_signature_from_an_untrusted_key_fails_closed() -> None:
    signer_private, _ = generate_keypair()
    _, other_public = generate_keypair()
    manifest = _manifest()
    signed = manifest.model_copy(
        update={"signature": sign_digest(manifest.provenance.digest, signer_private)}
    )
    registry = PluginRegistry(
        require_signature=True, verifier=make_verifier(trusted_public_key_pem=other_public)
    )
    with pytest.raises(SignatureError):
        registry.register(signed)
