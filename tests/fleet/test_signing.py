"""Asset signing + the Core Verifier hook (RM-P0-FLEET-06, CX-SEC)."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from astro_mine.core.registry import (
    PluginKind,
    PluginManifest,
    Provenance,
    Signature,
    SignatureScheme,
)
from astro_mine.fleet.packaging.verifier import make_verifier, sign_asset
from astro_mine.seal import SignatureError, generate_keypair

DIGEST = "sha256:" + "a" * 64


def _manifest(signature: Signature | None, *, digest: str = DIGEST) -> PluginManifest:
    return PluginManifest(
        name="test.rover",
        version="0.1.0",
        kind=PluginKind.ASSET,
        core_interfaces={"sadf": "0.1.0"},
        provenance=Provenance(digest=digest),
        signature=signature,
    )


def test_sign_produces_a_cosign_shaped_signature() -> None:
    private_pem, _ = generate_keypair()
    sig = sign_asset(DIGEST, private_pem)
    assert sig.scheme == SignatureScheme.SIGSTORE_COSIGN
    assert sig.payload == DIGEST
    assert sig.value and sig.certificate


def test_valid_signature_verifies() -> None:
    private_pem, public_pem = generate_keypair()
    manifest = _manifest(sign_asset(DIGEST, private_pem))
    make_verifier(trusted_public_key_pem=public_pem)(manifest)  # no raise


def test_verify_without_trusted_key_uses_embedded_cert() -> None:
    private_pem, _ = generate_keypair()
    make_verifier()(_manifest(sign_asset(DIGEST, private_pem)))  # no raise


def test_unsigned_or_wrong_scheme_is_rejected() -> None:
    verify = make_verifier()
    with pytest.raises(SignatureError, match="not signed with an astro-mine"):
        verify(_manifest(None))
    with pytest.raises(SignatureError, match="not signed with an astro-mine"):
        verify(_manifest(Signature(scheme=SignatureScheme.UNSIGNED)))


def test_incomplete_signature_block_is_rejected() -> None:
    sig = Signature(scheme=SignatureScheme.SIGSTORE_COSIGN, value="x", payload=DIGEST)  # no cert
    with pytest.raises(SignatureError, match="incomplete signature block"):
        make_verifier()(_manifest(sig))


def test_payload_must_match_the_artifact_digest() -> None:
    private_pem, public_pem = generate_keypair()
    manifest = _manifest(sign_asset(DIGEST, private_pem), digest="sha256:" + "b" * 64)
    with pytest.raises(SignatureError, match="payload does not match"):
        make_verifier(trusted_public_key_pem=public_pem)(manifest)


def test_untrusted_key_is_rejected() -> None:
    private_pem, _ = generate_keypair()
    _, other_public = generate_keypair()
    manifest = _manifest(sign_asset(DIGEST, private_pem))
    with pytest.raises(SignatureError, match="not the trusted key"):
        make_verifier(trusted_public_key_pem=other_public)(manifest)


def test_tampered_signature_does_not_verify() -> None:
    private_pem, public_pem = generate_keypair()
    sig = sign_asset(DIGEST, private_pem)
    tampered = sig.model_copy(update={"value": "AAAA" + (sig.value or "")[4:]})
    with pytest.raises(SignatureError, match="does not verify"):
        make_verifier(trusted_public_key_pem=public_pem)(_manifest(tampered))


def test_non_ecdsa_key_is_rejected() -> None:
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsa_pem = rsa_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with pytest.raises(SignatureError, match="not an ECDSA private key"):
        sign_asset(DIGEST, rsa_pem)


def test_malformed_public_key_in_signature_is_rejected() -> None:
    sig = Signature(
        scheme=SignatureScheme.SIGSTORE_COSIGN, value="AAAA", payload=DIGEST, certificate="garbage"
    )
    with pytest.raises(SignatureError, match="malformed public key"):
        make_verifier()(_manifest(sig))


def test_non_ecdsa_public_key_in_signature_is_rejected() -> None:
    rsa_public = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    rsa_pub_pem = rsa_public.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    sig = Signature(
        scheme=SignatureScheme.SIGSTORE_COSIGN,
        value="AAAA",
        payload=DIGEST,
        certificate=rsa_pub_pem,
    )
    with pytest.raises(SignatureError, match="not an ECDSA public key"):
        make_verifier()(_manifest(sig))
