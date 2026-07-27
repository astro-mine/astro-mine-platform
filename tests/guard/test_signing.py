"""The cosign ECDSA-P256 signing primitive (RM-P1-GUARD-05; guard.md §9.5, conventions §9).

Mirrors astro-mine-hub's signer on the Core :class:`~astro_mine.core.registry.Signature` type. Every
branch here is fail-closed: a wrong signer, a tampered payload, a mismatched digest, or a malformed
key must all refuse to verify.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from astro_mine.core.registry import Signature, SignatureKind, SignatureScheme
from astro_mine.guard.spec.keyid import signer_id
from astro_mine.seal import (
    SignatureError,
    generate_keypair,
    sign_digest,
    verify_signature,
)

_DEV_KEYS = Path("/mnt/d/MyProjects/AstroMine/files/hub-registry/keys")
_DIGEST = "sha256:" + "a" * 64
_OTHER = "sha256:" + "b" * 64


def test_generate_and_round_trip() -> None:
    private_pem, public_pem = generate_keypair()
    sig = sign_digest(_DIGEST, private_pem)
    assert sig.scheme == SignatureScheme.SIGSTORE_COSIGN
    assert sig.kind == SignatureKind.COSIGN_SIGNATURE
    assert sig.payload == _DIGEST
    # verifies against the artifact digest, and against the pinned public key
    verify_signature(sig, _DIGEST)
    verify_signature(sig, _DIGEST, trusted_public_key_pem=public_pem)


@pytest.mark.skipif(
    not (_DEV_KEYS / "anchor-dev.key.pem").exists(),
    reason="workspace dev keypair (files/hub-registry/keys) not present, e.g. a CI checkout",
)
def test_dev_keypair_signs_and_verifies() -> None:
    # The offline dev keypair the workspace convention ships (files/hub-registry/keys) round-trips.
    private_pem = (_DEV_KEYS / "anchor-dev.key.pem").read_bytes()
    public_pem = (_DEV_KEYS / "anchor-dev.pub.pem").read_bytes()
    sig = sign_digest(_DIGEST, private_pem)
    verify_signature(sig, _DIGEST, trusted_public_key_pem=public_pem)


def test_rejects_wrong_digest() -> None:
    private_pem, _ = generate_keypair()
    sig = sign_digest(_DIGEST, private_pem)
    # A valid signature over a DIFFERENT digest must not verify against this artifact.
    with pytest.raises(SignatureError, match="payload does not match"):
        verify_signature(sig, _OTHER)


def test_rejects_untrusted_signer() -> None:
    private_pem, _ = generate_keypair()
    _, other_public = generate_keypair()
    sig = sign_digest(_DIGEST, private_pem)
    with pytest.raises(SignatureError, match="not the trusted key"):
        verify_signature(sig, _DIGEST, trusted_public_key_pem=other_public)


def test_rejects_tampered_signature_value() -> None:
    private_pem, _ = generate_keypair()
    sig = sign_digest(_DIGEST, private_pem)
    raw = bytearray(base64.b64decode(sig.value or ""))
    raw[-1] ^= 0xFF  # flip a byte of the DER signature
    tampered = sig.model_copy(update={"value": base64.b64encode(bytes(raw)).decode()})
    with pytest.raises(SignatureError, match="does not verify"):
        verify_signature(tampered, _DIGEST)


def test_rejects_non_cosign_scheme() -> None:
    unsigned = Signature(scheme=SignatureScheme.UNSIGNED)
    with pytest.raises(SignatureError, match="not a cosign signature"):
        verify_signature(unsigned, _DIGEST)


def test_rejects_incomplete_envelope() -> None:
    # A cosign-scheme block missing value/payload/certificate is incomplete → fail closed.
    empty = Signature(scheme=SignatureScheme.SIGSTORE_COSIGN)
    with pytest.raises(SignatureError, match="incomplete signature envelope"):
        verify_signature(empty, _DIGEST)


def test_malformed_keys_fail_closed() -> None:
    with pytest.raises(SignatureError, match="malformed private key"):
        sign_digest(_DIGEST, b"-----BEGIN PRIVATE KEY-----\nnope\n-----END PRIVATE KEY-----\n")
    private_pem, _ = generate_keypair()
    sig = sign_digest(_DIGEST, private_pem)
    with pytest.raises(SignatureError, match="malformed public key"):
        verify_signature(sig, _DIGEST, trusted_public_key_pem=b"not a key")


def test_rejects_non_ecdsa_keys() -> None:
    # A well-formed PEM key of the wrong algorithm (RSA) is refused — ECDSA P-256 only.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsa_priv = rsa_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    rsa_pub = rsa_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with pytest.raises(SignatureError, match="not an ECDSA private key"):
        sign_digest(_DIGEST, rsa_priv)
    ec_private, _ = generate_keypair()
    sig = sign_digest(_DIGEST, ec_private)
    with pytest.raises(SignatureError, match="not an ECDSA public key"):
        verify_signature(sig, _DIGEST, trusted_public_key_pem=rsa_pub)


def test_signer_id_is_stable_and_key_specific() -> None:
    private_pem, public_pem = generate_keypair()
    # Stable across calls, and PEM whitespace-insensitive (same key -> same id).
    fid = signer_id(public_pem)
    assert fid.startswith("sha256:")
    assert fid == signer_id(public_pem + b"\n")
    _, other_public = generate_keypair()
    assert signer_id(other_public) != fid
    # The signer id derives from the signature's own carried public key too.
    sig = sign_digest(_DIGEST, private_pem)
    assert sig.certificate is not None
    assert signer_id(sig.certificate.encode()) == fid
