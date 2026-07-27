"""Signing / verification tests (RM-P1-SEAL-02): round-trip, fail-closed negatives, and a frozen
cross-package conformance vector.

The signer is a byte-compatible port of the one that lived in ``astro-mine-hub`` /
``astro-mine-fleet`` / ``astro-mine-guard``; the frozen vector below pins interoperability so any
future encoding drift turns CI red. ECDSA signing is non-deterministic (random nonce), so the guard
pins a *verifiable* ``(public key, digest, signature)`` triple — not produced bytes — which is
exactly what cross-verification between producers requires.
"""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from astro_mine.core.hashing import content_hash
from astro_mine.core.registry import (
    PluginKind,
    PluginManifest,
    Provenance,
    Signature,
    SignatureKind,
    SignatureScheme,
)
from astro_mine.seal import (
    SignatureError,
    generate_keypair,
    make_verifier,
    sign_digest,
    verify_signature,
)

DIGEST = content_hash(b"artifact-bytes")


def _rsa_private_pem() -> bytes:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


# --- round-trip -----------------------------------------------------------------------------


def test_sign_and_verify_roundtrip() -> None:
    priv, pub = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    assert sig.scheme == SignatureScheme.SIGSTORE_COSIGN
    assert sig.kind == SignatureKind.COSIGN_SIGNATURE
    verify_signature(sig, DIGEST)  # untrusted (self-attested) — no raise
    verify_signature(sig, DIGEST, trusted_public_key_pem=pub)  # pinned — no raise


# --- fail-closed negatives ------------------------------------------------------------------


def test_verify_signature_wrong_digest() -> None:
    priv, _ = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    with pytest.raises(SignatureError):
        verify_signature(sig, content_hash(b"other"))


def test_verify_signature_untrusted_key() -> None:
    priv, _ = generate_keypair()
    _, other_pub = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    with pytest.raises(SignatureError):
        verify_signature(sig, DIGEST, trusted_public_key_pem=other_pub)


def test_verify_signature_bad_signature() -> None:
    priv, _ = generate_keypair()
    # a signature over a different digest, re-labelled as covering DIGEST → ECDSA verify fails
    forged = sign_digest(content_hash(b"different"), priv).model_copy(update={"payload": DIGEST})
    with pytest.raises(SignatureError):
        verify_signature(forged, DIGEST)


def test_verify_signature_incomplete_and_wrong_scheme() -> None:
    priv, _ = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    with pytest.raises(SignatureError):
        verify_signature(sig.model_copy(update={"value": None}), DIGEST)
    with pytest.raises(SignatureError):
        verify_signature(sig.model_copy(update={"scheme": SignatureScheme.UNSIGNED}), DIGEST)


def test_sign_with_non_ec_key_rejected() -> None:
    with pytest.raises(SignatureError):
        sign_digest(DIGEST, _rsa_private_pem())


def test_sign_with_malformed_private_key_rejected() -> None:
    # Fail closed on unparseable key material — a raw ValueError must not leak (defense in depth;
    # the signer's whole contract is "refuse, never guess"). Mirrors the public-key path.
    with pytest.raises(SignatureError, match="malformed private key"):
        sign_digest(
            DIGEST, b"-----BEGIN PRIVATE KEY-----\nnot a real key\n-----END PRIVATE KEY-----\n"
        )


def test_verify_with_non_ec_or_malformed_cert_rejected() -> None:
    priv, _ = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    with pytest.raises(SignatureError):
        verify_signature(sig.model_copy(update={"certificate": "not a pem"}), DIGEST)
    rsa_pub = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    with pytest.raises(SignatureError):
        verify_signature(sig.model_copy(update={"certificate": rsa_pub}), DIGEST)


# --- make_verifier (the pre-Core-load hook) -------------------------------------------------


def _manifest(digest: str | None, *, sig: Signature | None = None) -> PluginManifest:
    provenance = (
        Provenance(input_hashes=[], source_content_hashes={}, digest=digest) if digest else None
    )
    return PluginManifest(
        name="p", version="1.0.0", kind=PluginKind.POLICY, provenance=provenance, signature=sig
    )


def test_make_verifier_ok_and_failures() -> None:
    priv, pub = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    make_verifier(trusted_public_key_pem=pub)(_manifest(DIGEST, sig=sig))  # ok

    with pytest.raises(SignatureError):  # no provenance digest to bind to
        make_verifier()(_manifest(None))
    with pytest.raises(SignatureError):  # provenance but no cosign signature
        make_verifier()(_manifest(DIGEST))
    _, other = generate_keypair()
    with pytest.raises(SignatureError):  # signed by an untrusted key
        make_verifier(trusted_public_key_pem=other)(_manifest(DIGEST, sig=sig))


# --- frozen cross-package conformance vector ------------------------------------------------
#
# A signature produced once by this cosign-ECDSA-P256 signer over CONFORMANCE_DIGEST. It MUST keep
# verifying byte-for-byte: any drift in digest/key/signature encoding (base64, DER, PEM, the payload
# binding) breaks this and turns CI red — the guarantee that Seal-signed artifacts interoperate with
# every producer/verifier that shares this encoding (Fleet/Guard/Hub).
CONFORMANCE_DIGEST = "sha256:7220c21b2506cf1c291485943598b6a043a4885ccc81f50a413d1128ec700869"
CONFORMANCE_PUBLIC_PEM = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEcwmJ4AtQ4lhBMegYnk2/ZSOg0kyW\n"
    "SVo5Cf0Kq1pHEbv2o6mhecAFON4mfJjh9bWHrjdfDhPHjuwt/BeQYcHYrg==\n"
    "-----END PUBLIC KEY-----\n"
)
CONFORMANCE_SIGNATURE_VALUE = (
    "MEQCIFz6UmnIoFY9Nx49LV0NTI9IAVNnoggIwQolTlXJYm/GAiBijTZXn4k7fqFLf"
    "+D/FFErkVLb4ZzqIb5zK3Thz3PIzQ=="
)


def _conformance_signature() -> Signature:
    return Signature(
        scheme=SignatureScheme.SIGSTORE_COSIGN,
        kind=SignatureKind.COSIGN_SIGNATURE,
        value=CONFORMANCE_SIGNATURE_VALUE,
        payload=CONFORMANCE_DIGEST,
        certificate=CONFORMANCE_PUBLIC_PEM,
    )


def test_frozen_conformance_vector_verifies() -> None:
    sig = _conformance_signature()
    verify_signature(sig, CONFORMANCE_DIGEST)  # self-attested
    verify_signature(  # pinned to the frozen key
        sig, CONFORMANCE_DIGEST, trusted_public_key_pem=CONFORMANCE_PUBLIC_PEM.encode()
    )


def test_frozen_conformance_vector_is_fail_closed() -> None:
    sig = _conformance_signature()
    # bound to its digest: it must NOT verify against a different artifact digest
    with pytest.raises(SignatureError):
        verify_signature(sig, content_hash(b"a different artifact"))
    # pinned trust: it must NOT verify against a key other than the one that signed it
    _, other_pub = generate_keypair()
    with pytest.raises(SignatureError):
        verify_signature(sig, CONFORMANCE_DIGEST, trusted_public_key_pem=other_pub)
