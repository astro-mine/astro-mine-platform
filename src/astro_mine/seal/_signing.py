"""cosign-keyed ECDSA signing + verification (RM-P1-SEAL-02).

The one shared signer for the platform: a keyed **ECDSA P-256** detached signature (cosign's
default curve) over an artifact's content digest — no Fulcio/Rekor/OIDC, so the tier-1 local path
stays **offline and accountless** (conventions §7 tier-1; `LUNAR-TR-004`). The envelope is Core's
cosign-shaped :class:`~astro_mine.core.registry.Signature`
(``scheme=sigstore_cosign``, ``kind=cosign_signature``); the keyless upgrade (Fulcio cert + Rekor
inclusion proof) keeps this scheme and adds fields additively, so it is a drop-in later.

Consolidated by [RFC-0005](https://github.com/astro-mine/docs/blob/main/rfc/0005-seal-supply-chain-companion.md)
from ``astro-mine-hub``'s ``supply_chain/_signing.py`` — **byte-compatible** with it (and with the
now-deleted Fleet / Guard copies), so every producer and verifier signs and checks the same seal.
Core stays crypto-free; the ``cryptography`` dependency lives only here.

Everything **fails closed**: a malformed key, an incomplete envelope, a payload that does not
match the digest, an untrusted signer, or a bad signature all raise :class:`SignatureError`.
"""

from __future__ import annotations

import base64

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from astro_mine.core.registry import (
    PluginManifest,
    Signature,
    SignatureKind,
    SignatureScheme,
    Verifier,
)

__all__ = [
    "SignatureError",
    "generate_keypair",
    "make_verifier",
    "sign_digest",
    "verify_signature",
]

_CURVE = ec.SECP256R1  # cosign's default key type (ECDSA P-256)


class SignatureError(Exception):
    """Signing material is malformed, or a signature does not verify (always fail closed)."""


def generate_keypair() -> tuple[bytes, bytes]:
    """A fresh ``(private_pem, public_pem)`` ECDSA P-256 keypair (PKCS8 / SPKI PEM)."""
    key = ec.generate_private_key(_CURVE())
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _load_private(private_pem: bytes) -> ec.EllipticCurvePrivateKey:
    try:
        key = serialization.load_pem_private_key(private_pem, password=None)
    except (ValueError, TypeError) as exc:
        raise SignatureError(f"malformed private key: {exc}") from exc
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise SignatureError("signing key is not an ECDSA private key")
    return key


def _load_public(public_pem: bytes) -> ec.EllipticCurvePublicKey:
    try:
        key = serialization.load_pem_public_key(public_pem)
    except (ValueError, TypeError) as exc:
        raise SignatureError(f"malformed public key: {exc}") from exc
    if not isinstance(key, ec.EllipticCurvePublicKey):
        raise SignatureError("verifying key is not an ECDSA public key")
    return key


def _keys_equal(a_pem: bytes, b_pem: bytes) -> bool:
    """Compare two public keys by canonical DER (PEM whitespace-insensitive)."""
    a = _load_public(a_pem).public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    b = _load_public(b_pem).public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return a == b


def sign_digest(digest: str, private_pem: bytes) -> Signature:
    """Sign a ``sha256:<hex>`` artifact ``digest`` and return a Core cosign :class:`Signature`.

    The signature is over the digest string; the signer's public key travels in ``certificate``
    (Phase-1 keyed trust; the keyless upgrade replaces it with a Fulcio cert). ``payload`` binds the
    signature to the artifact digest, so a verifier can confirm *what* was signed.
    """
    key = _load_private(private_pem)
    der = key.sign(digest.encode(), ec.ECDSA(hashes.SHA256()))
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return Signature(
        scheme=SignatureScheme.SIGSTORE_COSIGN,
        kind=SignatureKind.COSIGN_SIGNATURE,
        value=base64.b64encode(der).decode(),
        payload=digest,
        certificate=public_pem.decode(),
    )


def verify_signature(
    signature: Signature, digest: str, *, trusted_public_key_pem: bytes | None = None
) -> None:
    """Verify ``signature`` covers ``digest`` — raise :class:`SignatureError` on any failure.

    Checks, in order (fail closed): the scheme is cosign; the envelope is complete; the signed
    ``payload`` equals the artifact ``digest`` (the signature is *for this artifact*); if
    ``trusted_public_key_pem`` is given, the signer key equals it (pinned trust); and the ECDSA
    signature verifies. Returns ``None`` only when every check passes.
    """
    if signature.scheme != SignatureScheme.SIGSTORE_COSIGN:
        raise SignatureError(f"not a cosign signature (scheme={signature.scheme})")
    if signature.value is None or signature.payload is None or signature.certificate is None:
        raise SignatureError("incomplete signature envelope")
    if signature.payload != digest:
        raise SignatureError("signature payload does not match the artifact digest")

    signer_pem = signature.certificate.encode()
    if trusted_public_key_pem is not None and not _keys_equal(trusted_public_key_pem, signer_pem):
        raise SignatureError("signing key is not the trusted key")

    public_key = _load_public(signer_pem)
    try:
        public_key.verify(
            base64.b64decode(signature.value),
            signature.payload.encode(),
            ec.ECDSA(hashes.SHA256()),
        )
    except InvalidSignature:
        raise SignatureError("signature does not verify") from None


def make_verifier(*, trusted_public_key_pem: bytes | None = None) -> Verifier:
    """A Core ``Verifier`` checking a manifest's own cosign signature before Core loads it.

    The ``Callable[[PluginManifest], None]`` shape Core's ``PluginRegistry`` invokes at load — the
    client re-runs it before Core loads a pulled plugin (defense in depth, hub.md §2.3). It finds a
    cosign signature among the manifest's :attr:`~PluginManifest.all_signatures` and verifies it
    against the manifest's ``provenance.digest`` (the artifact's content identity). Raises
    :class:`SignatureError` on any failure — aborting the load.
    """

    def _verify(manifest: PluginManifest) -> None:
        digest = manifest.provenance.digest if manifest.provenance else None
        if digest is None:
            raise SignatureError("manifest has no provenance.digest to bind a signature to")
        cosign = [s for s in manifest.all_signatures if s.scheme == SignatureScheme.SIGSTORE_COSIGN]
        if not cosign:
            raise SignatureError("manifest carries no cosign signature")
        for signature in cosign:
            verify_signature(signature, digest, trusted_public_key_pem=trusted_public_key_pem)

    return _verify
