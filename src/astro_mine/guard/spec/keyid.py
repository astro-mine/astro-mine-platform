"""Signer-identity fingerprint for a loaded artifact's provenance (guard.md §9.5).

The one bit of key handling that stays in Guard after adopting the shared
:mod:`astro_mine.seal` signer (RFC-0005): a stable ``sha256:<hex>`` fingerprint of a signer's
public key, recorded as ``SignedArtifact.signer_id`` so provenance can name *which* key signed
an artifact without carrying the whole PEM. It is Guard-specific provenance identity, not part
of the shared signer surface, so it is **not** re-homed into Seal.

Fails closed on malformed / non-ECDSA key material, mirroring the signer (Seal's
:class:`~astro_mine.seal.SignatureError`).
"""

from __future__ import annotations

import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from astro_mine.seal import SignatureError

__all__ = ["signer_id"]


def _public_der(public_pem: bytes) -> bytes:
    """The canonical DER (SPKI) encoding of a public key — PEM whitespace-insensitive."""
    try:
        key = serialization.load_pem_public_key(public_pem)
    except (ValueError, TypeError) as exc:
        raise SignatureError(f"malformed public key: {exc}") from exc
    if not isinstance(key, ec.EllipticCurvePublicKey):
        raise SignatureError("verifying key is not an ECDSA public key")
    return key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def signer_id(public_pem: bytes) -> str:
    """A stable ``sha256:<hex>`` fingerprint of a signer's public key (its provenance identity).

    The content address of the key's canonical DER (SPKI), so a loaded artifact's provenance can
    record *which* key signed it without carrying the whole PEM. Raises
    :class:`~astro_mine.seal.SignatureError` on a malformed key (fail closed)."""
    return "sha256:" + hashlib.sha256(_public_der(public_pem)).hexdigest()
