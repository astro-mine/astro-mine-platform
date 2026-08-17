# SPDX-License-Identifier: Apache-2.0
"""Asset-manifest signing glue over the shared signer (RM-P0-FLEET-06, CX-SEC; RFC-0005 dedup).

The ECDSA P-256 cosign signer itself lives once in :mod:`astro_mine.seal` (RFC-0005 — one signer
for every producer and verifier). This module is Fleet's thin manifest-level glue over it:

- :func:`sign_asset` signs a SADF asset's ``sha256:<hex>`` content digest (a Fleet-domain name for
  :func:`astro_mine.seal.sign_digest` — byte-identical envelope);
- :func:`make_verifier` is the ``Callable[[PluginManifest], None]`` Core's ``PluginRegistry``
  invokes before load, mapping Fleet's single-``signature`` asset manifest onto Seal's fail-closed
  :func:`~astro_mine.seal.verify_signature`.

Everything **fails closed**: an unsigned / wrong-scheme / incomplete / mismatched / tampered /
wrong-key manifest raises :class:`~astro_mine.seal.SignatureError`, aborting the load.
"""

from __future__ import annotations

from astro_mine.core.registry import PluginManifest, Signature, SignatureScheme, Verifier
from astro_mine.seal import SignatureError, sign_digest, verify_signature

__all__ = ["make_verifier", "sign_asset"]


def sign_asset(asset_digest: str, private_pem: bytes) -> Signature:
    """Sign a SADF asset's ``sha256:<hex>`` content digest, returning a Core cosign ``Signature``.

    A Fleet-domain alias for the shared :func:`astro_mine.seal.sign_digest`; the signature binds
    the asset's content identity (``provenance.digest``), so :func:`make_verifier` checks from the
    manifest alone.
    """
    return sign_digest(asset_digest, private_pem)


def make_verifier(*, trusted_public_key_pem: bytes | None = None) -> Verifier:
    """Build a Core ``Verifier`` that checks an asset manifest's own cosign signature.

    If *trusted_public_key_pem* is given, the signer's key must equal it (pinned trust); otherwise
    the key embedded in the signature is used (self-attested dev mode). Fleet's single-``signature``
    asset manifest is mapped onto Seal's :func:`~astro_mine.seal.verify_signature`, which performs
    the key comparison and the ECDSA verification. Raises :class:`~astro_mine.seal.SignatureError`
    on any failure, aborting the load.
    """

    def _verify(manifest: PluginManifest) -> None:
        signature = manifest.signature
        if signature is None or signature.scheme != SignatureScheme.SIGSTORE_COSIGN:
            raise SignatureError(
                "manifest is not signed with an astro-mine ECDSA signature (scheme sigstore_cosign)"
            )
        if signature.value is None or signature.payload is None or signature.certificate is None:
            raise SignatureError("incomplete signature block")
        digest = manifest.provenance.digest if manifest.provenance else None
        if digest is None:
            raise SignatureError("signature payload does not match the artifact digest")
        verify_signature(signature, digest, trusted_public_key_pem=trusted_public_key_pem)

    return _verify
