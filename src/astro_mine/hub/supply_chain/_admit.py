# SPDX-License-Identifier: Apache-2.0
"""The admission gate — the *first* of the two verifications (RM-P1-HUB-03; hub.md §2.3).

``hub.md`` §2 principle 3 states three clauses: evidence is checked at **publish (admission)** and
at **pull**, and *"unsigned content is never promoted to a verified namespace"*. The pull side
shipped; this module is the admission side, and it exists as **one** function because the three
paths that admit content — the library :class:`~astro_mine.hub.HubClient`, the service's
``POST /publish``, and curation's ``promote`` — were three separate routes, so a check added to one
was absent from the others. That drift is the failure mode :func:`admit` closes.

What admission proves, in order, before anything is indexed:

1. **The artifact exists and its bytes are its address.** ``registry.verify`` re-hashes the manifest
   and every blob. A digest the caller merely *asserts* proves nothing — the service endpoint took
   one on trust.
2. **The manifest is the one that was stored.** The indexed manifest is what discovery answers
   queries from; if it may disagree with the artifact's config blob, the index describes something
   other than the bytes a consumer will pull.
3. **The evidence verifies, fail closed.** Signature → SLSA provenance → SBOM, through
   :mod:`astro_mine.seal` (RFC-0005) — Hub adds no crypto, it calls the one implementation.

Only then does the entry reach the catalog. A failure raises and leaves **nothing** indexed: a
half-admitted artifact — bytes present, evidence absent, entry queryable — is precisely the state
an attacker wants.

**Unsigned content is refused.** ``hub.md`` §9 tiers artifacts as *open* ("self-published, signed
but unreviewed"), *curated*, and *verified* — there is no tier for unsigned content, so admitting
it would index something the trust model cannot describe. Signing is available offline with no
account (keyed ECDSA is Seal's default; ``astro-mine hub keygen`` mints a key), so this costs the
local tier nothing (CX-LOCAL).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from astro_mine.core.registry.model import PluginManifest
from astro_mine.hub.index import Catalog, CatalogEntry, ingest
from astro_mine.hub.registry import RegistryClient, artifact_kind_of
from astro_mine.hub.supply_chain._supply_chain import verify
from astro_mine.seal import ARTIFACT_TYPE_SIGNATURE, DEFAULT_REQUIRED, SupplyChainError

__all__ = [
    "UnsignedArtifactError",
    "admit",
    "has_signature",
    "require_signed",
    "stored_artifact_kind",
    "verify_admissible",
]


class UnsignedArtifactError(SupplyChainError):
    """An artifact carrying no signature was offered for admission.

    A subclass of :class:`SupplyChainError`, so a caller that fails closed on supply-chain errors
    already fails closed on this one.
    """


def _stored_manifest(registry: RegistryClient, digest: str) -> dict[str, object]:
    """The Core manifest actually stored as the artifact's config blob."""
    raw = registry.read_config(digest)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise SupplyChainError(f"artifact {digest} config is not a JSON object")
    return parsed


def has_signature(registry: RegistryClient, digest: str) -> bool:
    """Whether *any* signature attestation is attached to ``digest`` — a cheap presence probe.

    Presence is not validity: :func:`verify_admissible` is what proves the signature verifies. This
    exists only so a caller can raise :class:`UnsignedArtifactError` with an actionable message
    instead of a generic missing-attestation error.
    """
    return bool(registry.referrers(digest, artifact_type=ARTIFACT_TYPE_SIGNATURE))


def require_signed(registry: RegistryClient, digest: str) -> None:
    """Raise :class:`UnsignedArtifactError` when ``digest`` carries no signature at all."""
    if not has_signature(registry, digest):
        raise UnsignedArtifactError(
            f"artifact {digest} is unsigned and cannot be admitted; sign it at publish "
            f"(`astro-mine hub keygen` mints a key; keyed ECDSA works offline with no account). "
            f"hub.md §9 defines no namespace tier for unsigned content"
        )


def verify_admissible(
    registry: RegistryClient,
    manifest: PluginManifest,
    *,
    digest: str,
    trusted_public_key_pem: bytes | None = None,
    require: Sequence[str] = DEFAULT_REQUIRED,
) -> None:
    """Run the admission checks for ``digest``; raise on the first failure, index nothing.

    Separated from :func:`admit` so curation can re-run the same evidence check on an
    already-indexed artifact without re-ingesting it.
    """
    # 1. The bytes are their address. A digest the caller merely asserted may not exist at all —
    #    normalized into a SupplyChainError so every admission path fails closed with one error
    #    contract, rather than leaking the registry's KeyError as an opaque 500.
    try:
        registry.verify(digest)
    except KeyError as exc:
        raise SupplyChainError(
            f"artifact {digest} is not present in this registry; a digest cannot be indexed on "
            f"the caller's word"
        ) from exc

    # 2. The manifest offered for indexing is the manifest that was stored. Compared through the
    #    Core model's own JSON projection so field ordering and defaults cannot manufacture a
    #    difference that is not one.
    stored = _stored_manifest(registry, digest)
    submitted = manifest.model_dump(mode="json")
    if stored != submitted:
        raise SupplyChainError(
            f"artifact {digest} manifest does not match its stored config: the index would "
            f"describe something other than the bytes a consumer pulls"
        )

    # 3. Unsigned content is refused outright, ahead of the generic evidence check, so the caller
    #    gets "sign it, here is how" rather than "missing required attestation". Unconditional:
    #    narrowing `require` cannot buy an exemption, because hub.md §9 has no unsigned tier.
    require_signed(registry, digest)

    # 4. The evidence verifies, fail closed (signature -> SLSA -> SBOM), through Seal.
    verify(registry, digest, trusted_public_key_pem=trusted_public_key_pem, require=require)


def admit(
    registry: RegistryClient,
    catalog: Catalog,
    manifest: PluginManifest,
    *,
    digest: str,
    publisher: str,
    namespace: str = "open",
    trusted_public_key_pem: bytes | None = None,
    require: Sequence[str] = DEFAULT_REQUIRED,
    provider: object | None = None,
) -> CatalogEntry:
    """Verify ``digest`` fail-closed, then index it — the single admission path.

    Raises :class:`SupplyChainError` (or a subclass) on any failure, having indexed nothing.
    """
    verify_admissible(
        registry,
        manifest,
        digest=digest,
        trusted_public_key_pem=trusted_public_key_pem,
        require=require,
    )
    return ingest(
        catalog,
        manifest,
        digest=digest,
        publisher=publisher,
        namespace=namespace,
        # Read the container kind back off the artifact that was actually stored, never from a
        # caller: on this path every field a request supplies is a claim, and the whole point of
        # admission is that claims are re-derived from the bytes (hub.md §2 principle 3).
        artifact_kind=stored_artifact_kind(registry, digest),
        provider=provider,  # type: ignore[arg-type]
    )


def stored_artifact_kind(registry: RegistryClient, digest: str) -> str | None:
    """Hub's container kind for ``digest``, recovered from its stored OCI ``artifactType``.

    ``None`` when the artifact carries none — it was published by another OCI tool, or predates the
    facet. Absence is not an error: the artifact is still storable, pullable, and indexed by its
    Core manifest kind, which is the contract consumers negotiate against.
    """
    try:
        oci_manifest = registry.read_manifest(digest)
    except (KeyError, ValueError):  # unreadable manifest is not an indexing failure
        return None
    return artifact_kind_of(oci_manifest.get("artifactType"))
