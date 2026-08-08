"""Verify-twice supply chain — attest at publish, re-verify at pull (RM-P1-HUB-03).

The supply-chain trust boundary (hub.md §9): an artifact is **signed** (cosign-keyed) and carries
**SLSA provenance** + an **SBOM**, attached as OCI referrers by digest. Verification runs **at
admission (publish)** *and* **at pull** (the client re-verifies before Core loads the plugin) — a
compromised registry cannot make a consumer accept tampered bytes. Every check **fails closed**.

The *policy* is not Hub's. Which evidence is required, what makes an attestation well-shaped, and
the fail-closed check itself all live in :mod:`astro_mine.seal` (RM-P1-SEAL-03; RFC-0005
§Sequencing: "Hub imports them from there, behavior-preserving") — so Hub, Bench, Fleet, and Guard
share **one** implementation of the supply-chain policy instead of each carrying a copy that can
drift. Seal's orchestrators drive the :class:`~astro_mine.seal.AttestationStore` port (``str`` and
``bytes`` only), because Seal depends on Core alone and may not know about OCI referrers
(seal.md §1: "storing and serving artifacts is Hub").

What is left here is exactly the half Hub owns — **the registry plane**:

- :class:`RegistryAttestationStore` — binds Hub's OCI :class:`RegistryClient` to Seal's port.
- :func:`attest` / :func:`verify` — thin façades that wrap the registry in that store and delegate,
  keeping Hub's ``(registry, subject, ...)`` call shape for its existing callers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from astro_mine.hub.registry import Blob, RegistryClient
from astro_mine.seal import (
    DEFAULT_REQUIRED,
    AttestationSet,
    AttestationStore,
    SupplyChainError,
    TrustRoot,
)
from astro_mine.seal import attest as _seal_attest
from astro_mine.seal import verify as _seal_verify

__all__ = [
    "BUILDER_ID",
    "DEFAULT_REQUIRED",
    "AttestationSet",
    "RegistryAttestationStore",
    "SupplyChainError",
    "attest",
    "verify",
]

#: Hub's identity in the SLSA provenance it produces. Seal requires ``builder_id`` — a provenance
#: claim it will not default on a producer's behalf — so the producer names itself here.
BUILDER_ID = "astro-mine-hub"


class RegistryAttestationStore:
    """Hub's OCI :class:`RegistryClient` bound to Seal's :class:`AttestationStore` port.

    The one adapter between the two planes: Seal owns the supply-chain *policy* and speaks only
    digests and bytes; Hub owns the *registry* and knows that an attestation is a referrer manifest
    whose single layer carries the payload. Reads are integrity-checked and **raise** on tampered
    bytes, as the port's contract requires — Seal turns any such failure into a fail-closed
    :class:`SupplyChainError`, never into "no evidence, therefore fine".
    """

    def __init__(self, registry: RegistryClient) -> None:
        self._registry = registry

    def attach_attestation(
        self, *, subject: str, artifact_type: str, media_type: str, payload: bytes
    ) -> str:
        """Attach ``payload`` to ``subject`` as a referrer; return that referrer's digest."""
        return self._registry.attach(
            subject=subject,
            artifact_type=artifact_type,
            blob=Blob(media_type, payload),
        ).digest

    def attestation_digests(self, subject: str, *, artifact_type: str) -> Sequence[str]:
        """``subject``'s referrer digests of ``artifact_type`` — empty when it carries none."""
        refs = self._registry.referrers(subject, artifact_type=artifact_type)
        return [ref.digest for ref in refs]

    def read_attestation(self, digest: str) -> bytes:
        """The payload the referrer manifest at ``digest`` carries (its single layer).

        Re-hashes the manifest and its blobs first, so a tampered attestation raises rather than
        returning bytes (tamper-on-read: hub.md §2.3).
        """
        self._registry.verify(digest)
        manifest = self._registry.read_manifest(digest)
        return self._registry.pull_blob(manifest["layers"][0]["digest"])

    def verify_integrity(self, subject: str) -> None:
        """Re-hash ``subject``'s manifest, config, and layers against their addresses."""
        self._registry.verify(subject)


def attest(
    registry: RegistryClient,
    subject: str,
    *,
    private_key_pem: bytes,
    name: str,
    version: str,
    builder_id: str = BUILDER_ID,
    inputs: Sequence[str] = (),
    components: Sequence[Mapping[str, str]] = (),
) -> AttestationSet:
    """Sign ``subject`` and attach the cosign signature, SLSA provenance, and SBOM as referrers.

    ``subject`` is the artifact's manifest digest. Returns Seal's :class:`AttestationSet` — the
    ``sha256:<hex>`` digests of the three attachments, which :func:`verify` reads back. This is the
    **publish-side** half — call :func:`verify` immediately after for verify-at-admission.
    """
    store: AttestationStore = RegistryAttestationStore(registry)
    return _seal_attest(
        store,
        subject,
        private_key_pem=private_key_pem,
        name=name,
        version=version,
        builder_id=builder_id,
        inputs=inputs,
        components=components,
    )


def verify(
    registry: RegistryClient,
    subject: str,
    *,
    trusted_public_key_pem: bytes | None = None,
    trust_root: TrustRoot | None = None,
    kind: str | None = None,
    require: Sequence[str] = DEFAULT_REQUIRED,
) -> None:
    """Re-verify ``subject``'s integrity and required attestations — raise on any failure.

    The same check at admission and at pull (hub.md §2.3, §9), enforced by Seal: the artifact's own
    bytes hash to their addresses; *every* attached cosign signature is intact and verifies over
    ``subject`` (accepted by ``trust_root``, or ``trusted_public_key_pem`` for the one-key
    case, when given); SLSA provenance is present,
    intact, and well-shaped; an SBOM is present, intact, and CycloneDX.

    A ``require`` token Seal does not know is **refused**, not ignored — a typo can never quietly
    disable a check. Any failure, including one raised by the registry, is a
    :class:`SupplyChainError` — **fail closed**.
    """
    store: AttestationStore = RegistryAttestationStore(registry)
    _seal_verify(
        store,
        subject,
        trusted_public_key_pem=trusted_public_key_pem,
        trust_root=trust_root,
        kind=kind,
        require=require,
    )
