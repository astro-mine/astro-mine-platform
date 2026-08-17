# SPDX-License-Identifier: Apache-2.0
"""Verify-twice supply chain — cosign + SLSA + SBOM at publish and pull (RM-P1-HUB-03).

Hub's supply-chain trust boundary (hub.md §9; ``LUNAR-SR-002``): every shared artifact is
**signed** (cosign-keyed ECDSA, the offline default), carries **SLSA provenance** and an **SBOM**,
and is **re-verified at pull** before Core loads it. Attestations attach to an artifact by digest
via the OCI Referrers model (the :mod:`astro_mine.hub.registry` hook). All checks **fail closed**.

- :func:`attest` — sign an artifact and attach its signature / SLSA provenance / SBOM (publish).
- :func:`verify` — re-verify integrity + required attestations, raising on any failure (pull).
- :func:`make_verifier` — a Core ``Verifier`` the client runs *before* the Core registry loads a
  pulled plugin (defense in depth, hub.md §2.3).

This module is a **facade**. The signer, the attestation documents, and the verify-twice
orchestration all live in :mod:`astro_mine.seal` (RFC-0005; RM-P1-SEAL-03) — one implementation of
the supply-chain policy for Hub, Bench, Fleet, and Guard. Hub keeps only the registry plane:
:class:`~astro_mine.hub.supply_chain._supply_chain.RegistryAttestationStore` binds its OCI registry
to Seal's ``AttestationStore`` port. ``SupplyChainError`` is re-exported **from Seal**, so a caller
that catches ``astro_mine.hub.supply_chain.SupplyChainError`` still catches everything the
orchestrator raises.

Keyless Sigstore (Fulcio/Rekor/OIDC) is deferred; the ``sigstore_cosign`` scheme is kept so the
upgrade is additive.

Backlog: RM-P1-HUB-03 — astro-mine-hub#3
"""

from __future__ import annotations

# The admission gate — the publish-side half of verify-twice, shared by the library, the service,
# and curation so a check cannot exist on one path and be missing from another (hub.md §2.3).
from astro_mine.hub.supply_chain._admit import (
    UnsignedArtifactError,
    admit,
    has_signature,
    require_signed,
    stored_artifact_kind,
    verify_admissible,
)
from astro_mine.hub.supply_chain._supply_chain import (
    BUILDER_ID,
    RegistryAttestationStore,
    attest,
    verify,
)

# The shared artifact-integrity surface lives in astro-mine-seal (RFC-0005): the signer, the SLSA /
# SBOM builders, the required-evidence policy, and the fail-closed errors. Hub re-exports them so
# its callers keep one import site — and, critically, so `except SupplyChainError` on Hub's name
# catches the error Seal's orchestrator actually raises.
from astro_mine.seal import (
    ARTIFACT_TYPE_SBOM,
    ARTIFACT_TYPE_SIGNATURE,
    ARTIFACT_TYPE_SLSA,
    DEFAULT_REQUIRED,
    AttestationError,
    AttestationSet,
    SignatureError,
    SupplyChainError,
    build_cyclonedx_sbom,
    build_slsa_provenance,
    generate_keypair,
    make_verifier,
    sign_digest,
    verify_signature,
)

__all__ = [
    "ARTIFACT_TYPE_SBOM",
    "ARTIFACT_TYPE_SIGNATURE",
    "ARTIFACT_TYPE_SLSA",
    "BUILDER_ID",
    "DEFAULT_REQUIRED",
    "AttestationError",
    "AttestationSet",
    "RegistryAttestationStore",
    "SignatureError",
    "SupplyChainError",
    "UnsignedArtifactError",
    "admit",
    "attest",
    "build_cyclonedx_sbom",
    "build_slsa_provenance",
    "generate_keypair",
    "has_signature",
    "make_verifier",
    "require_signed",
    "sign_digest",
    "stored_artifact_kind",
    "verify",
    "verify_admissible",
    "verify_signature",
]
