"""Astro-Mine-Seal — shared artifact-integrity companion.

The single home for the platform's signing, verification, SLSA provenance, and SBOM
facilities — a thin Core companion (the RFC-0002 ``astro-mine-spice`` shape) built on
Core's already-frozen :class:`astro_mine.core.registry.Signature` / ``Verifier`` surface
and the ``astro_mine.core.hashing`` content-hash primitive. Every producer signs a
**seal** on its artifacts and the intactness of that seal is what verification tests.
Core stays crypto-free; this package is the one home for the ``cryptography`` dependency.

The full artifact-integrity surface: the signer (``RM-P1-SEAL-02``), the SLSA / SBOM attestation
documents, and the **registry-agnostic** ``attest`` / ``verify`` verify-twice orchestrators
(``RM-P1-SEAL-03``). ``attest`` and ``verify`` drive the :class:`AttestationStore` port — ``str``
digests and ``bytes`` payloads, no registry type — so Hub, Bench, Fleet, and Guard share one
implementation of the supply-chain policy while Seal depends on Core alone. See
``docs/rfc/0005-seal-supply-chain-companion.md``, ``docs/architecture/seal.md`` §3, and
``docs/architecture/guard.md`` §9.5.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0"

# The artifact-integrity surface (RM-P1-SEAL-02/-03). ``__version__`` is defined above before these
# imports so any module below can read it without a cycle. ``attest`` / ``verify`` orchestrate over
# the ``AttestationStore`` port rather than a concrete registry, so the registry/publish plane stays
# in Hub while the supply-chain *policy* lives here, once (seal.md §1; RFC-0005 §Sequencing).
from astro_mine.seal._attest import (
    ARTIFACT_TYPE_SBOM,
    ARTIFACT_TYPE_SIGNATURE,
    ARTIFACT_TYPE_SLSA,
    CYCLONEDX_SPEC_VERSION,
    INTOTO_STATEMENT_TYPE,
    MEDIA_SBOM,
    MEDIA_SIGNATURE,
    MEDIA_SLSA,
    SLSA_PREDICATE_TYPE,
    AttestationSet,
    AttestationStore,
    attest,
    build_cyclonedx_sbom,
    build_slsa_provenance,
)
from astro_mine.seal._signing import (
    SignatureError,
    generate_keypair,
    make_verifier,
    sign_digest,
    verify_signature,
)
from astro_mine.seal._supply_chain import (
    DEFAULT_REQUIRED,
    AttestationError,
    SupplyChainError,
    verify,
    verify_sbom_document,
    verify_slsa_document,
)
from astro_mine.seal._trust import (
    TRUST_ROOT_ENV,
    TrustedKey,
    TrustRoot,
    TrustRootError,
    default_trust_root,
    load_trust_root,
    resolve_trust_root,
    same_key,
    trust_root_from_env,
)

__all__ = [
    "ARTIFACT_TYPE_SBOM",
    "ARTIFACT_TYPE_SIGNATURE",
    "ARTIFACT_TYPE_SLSA",
    "CYCLONEDX_SPEC_VERSION",
    "DEFAULT_REQUIRED",
    "INTOTO_STATEMENT_TYPE",
    "MEDIA_SBOM",
    "MEDIA_SIGNATURE",
    "MEDIA_SLSA",
    "SLSA_PREDICATE_TYPE",
    "TRUST_ROOT_ENV",
    "AttestationError",
    "AttestationSet",
    "AttestationStore",
    "SignatureError",
    "SupplyChainError",
    "TrustRoot",
    "TrustRootError",
    "TrustedKey",
    "__version__",
    "attest",
    "build_cyclonedx_sbom",
    "build_slsa_provenance",
    "default_trust_root",
    "generate_keypair",
    "load_trust_root",
    "make_verifier",
    "resolve_trust_root",
    "same_key",
    "sign_digest",
    "trust_root_from_env",
    "verify",
    "verify_sbom_document",
    "verify_signature",
    "verify_slsa_document",
]
