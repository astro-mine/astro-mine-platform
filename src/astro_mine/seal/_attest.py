# SPDX-License-Identifier: Apache-2.0
"""SLSA provenance + SBOM documents, the referrer vocabulary, and publish-side ``attest``.

The two non-signature attestations the supply chain attaches to an artifact alongside the cosign
signature (guard.md §9.5; hub.md §9):

- **SLSA provenance** — an in-toto Statement (``predicateType`` ``https://slsa.dev/provenance/v1``)
  binding the artifact digest to its builder and resolved inputs (the SLSA build track).
- **SBOM** — a **CycloneDX** bill of materials of the artifact's components.

These are documented, **reduced-order** generators (no live Syft / slsa-github-generator), so the
tier-1 local path builds real, well-shaped attestations **offline and deterministically** (no
clock); the hosted tier can swap in Syft/CycloneDX + a real SLSA generator without changing the
wiring. Documents are content-addressed with the platform canonical form
(:func:`astro_mine.core.hashing.canonical_json`).

:func:`attest` is the **publish-side orchestrator** (RM-P1-SEAL-03): it signs an artifact digest and
attaches the signature, provenance, and SBOM. It is **registry-agnostic** — it drives the
:class:`AttestationStore` port below (``str``/``bytes`` only), never a concrete registry type, so
Seal keeps its Core-only dependency and stays out of the registry/publish plane (seal.md §1, §2.1).
Hub adapts its OCI ``Registry`` to the port and
imports this function rather than reimplementing the orchestration; Bench, Fleet, and any other
producer bind whatever store they have.

Relocated by [RFC-0005](https://github.com/astro-mine/docs/blob/main/rfc/0005-seal-supply-chain-companion.md)
§Sequencing from ``astro-mine-hub``'s ``supply_chain/_attest.py`` + ``_supply_chain.py``,
behavior-preserving (same document bytes, same referrer types).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from astro_mine.core.hashing import canonical_json
from astro_mine.seal._signing import sign_digest

__all__ = [
    "ARTIFACT_TYPE_SBOM",
    "ARTIFACT_TYPE_SIGNATURE",
    "ARTIFACT_TYPE_SLSA",
    "CYCLONEDX_SPEC_VERSION",
    "INTOTO_STATEMENT_TYPE",
    "MEDIA_SBOM",
    "MEDIA_SIGNATURE",
    "MEDIA_SLSA",
    "SLSA_PREDICATE_TYPE",
    "AttestationSet",
    "AttestationStore",
    "attest",
    "build_cyclonedx_sbom",
    "build_slsa_provenance",
]

# in-toto / SLSA / CycloneDX identifiers
INTOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
SLSA_PREDICATE_TYPE = "https://slsa.dev/provenance/v1"
CYCLONEDX_SPEC_VERSION = "1.5"

# OCI artifactTypes for the three attestation referrers (fetchable by type via the Referrers API).
ARTIFACT_TYPE_SIGNATURE = "application/vnd.astro-mine.signature.v1"
ARTIFACT_TYPE_SLSA = "application/vnd.astro-mine.provenance.slsa.v1"
ARTIFACT_TYPE_SBOM = "application/vnd.astro-mine.sbom.cyclonedx.v1"

# Blob media types the referrer layers carry.
MEDIA_SIGNATURE = "application/vnd.astro-mine.signature.v1+json"
MEDIA_SLSA = "application/vnd.in-toto+json"
MEDIA_SBOM = "application/vnd.cyclonedx+json"


@runtime_checkable
class AttestationStore(Protocol):
    """The **registry-agnostic port** Seal's ``attest`` / ``verify`` orchestrators drive.

    The seam that keeps Seal out of the registry plane (seal.md §1: "storing and serving artifacts
    is Hub; Seal only proves and checks what an artifact is and who produced it"). Every method
    speaks only ``str`` digests, ``str`` media/artifact types, and ``bytes`` payloads — **no OCI,
    Hub, or other concrete registry type crosses this boundary** — so Seal depends on Core alone
    (seal.md §2.1) while Hub, Bench, or any producer binds its own store to it.

    An implementation is a thin adapter over whatever content-addressed store the caller has (an OCI
    registry with the Referrers API, an in-memory dict, a directory of files). The contract:

    - **Content addresses are the identity.** A ``digest`` is ``"sha256:<hex>"`` and always
      addresses the bytes it names.
    - **Reads are integrity-checked and fail closed.** :meth:`read_attestation` and
      :meth:`verify_integrity` MUST **raise** if the stored bytes do not hash to their address (a
      tampered store must never be able to return bytes silently). Seal converts *any* exception
      from a store method into a fail-closed
      :class:`~astro_mine.seal._supply_chain.SupplyChainError` — it never treats a store failure as
      "no evidence, therefore fine".
    - **Missing evidence is empty, not an error.** :meth:`attestation_digests` returns an empty
      sequence when an artifact carries no attestation of that type; Seal is what turns that absence
      into a refusal, so the *policy* lives in one place.
    """

    def attach_attestation(
        self, *, subject: str, artifact_type: str, media_type: str, payload: bytes
    ) -> str:
        """Store ``payload`` as an attestation of ``subject``; return the attestation's digest.

        ``artifact_type`` is one of the ``ARTIFACT_TYPE_*`` constants (how the attestation is found
        again); ``media_type`` is the payload's own media type. On an OCI store this writes a
        referrer manifest whose subject is ``subject`` and whose single layer is ``payload``.
        """
        ...

    def attestation_digests(self, subject: str, *, artifact_type: str) -> Sequence[str]:
        """``subject``'s attestation digests of ``artifact_type`` — empty when there are none."""
        ...

    def read_attestation(self, digest: str) -> bytes:
        """The attestation payload stored at ``digest`` — **raise** if its bytes are not intact."""
        ...

    def verify_integrity(self, subject: str) -> None:
        """Re-hash ``subject``'s bytes against their content address — **raise** on mismatch."""
        ...


@dataclass(frozen=True)
class AttestationSet:
    """The digests of an artifact's signature, SLSA provenance, and SBOM attestations.

    What :func:`attest` returns and :func:`~astro_mine.seal._supply_chain.verify` reads back. Plain
    ``sha256:<hex>`` strings — deliberately **not** a registry descriptor type, so the publish-side
    result crosses no registry boundary (seal.md §1).
    """

    signature: str
    slsa: str
    sbom: str


def _digest_parts(digest: str) -> tuple[str, str]:
    algorithm, _, hexpart = digest.partition(":")
    if algorithm != "sha256" or not hexpart:
        raise ValueError(f"expected a 'sha256:<hex>' digest, got {digest!r}")
    return algorithm, hexpart


def build_slsa_provenance(
    *,
    subject_name: str,
    subject_digest: str,
    builder_id: str,
    build_type: str = "https://astro-mine.org/hub/publish/v1",
    inputs: Sequence[str] = (),
) -> dict[str, Any]:
    """An in-toto Statement carrying SLSA v1 provenance for ``subject_digest``.

    ``inputs`` are the ``sha256:<hex>`` content hashes of the build inputs (recorded as resolved
    dependencies). Deterministic — no clock — so the same build reproduces the same provenance.
    """
    algorithm, hexpart = _digest_parts(subject_digest)
    return {
        "_type": INTOTO_STATEMENT_TYPE,
        "predicateType": SLSA_PREDICATE_TYPE,
        "subject": [{"name": subject_name, "digest": {algorithm: hexpart}}],
        "predicate": {
            "buildDefinition": {
                "buildType": build_type,
                "externalParameters": {},
                "resolvedDependencies": [
                    {"digest": {_digest_parts(h)[0]: _digest_parts(h)[1]}} for h in inputs
                ],
            },
            "runDetails": {"builder": {"id": builder_id}, "metadata": {}},
        },
    }


def build_cyclonedx_sbom(
    *,
    name: str,
    version: str,
    components: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """A CycloneDX SBOM for artifact ``name``/``version`` over its ``components``.

    Each component is ``{"name": ..., "version": ...}``. Deterministic and offline (no Syft scan).
    """
    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_SPEC_VERSION,
        "version": 1,
        "metadata": {"component": {"type": "data", "name": name, "version": version}},
        "components": [
            {"type": "library", "name": c["name"], "version": c.get("version", "")}
            for c in components
        ],
    }


def attest(
    store: AttestationStore,
    subject: str,
    *,
    private_key_pem: bytes,
    name: str,
    version: str,
    builder_id: str,
    inputs: Sequence[str] = (),
    components: Sequence[Mapping[str, str]] = (),
) -> AttestationSet:
    """Sign ``subject`` and attach its cosign signature, SLSA provenance, and SBOM to ``store``.

    The **publish-side** half of verify-twice (hub.md §9; RM-P1-SEAL-03), registry-agnostic over the
    :class:`AttestationStore` port. ``subject`` is the artifact's content digest (``sha256:<hex>``).
    ``builder_id`` identifies *what produced the artifact* and is **required** — it is a provenance
    claim, so Seal will not default it on a producer's behalf. ``inputs`` are the content digests of
    the build inputs; ``components`` are the SBOM components (``{"name": ..., "version": ...}``).

    Documents are serialized with the platform canonical form
    (:func:`astro_mine.core.hashing.canonical_json`), so the same artifact + key yields
    byte-identical attestations on every producer.

    Returns the :class:`AttestationSet` of attachment digests. This function *attests*; it does not
    verify — call :func:`~astro_mine.seal._supply_chain.verify` immediately after for
    verify-at-admission (the first of the two verifies).
    """
    signature = sign_digest(subject, private_key_pem)
    signature_digest = store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SIGNATURE,
        media_type=MEDIA_SIGNATURE,
        payload=signature.model_dump_json().encode("utf-8"),
    )
    slsa = build_slsa_provenance(
        subject_name=f"{name}:{version}",
        subject_digest=subject,
        builder_id=builder_id,
        inputs=inputs,
    )
    slsa_digest = store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SLSA,
        media_type=MEDIA_SLSA,
        payload=canonical_json(slsa),
    )
    sbom = build_cyclonedx_sbom(name=name, version=version, components=components)
    sbom_digest = store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SBOM,
        media_type=MEDIA_SBOM,
        payload=canonical_json(sbom),
    )
    return AttestationSet(signature=signature_digest, slsa=slsa_digest, sbom=sbom_digest)
