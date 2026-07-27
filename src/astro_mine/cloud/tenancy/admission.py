"""Cosign-verified-images-only admission -- the supply-chain trust boundary.

Cloud never runs unsigned images in shared tenancy (``cloud.md`` §9): only an image that is
**cosign-verified**, carries **SLSA provenance** and an **SBOM**, is **digest-pinned**, and
declares a **compatible Core interface version** is admitted. :func:`admit` is the pure,
unit-testable decision (the in-process mirror); :func:`cosign_cluster_policy` is the Kyverno
``ClusterPolicy`` that enforces the same rule at the actual cluster boundary. An unsigned or
Core-incompatible image is refused (``cloud.md`` §6, §9; ``conventions.md`` §9).

Backlog: RM-P1-CLOUD-05 -- https://github.com/astro-mine/astro-mine-cloud/issues/16
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.cloud._compat import validate_core_interface_version
from astro_mine.cloud.k8s import Manifest, object_meta
from astro_mine.cloud.packaging.image import ImageRef

__all__ = ["AdmissionDecision", "ImageAttestation", "admit", "cosign_cluster_policy"]


class ImageAttestation(BaseModel):
    """What is known about an image at admission time: signature + provenance + Core version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str
    cosign_verified: bool = False
    slsa_provenance: bool = False
    sbom: bool = False
    core_interface_version: str | None = None


class AdmissionDecision(BaseModel):
    """The admission outcome: admitted or not, with the reasons for a refusal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    admitted: bool
    reasons: list[str] = Field(default_factory=list)


def admit(
    attestation: ImageAttestation,
    *,
    require_slsa: bool = True,
    require_sbom: bool = True,
    require_core_version: bool = True,
) -> AdmissionDecision:
    """Decide whether *attestation* may run in shared tenancy; collect every failing reason."""
    reasons: list[str] = []
    try:
        ImageRef.parse(attestation.image)
    except ValueError:
        reasons.append("image is not digest-pinned")
    if not attestation.cosign_verified:
        reasons.append("image is not cosign-verified")
    if require_slsa and not attestation.slsa_provenance:
        reasons.append("missing SLSA provenance")
    if require_sbom and not attestation.sbom:
        reasons.append("missing SBOM")
    if require_core_version and attestation.core_interface_version is None:
        reasons.append("no declared Core interface version")
    elif attestation.core_interface_version is not None:
        try:
            validate_core_interface_version(attestation.core_interface_version)
        except ValueError as exc:
            reasons.append(f"incompatible Core interface version: {exc}")
    return AdmissionDecision(admitted=not reasons, reasons=reasons)


def cosign_cluster_policy(name: str = "require-signed-images", *, key_ref: str = "") -> Manifest:
    """A Kyverno ``ClusterPolicy`` admitting only cosign-verified, attested images.

    This is the cluster-boundary enforcement of :func:`admit`; the ``ClusterPolicy`` verifies
    the cosign signature and requires SLSA-provenance + SBOM attestations before a pod runs.
    """
    return {
        "apiVersion": "kyverno.io/v1",
        "kind": "ClusterPolicy",
        "metadata": object_meta(name, component="tenancy"),
        "spec": {
            "validationFailureAction": "Enforce",
            "webhookTimeoutSeconds": 30,
            "rules": [
                {
                    "name": "verify-signature",
                    "match": {"any": [{"resources": {"kinds": ["Pod"]}}]},
                    "verifyImages": [
                        {
                            "imageReferences": ["*"],
                            "attestors": [{"entries": [{"keys": {"publicKeys": key_ref}}]}],
                            "attestations": [
                                {"predicateType": "https://slsa.dev/provenance/v1"},
                                {"predicateType": "https://cyclonedx.org/bom"},
                            ],
                        }
                    ],
                }
            ],
        },
    }
