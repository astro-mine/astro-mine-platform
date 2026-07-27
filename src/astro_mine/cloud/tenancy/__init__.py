"""Tenancy -- namespace-per-tenant isolation, cosign admission, OPA authz.

Cloud runs *others' code* for *multiple tenants* on *shared hardware*, so isolation is the
central security concern (``cloud.md`` §9):

- :mod:`.namespace` compiles the **namespace-per-tenant** baseline -- Namespace, least-privilege
  ServiceAccount + RBAC, ResourceQuota / LimitRange, and **default-deny NetworkPolicies** --
  so one tenant cannot reach or exhaust another;
- :mod:`.admission` is the **cosign-verified-images-only** admission rail: only signed images
  carrying SLSA provenance + an SBOM and a compatible Core interface version are admitted, at
  the cluster boundary;
- :mod:`.opa` is the RBAC/OPA authorization decision (submission / namespace access), mirrored
  by the shipped Rego policy.

Backlog: RM-P1-CLOUD-05 -- https://github.com/astro-mine/astro-mine-cloud/issues/16
"""

from __future__ import annotations

from astro_mine.cloud.tenancy.admission import (
    AdmissionDecision,
    ImageAttestation,
    admit,
    cosign_cluster_policy,
)
from astro_mine.cloud.tenancy.namespace import (
    default_deny_network_policy,
    tenant_manifests,
    tenant_namespace,
)
from astro_mine.cloud.tenancy.opa import AuthzRequest, authorize, rego_policy

__all__ = [
    "AdmissionDecision",
    "AuthzRequest",
    "ImageAttestation",
    "admit",
    "authorize",
    "cosign_cluster_policy",
    "default_deny_network_policy",
    "rego_policy",
    "tenant_manifests",
    "tenant_namespace",
]
