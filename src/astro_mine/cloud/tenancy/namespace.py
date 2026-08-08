"""Namespace-per-tenant isolation -- the baseline tenancy bundle.

Compiles the objects that isolate a tenant on shared hardware (``cloud.md`` §9): a
``Namespace``, a least-privilege ``ServiceAccount`` + namespaced RBAC (``Role`` /
``RoleBinding``), a ``ResourceQuota`` and ``LimitRange``, and **default-deny**
``NetworkPolicies`` (deny all ingress/egress, then narrowly allow intra-namespace traffic and
DNS). :func:`tenant_manifests` assembles the full bundle plus the tenant's Kueue
``LocalQueue``, so standing up a tenant is one call and every object carries the tenant label
for selection and cost attribution (``cloud.md`` §10).

Backlog: RM-P1-CLOUD-05 -- astro-mine-cloud#16
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.cloud.k8s import Manifest, object_meta, sanitize_name
from astro_mine.cloud.sched.kueue import local_queue

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "allow_platform_egress_network_policy",
    "default_deny_network_policy",
    "limit_range",
    "resource_quota",
    "role",
    "role_binding",
    "service_account",
    "tenant_manifests",
    "tenant_namespace",
]


def tenant_namespace(tenant: str) -> str:
    """The Kubernetes namespace name for *tenant* (``tenant-<name>``)."""
    return sanitize_name(f"tenant-{tenant}")


def namespace(tenant: str) -> Manifest:
    """The tenant ``Namespace`` (labelled with the tenant for selection)."""
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": object_meta(tenant_namespace(tenant), tenant=tenant, component="tenancy"),
    }


def service_account(tenant: str) -> Manifest:
    """A least-privilege ``ServiceAccount`` for the tenant's workloads."""
    ns = tenant_namespace(tenant)
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": object_meta(f"{ns}-workload", namespace=ns, tenant=tenant, component="tenancy"),
    }


def role(tenant: str) -> Manifest:
    """A namespaced ``Role`` letting the tenant manage its own jobs/pods/workflows only."""
    ns = tenant_namespace(tenant)
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": object_meta(f"{ns}-workload", namespace=ns, tenant=tenant, component="tenancy"),
        "rules": [
            {
                "apiGroups": [""],
                "resources": ["pods", "pods/log"],
                "verbs": ["get", "list", "watch"],
            },
            {
                "apiGroups": ["batch"],
                "resources": ["jobs"],
                "verbs": ["create", "get", "list", "watch", "delete"],
            },
            {
                "apiGroups": ["argoproj.io", "ray.io"],
                "resources": ["workflows", "rayjobs"],
                "verbs": ["create", "get", "list", "watch", "delete"],
            },
        ],
    }


def role_binding(tenant: str, *, subjects: Sequence[str]) -> Manifest:
    """Bind the tenant ``Role`` to its users (RBAC group members)."""
    ns = tenant_namespace(tenant)
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": object_meta(f"{ns}-workload", namespace=ns, tenant=tenant, component="tenancy"),
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": f"{ns}-workload",
        },
        "subjects": [
            {"kind": "User", "name": user, "apiGroup": "rbac.authorization.k8s.io"}
            for user in subjects
        ],
    }


def resource_quota(tenant: str, *, hard: Mapping[str, str]) -> Manifest:
    """A ``ResourceQuota`` capping the tenant's total requests (a DoS rail, ``cloud.md`` §9)."""
    ns = tenant_namespace(tenant)
    return {
        "apiVersion": "v1",
        "kind": "ResourceQuota",
        "metadata": object_meta(f"{ns}-quota", namespace=ns, tenant=tenant, component="tenancy"),
        "spec": {"hard": dict(sorted(hard.items()))},
    }


def limit_range(
    tenant: str,
    *,
    default: Mapping[str, str] | None = None,
    default_request: Mapping[str, str] | None = None,
) -> Manifest:
    """A ``LimitRange`` giving unqualified pods sane default limits/requests."""
    ns = tenant_namespace(tenant)
    limits: Manifest = {"type": "Container"}
    limits["default"] = dict(default or {"cpu": "1", "memory": "1Gi"})
    limits["defaultRequest"] = dict(default_request or {"cpu": "100m", "memory": "128Mi"})
    return {
        "apiVersion": "v1",
        "kind": "LimitRange",
        "metadata": object_meta(f"{ns}-limits", namespace=ns, tenant=tenant, component="tenancy"),
        "spec": {"limits": [limits]},
    }


def default_deny_network_policy(tenant: str) -> Manifest:
    """Deny **all** ingress and egress in the tenant namespace (the isolation baseline)."""
    ns = tenant_namespace(tenant)
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": object_meta(
            f"{ns}-default-deny", namespace=ns, tenant=tenant, component="tenancy"
        ),
        # empty podSelector selects every pod; no ingress/egress rules => deny everything.
        "spec": {"podSelector": {}, "policyTypes": ["Ingress", "Egress"]},
    }


def allow_intra_namespace_network_policy(tenant: str) -> Manifest:
    """Allow traffic *within* the tenant namespace (pods of one tenant may talk)."""
    ns = tenant_namespace(tenant)
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": object_meta(
            f"{ns}-allow-intra", namespace=ns, tenant=tenant, component="tenancy"
        ),
        "spec": {
            "podSelector": {},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [{"from": [{"podSelector": {}}]}],
            "egress": [{"to": [{"podSelector": {}}]}],
        },
    }


def allow_platform_egress_network_policy(
    tenant: str, *, platform_namespace: str = "astro-mine-system"
) -> Manifest:
    """Allow egress to the platform namespace, where the artifact store lives.

    Without this a tenant can run **nothing**. The default-deny baseline blocks all egress; the
    intra-namespace and DNS allowances do not cover the object store, which lives in the platform
    namespace. But every workload stages its inputs from and writes its outputs to that store
    (``cloud.md`` §5) -- so a tenant pod that cannot reach it fails on its first read, and a tenant
    assembled from :func:`tenant_manifests` was therefore incapable of completing a single job.

    Found by the live-cluster harness: a tenant-scoped job was admitted through Kueue exactly as
    designed, ran, and then died -- because it could not talk to the store. No in-process test could
    have seen it; a NetworkPolicy only exists once a real CNI enforces it.

    Scoped to the namespace, not opened to the world: the isolation baseline stands, and this is the
    one hole it needs.
    """
    ns = tenant_namespace(tenant)
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": object_meta(
            f"{ns}-allow-platform", namespace=ns, tenant=tenant, component="tenancy"
        ),
        "spec": {
            "podSelector": {},
            "policyTypes": ["Egress"],
            "egress": [
                {
                    "to": [
                        {
                            "namespaceSelector": {
                                "matchLabels": {"kubernetes.io/metadata.name": platform_namespace}
                            }
                        }
                    ]
                }
            ],
        },
    }


def allow_dns_network_policy(tenant: str) -> Manifest:
    """Allow egress to cluster DNS (kube-system), without which pods cannot resolve names."""
    ns = tenant_namespace(tenant)
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": object_meta(
            f"{ns}-allow-dns", namespace=ns, tenant=tenant, component="tenancy"
        ),
        "spec": {
            "podSelector": {},
            "policyTypes": ["Egress"],
            "egress": [
                {
                    "to": [
                        {
                            "namespaceSelector": {
                                "matchLabels": {"kubernetes.io/metadata.name": "kube-system"}
                            }
                        }
                    ],
                    "ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}],
                }
            ],
        },
    }


def tenant_manifests(
    tenant: str,
    *,
    quota: Mapping[str, str],
    subjects: Sequence[str] = (),
    cluster_queue: str | None = None,
) -> list[Manifest]:
    """Assemble the full namespace-per-tenant bundle (isolation + RBAC + quota + queue)."""
    ns = tenant_namespace(tenant)
    manifests = [
        namespace(tenant),
        service_account(tenant),
        role(tenant),
        role_binding(tenant, subjects=subjects),
        resource_quota(tenant, hard=quota),
        limit_range(tenant),
        default_deny_network_policy(tenant),
        allow_intra_namespace_network_policy(tenant),
        allow_dns_network_policy(tenant),
        allow_platform_egress_network_policy(tenant),
        local_queue(ns, namespace=ns, cluster_queue=cluster_queue or tenant),
    ]
    return manifests
