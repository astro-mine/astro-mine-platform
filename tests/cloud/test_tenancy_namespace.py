"""Namespace-per-tenant isolation -- default-deny, RBAC, quota, full bundle."""

from __future__ import annotations

from astro_mine.cloud.tenancy.namespace import (
    allow_platform_egress_network_policy,
    default_deny_network_policy,
    limit_range,
    resource_quota,
    role,
    role_binding,
    tenant_manifests,
    tenant_namespace,
)


def test_tenant_namespace_is_sanitized() -> None:
    assert tenant_namespace("Acme Corp") == "tenant-acme-corp"


def test_default_deny_blocks_all_traffic() -> None:
    policy = default_deny_network_policy("acme")
    assert policy["kind"] == "NetworkPolicy"
    spec = policy["spec"]
    assert spec["podSelector"] == {}  # selects every pod
    assert set(spec["policyTypes"]) == {"Ingress", "Egress"}
    assert "ingress" not in spec and "egress" not in spec  # no allow rules => deny all


def test_resource_quota_caps_requests() -> None:
    quota = resource_quota("acme", hard={"requests.cpu": "100", "requests.nvidia.com/gpu": "8"})
    assert quota["kind"] == "ResourceQuota"
    assert quota["spec"]["hard"]["requests.cpu"] == "100"


def test_role_is_namespaced_and_scoped() -> None:
    r = role("acme")
    assert r["metadata"]["namespace"] == "tenant-acme"
    api_groups = {g for rule in r["rules"] for g in rule["apiGroups"]}
    assert {"batch", "argoproj.io", "ray.io"} <= api_groups


def test_role_binding_binds_subjects() -> None:
    rb = role_binding("acme", subjects=["alice", "bob"])
    names = {s["name"] for s in rb["subjects"]}
    assert names == {"alice", "bob"}
    assert rb["roleRef"]["name"] == "tenant-acme-workload"


def test_limit_range_defaults() -> None:
    lr = limit_range("acme")
    limits = lr["spec"]["limits"][0]
    assert limits["default"]["cpu"] == "1"
    assert limits["defaultRequest"]["memory"] == "128Mi"


def test_tenant_manifests_bundle_is_complete() -> None:
    bundle = tenant_manifests("acme", quota={"requests.cpu": "50"}, subjects=["alice"])
    kinds = [m["kind"] for m in bundle]
    assert kinds == [
        "Namespace",
        "ServiceAccount",
        "Role",
        "RoleBinding",
        "ResourceQuota",
        "LimitRange",
        "NetworkPolicy",  # default-deny
        "NetworkPolicy",  # allow-intra
        "NetworkPolicy",  # allow-dns
        "NetworkPolicy",  # allow-platform: egress to the artifact store, or the tenant runs nothing
        "LocalQueue",
    ]
    # every object lives in the tenant's own namespace (except the cluster-scoped Namespace)
    namespaced = [m for m in bundle if m["kind"] != "Namespace"]
    assert all(m["metadata"]["namespace"] == "tenant-acme" for m in namespaced)


def test_a_tenant_may_reach_the_platform_object_store() -> None:
    """Egress to the platform namespace is allowed -- without it a tenant can run nothing.

    The default-deny baseline blocks all egress, and neither the intra-namespace nor the DNS
    allowance covers the artifact store. But every workload stages its inputs from and writes its
    outputs to that store, so a tenant that cannot reach it fails on its first read. A live cluster
    found this: a job was admitted through Kueue exactly as designed, ran, and then died.
    """
    policy = allow_platform_egress_network_policy("acme")

    assert policy["kind"] == "NetworkPolicy"
    assert policy["spec"]["policyTypes"] == ["Egress"]
    assert policy["spec"]["podSelector"] == {}  # every pod in the tenant

    (rule,) = policy["spec"]["egress"]
    (destination,) = rule["to"]
    assert destination["namespaceSelector"]["matchLabels"] == {
        "kubernetes.io/metadata.name": "astro-mine-system"
    }

    # ...and it is *only* that namespace. An empty selector here would allow egress everywhere and
    # quietly undo the default-deny baseline the rest of the bundle exists to establish.
    assert "podSelector" not in destination
    assert allow_platform_egress_network_policy("acme", platform_namespace="elsewhere")["spec"][
        "egress"
    ][0]["to"][0]["namespaceSelector"]["matchLabels"] == {
        "kubernetes.io/metadata.name": "elsewhere"
    }
