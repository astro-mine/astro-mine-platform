"""RBAC/OPA authorization -- cross-tenant access is denied."""

from __future__ import annotations

from astro_mine.cloud.tenancy.opa import AuthzRequest, authorize, rego_policy

MEMBERSHIPS = {"acme": {"alice", "bob"}, "globex": {"carol"}}


def test_member_may_submit_into_own_namespace() -> None:
    request = AuthzRequest(user="alice", tenant="acme", action="submit", namespace="tenant-acme")
    assert authorize(request, memberships=MEMBERSHIPS) is True


def test_non_member_is_denied() -> None:
    request = AuthzRequest(user="carol", tenant="acme", action="submit", namespace="tenant-acme")
    assert authorize(request, memberships=MEMBERSHIPS) is False


def test_cross_namespace_access_is_denied() -> None:
    # alice is an acme member but targets globex's namespace -> denied
    request = AuthzRequest(user="alice", tenant="acme", action="submit", namespace="tenant-globex")
    assert authorize(request, memberships=MEMBERSHIPS) is False


def test_unknown_tenant_is_denied() -> None:
    request = AuthzRequest(user="alice", tenant="ghost", action="submit", namespace="tenant-ghost")
    assert authorize(request, memberships=MEMBERSHIPS) is False


def test_rego_policy_mirrors_the_rule() -> None:
    rego = rego_policy()
    assert "default allow := false" in rego
    assert "data.memberships[input.tenant]" in rego
