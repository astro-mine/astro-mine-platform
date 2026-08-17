# SPDX-License-Identifier: Apache-2.0
"""RBAC/OPA authorization -- who may submit into which tenant namespace.

Fine-grained authorization on submission and namespace access is enforced with OPA
(``cloud.md`` §9; ``conventions.md`` §9). :func:`authorize` is the in-process decision -- a
user may act on a tenant only if they are a member of it *and* the request targets that
tenant's namespace -- so cross-tenant access is denied. :func:`rego_policy` ships the
equivalent Rego for the cluster OPA/Gatekeeper deployment, keeping one rule expressed in two
places that must agree.

Backlog: RM-P1-CLOUD-05 -- astro-mine-cloud#16
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from astro_mine.cloud.tenancy.namespace import tenant_namespace

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["AuthzRequest", "authorize", "rego_policy"]

Action = Literal["submit", "view", "admin"]


class AuthzRequest(BaseModel):
    """An access request: a user acting on a tenant's namespace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user: str
    tenant: str
    action: Action
    namespace: str


def authorize(request: AuthzRequest, *, memberships: Mapping[str, set[str]]) -> bool:
    """Whether *request* is allowed, given tenant *memberships* (tenant -> member users).

    Allowed iff the user is a member of the tenant and the request targets that tenant's own
    namespace -- so a member of tenant A cannot submit into tenant B's namespace.
    """
    members = memberships.get(request.tenant, set())
    if request.user not in members:
        return False
    return request.namespace == tenant_namespace(request.tenant)


def rego_policy() -> str:
    """Return the Rego policy mirroring :func:`authorize` for cluster OPA/Gatekeeper."""
    return """package astro_mine.tenancy

default allow := false

# A user may act on a tenant only if they are a member and target that tenant's namespace.
allow if {
    input.user in data.memberships[input.tenant]
    input.namespace == sprintf("tenant-%s", [input.tenant])
}
"""
