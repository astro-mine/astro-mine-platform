"""The shipped Rego really decides what ``authorize()`` decides -- checked with a real OPA.

**A deliberate scope refinement, and why.** The issue asks for "the OPA policy ... applied to a
live admission controller". A Kubernetes admission controller validates *Kubernetes objects*.
``tenancy/opa.authorize()`` decides something else entirely: **who may submit into which tenant
namespace** -- a submission-time question about a user, a tenant and an action, asked before any
Kubernetes object exists. There is no AdmissionReview for a ``submit()`` call, so an admission
controller structurally cannot enforce it. Wiring one up would have produced a green test that
enforced nothing.

The real risk ``rego_policy()`` carries is **drift**: one rule written twice, in Python and in
Rego, with nothing checking they still agree. So that is what is tested. Every case runs through
a real ``opa eval`` against the shipped policy text and through ``authorize()``, and the two
verdicts must match -- across members, non-members, cross-tenant namespaces, unknown tenants and
every action.

The Kyverno half of the same acceptance criterion *is* genuinely an admission concern, and it is
enforced literally against a live admission controller in ``test_admission.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from astro_mine.cloud.tenancy.namespace import tenant_namespace
from astro_mine.cloud.tenancy.opa import Action, AuthzRequest, authorize, rego_policy
from tests.cloud.cluster.conftest import requires

# Needs `opa`, not a cluster -- so it gates on the binary rather than on the kubeconfig.
pytestmark = [pytest.mark.cluster, requires("opa")]

MEMBERSHIPS: dict[str, set[str]] = {"acme": {"alice", "bob"}, "globex": {"carol"}}
USERS = ["alice", "bob", "carol", "mallory"]
TENANTS = ["acme", "globex", "ghost"]
NAMESPACES = ["tenant-acme", "tenant-globex", "tenant-ghost", "default"]
ACTIONS: list[Action] = ["submit", "view", "admin"]


@pytest.fixture(scope="module")
def policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The policy text the *library* ships -- not a copy maintained alongside it."""
    directory = tmp_path_factory.mktemp("opa")
    (directory / "policy.rego").write_text(rego_policy())
    # OPA takes sets as JSON arrays; the Python side keeps them as sets.
    (directory / "data.json").write_text(
        json.dumps({"memberships": {t: sorted(m) for t, m in MEMBERSHIPS.items()}})
    )
    return directory


def _opa_allows(policy: Path, request: AuthzRequest) -> bool:
    result = subprocess.run(
        [
            "opa",
            "eval",
            "-d",
            str(policy / "policy.rego"),
            "-d",
            str(policy / "data.json"),
            "--stdin-input",
            "--format=json",
            "data.astro_mine.tenancy.allow",
        ],
        input=request.model_dump_json(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.stdout, result.stderr
    parsed = json.loads(result.stdout)
    values = parsed.get("result", [])
    if not values:
        return False  # undefined -> the `default allow := false` case
    return bool(values[0]["expressions"][0]["value"])


def test_the_rego_and_the_python_agree_on_every_case(policy: Path) -> None:
    """The whole truth table -- 144 cases -- through a real OPA and through ``authorize()``."""
    disagreements = []
    for user in USERS:
        for tenant in TENANTS:
            for namespace in NAMESPACES:
                for action in ACTIONS:
                    request = AuthzRequest(
                        user=user, tenant=tenant, action=action, namespace=namespace
                    )
                    in_process = authorize(request, memberships=MEMBERSHIPS)
                    in_opa = _opa_allows(policy, request)
                    if in_process != in_opa:
                        disagreements.append((request, in_process, in_opa))

    assert not disagreements, f"the Rego has drifted from authorize(): {disagreements}"


def test_the_truth_table_is_not_uniformly_false(policy: Path) -> None:
    """A control: `default allow := false` would pass the parity test above trivially."""
    allowed = AuthzRequest(
        user="alice", tenant="acme", action="submit", namespace=tenant_namespace("acme")
    )
    denied = AuthzRequest(
        user="alice", tenant="acme", action="submit", namespace=tenant_namespace("globex")
    )
    assert _opa_allows(policy, allowed) is True
    assert _opa_allows(policy, denied) is False
    assert authorize(allowed, memberships=MEMBERSHIPS) is True
    assert authorize(denied, memberships=MEMBERSHIPS) is False
