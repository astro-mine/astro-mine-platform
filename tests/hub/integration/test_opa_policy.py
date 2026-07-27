"""Integration: the OPA/Rego engine vs. the Python evaluator — the conformance suite (RM-P1-HUB-05).

hub.md §3/§11 require download gating to be **data-driven Rego bundles** so governance rules evolve
without code changes, while principle 7 requires the offline path to gate with nothing installed.
Two engines therefore implement one rule — and this is the test that stops them drifting: the *same*
:data:`~tests.conftest.POLICY_CASES` are fed to both, and every allow/deny outcome (and the rule
that fired) must be identical.

Runs in the `integration` CI job (which installs the `opa` binary) or locally whenever `opa` is on
PATH / ``HUB_OPA_URL`` points at a sidecar; skipped otherwise, since the pure-Python evaluator is
what the offline gate exercises (``tests/test_policy.py``).
"""

from __future__ import annotations

import os
import shutil

import pytest

from astro_mine.hub.policy import (
    DownloadRequest,
    GatedDownload,
    InMemoryAuditLog,
    OpaPolicyEngine,
    OpaUnavailable,
    PythonPolicyEngine,
    evaluate,
    gate,
    policy_data,
)

from ..conftest import POLICY_CASES, PolicyCase

pytestmark = pytest.mark.integration

_OPA = shutil.which("opa")
_OPA_URL = os.environ.get("HUB_OPA_URL")

requires_opa = pytest.mark.skipif(
    not (_OPA or _OPA_URL), reason="the `opa` binary or HUB_OPA_URL is required"
)


@pytest.fixture(scope="module")
def opa() -> OpaPolicyEngine:
    return OpaPolicyEngine(url=_OPA_URL) if _OPA_URL else OpaPolicyEngine()


@requires_opa
@pytest.mark.parametrize("case", POLICY_CASES, ids=lambda case: case.id)
def test_engines_agree_on_every_case(case: PolicyCase, opa: OpaPolicyEngine) -> None:
    """The conformance contract: identical inputs, identical outcomes, on both engines."""
    entry, request = case.entry, case.request
    python = evaluate(entry, request, engine=PythonPolicyEngine())
    rego = evaluate(entry, request, engine=opa)

    assert rego.allowed is python.allowed is case.allowed
    assert rego.code == python.code == case.code
    assert rego.reference == python.reference


@requires_opa
def test_opa_decisions_record_the_bundle_version(opa: OpaPolicyEngine) -> None:
    """The Rego bundle's revision travels with the decision and into the audit log (hub.md §10)."""
    case = POLICY_CASES[0]
    decision = evaluate(case.entry, case.request, engine=opa)
    assert decision.engine == "opa"
    assert decision.policy_version == policy_data().version == opa.version

    audit = InMemoryAuditLog()
    gate(case.entry, case.request, audit=audit, engine=opa)
    assert audit.records[0].policy_version == opa.version
    assert audit.records[0].engine == "opa"


@requires_opa
def test_opa_gate_fails_closed_and_audits(opa: OpaPolicyEngine) -> None:
    denied = next(case for case in POLICY_CASES if not case.allowed)
    audit = InMemoryAuditLog()
    with pytest.raises(GatedDownload):
        gate(denied.entry, denied.request, audit=audit, engine=opa)
    assert audit.records[0].allowed is False
    assert audit.records[0].engine == "opa"


@requires_opa
def test_gated_capability_tag_is_enforced_by_rego(opa: OpaPolicyEngine) -> None:
    """conventions.md §12 / LUNAR-SR-001: `operational_targeting` is gated at the boundary."""
    case = next(c for c in POLICY_CASES if c.id == "operational-targeting-gated")
    assert evaluate(case.entry, case.request, engine=opa).allowed is False

    granted = DownloadRequest(grants=frozenset({"operational_targeting"}))
    assert evaluate(case.entry, granted, engine=opa).allowed is True


@requires_opa
def test_unreachable_sidecar_fails_closed() -> None:
    """Policy that cannot be evaluated denies; there is no "allow because OPA did not answer"."""
    engine = OpaPolicyEngine(url="http://127.0.0.1:1", timeout=1.0)
    case = POLICY_CASES[0]
    with pytest.raises(OpaUnavailable):
        engine.evaluate(case.entry, case.request)
