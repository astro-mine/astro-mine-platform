"""Policy tests (RM-P1-HUB-05): license + dual-use download gating, fail-closed, audited, versioned.

The pure-Python evaluator — the **default**, and the offline tier-1 path (hub.md principle 7). The
same :data:`~tests.conftest.POLICY_CASES` are replayed against the OPA/Rego engine in
``tests/integration/test_opa_policy.py``, which asserts the two engines agree, so the offline
evaluator and the versioned bundle cannot drift.
"""

from __future__ import annotations

import pytest

from astro_mine.core.sadf.enums import GATED_CAPABILITY_TAGS
from astro_mine.hub.index import CatalogEntry
from astro_mine.hub.policy import (
    DEFAULT_ALLOWED_LICENSES,
    Decision,
    DownloadRequest,
    GatedDownload,
    InMemoryAuditLog,
    PythonPolicyEngine,
    evaluate,
    gate,
    policy_data,
    policy_input,
)

from .conftest import DIGEST, POLICY_CASES, PolicyCase, make_manifest


def _entry(
    *, license: str | None = "Apache-2.0", namespace: str = "open", tags: tuple[str, ...] = ()
) -> CatalogEntry:
    return CatalogEntry(
        manifest=make_manifest(license=license, tags=tags),
        digest=DIGEST,
        publisher="p",
        namespace=namespace,
    )


@pytest.mark.parametrize("case", POLICY_CASES, ids=lambda case: case.id)
def test_python_engine_conformance(case: PolicyCase) -> None:
    """The rule table the Rego bundle must reproduce exactly (the shared conformance fixtures)."""
    decision = evaluate(case.entry, case.request)
    assert decision.allowed is case.allowed
    assert decision.code == case.code


def test_allow_permissive_license() -> None:
    decision = evaluate(_entry(), DownloadRequest())
    assert decision.allowed and decision.reason == "allowed"


def test_license_denied() -> None:
    assert not evaluate(_entry(license="GPL-3.0-only"), DownloadRequest()).allowed
    assert not evaluate(_entry(license=None), DownloadRequest()).allowed


def test_require_verified_namespace() -> None:
    request = DownloadRequest(require_verified=True)
    assert not evaluate(_entry(namespace="open"), request).allowed
    assert evaluate(_entry(namespace="curated"), request).allowed


def test_gated_tag_requires_grant() -> None:
    entry = _entry(tags=("operational_targeting",))
    assert not evaluate(entry, DownloadRequest()).allowed  # gated, no grant
    granted = DownloadRequest(grants=frozenset({"operational_targeting"}))
    assert evaluate(entry, granted).allowed


def test_gate_raises_and_audits_on_deny() -> None:
    audit = InMemoryAuditLog()
    with pytest.raises(GatedDownload):
        gate(_entry(license=None), DownloadRequest(), audit=audit)
    assert len(audit.records) == 1
    assert audit.records[0].action == "download" and not audit.records[0].allowed


def test_gate_allows_audits_and_is_append_only() -> None:
    audit = InMemoryAuditLog()
    assert gate(_entry(), DownloadRequest(), audit=audit).allowed
    assert audit.records[0].allowed
    gate(_entry(), DownloadRequest(), audit=audit)
    assert len(audit.records) == 2  # append-only


def test_gate_without_audit() -> None:
    assert gate(_entry(), DownloadRequest()).allowed


# -- the versioned, data-driven bundle ------------------------------------------------------------


def test_rule_data_is_externalized_in_the_bundle() -> None:
    """Licenses/namespaces/gated tags are bundle *data* — changing them is not a code change."""
    data = policy_data()
    assert data.version == "1.0.0"
    assert data.allowed_licenses == DEFAULT_ALLOWED_LICENSES
    assert "Apache-2.0" in data.allowed_licenses and "GPL-3.0-only" not in data.allowed_licenses
    assert data.verified_namespaces == frozenset({"curated", "verified"})


def test_bundle_gated_tags_cannot_drift_from_core() -> None:
    """The bundle restates Core's dual-use taxonomy so Rego can read it as data — it must match.

    Core owns the vocabulary (hub.md §2 principle 2); if Core gates a new tag, this fails until the
    bundle is updated, so a new dual-use capability can never land silently ungated.
    """
    assert policy_data().gated_capability_tags == frozenset(
        tag.value for tag in GATED_CAPABILITY_TAGS
    )


def test_decision_records_the_policy_version_and_engine() -> None:
    decision = evaluate(_entry(), DownloadRequest())
    assert decision.policy_version == policy_data().version
    assert decision.engine == "python"
    assert decision.code == "allowed"


def test_audit_record_carries_the_policy_version() -> None:
    """ "Allowed in March" is only auditable if the rules of March are identifiable (hub.md §10)."""
    audit = InMemoryAuditLog()
    gate(_entry(), DownloadRequest(), audit=audit)
    with pytest.raises(GatedDownload):
        gate(_entry(license="GPL-3.0-only"), DownloadRequest(), audit=audit)

    assert [record.policy_version for record in audit.records] == ["1.0.0", "1.0.0"]
    assert [record.engine for record in audit.records] == ["python", "python"]


def test_engine_seam_is_injectable() -> None:
    """`evaluate()`'s signature is unchanged for existing callers; the engine is a keyword seam."""
    engine = PythonPolicyEngine()
    assert engine.name == "python" and engine.version == "1.0.0"
    assert evaluate(_entry(), DownloadRequest(), engine=engine).allowed
    assert gate(_entry(), DownloadRequest(), engine=engine).allowed


def test_policy_input_is_the_shared_contract() -> None:
    """Both engines see byte-identical input — that is what makes them conformance-testable."""
    entry = _entry(tags=("operational_targeting",), namespace="curated")
    request = DownloadRequest(grants=frozenset({"operational_targeting"}), require_verified=True)
    document = policy_input(entry, request)

    assert document == {
        "reference": "pol:1.0.0",
        "license": "Apache-2.0",
        "namespace": "curated",
        "capability_tags": ["operational_targeting"],
        "grants": ["operational_targeting"],
        "allowed_licenses": sorted(DEFAULT_ALLOWED_LICENSES),
        "require_verified": True,
    }


def test_decision_defaults_stay_backwards_compatible() -> None:
    decision = Decision(True, "pol:1.0.0", "allowed")
    assert decision.code == "" and decision.policy_version == "" and decision.engine == "python"
