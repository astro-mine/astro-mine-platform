"""Shared test helpers — a Core plugin-manifest factory + the policy conformance fixtures.

:data:`POLICY_CASES` is the **shared** gating-conformance table (RM-P1-HUB-05): the *same* inputs
are fed to the pure-Python evaluator (``test_policy``) and to the OPA/Rego engine
(``tests/integration/test_opa_policy``), which asserts the two agree on every one. That is what
keeps the offline evaluator and the versioned Rego bundle from silently drifting apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from astro_mine.core.registry import PluginKind, PluginManifest, Provenance
from astro_mine.hub._content import content_hash
from astro_mine.hub.index import CatalogEntry
from astro_mine.hub.policy import DownloadRequest


def make_manifest(
    name: str = "pol",
    version: str = "1.0.0",
    *,
    kind: PluginKind = PluginKind.POLICY,
    interfaces: Mapping[str, str] | None = None,
    tags: Sequence[str] = (),
    license: str | None = "Apache-2.0",
    description: str = "",
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
) -> PluginManifest:
    """A valid Core :class:`PluginManifest` with a content-addressed provenance digest."""
    return PluginManifest(
        name=name,
        version=version,
        kind=kind,
        core_interfaces=dict(interfaces or {"policy": "0.1.0"}),
        capability_tags=list(tags),
        license=license,
        description=description or None,
        inputs=list(inputs),
        outputs=list(outputs),
        provenance=Provenance(
            input_hashes=[],
            source_content_hashes={},
            digest=content_hash(f"{name}:{version}".encode()),
        ),
    )


DIGEST = "sha256:" + "a" * 64


@dataclass(frozen=True)
class PolicyCase:
    """One download-gating scenario and the outcome **both** engines must produce."""

    id: str
    allowed: bool
    code: str
    license: str | None = "Apache-2.0"
    namespace: str = "open"
    tags: tuple[str, ...] = ()
    grants: frozenset[str] = frozenset()
    require_verified: bool = False
    allowed_licenses: frozenset[str] | None = field(default=None)

    @property
    def entry(self) -> CatalogEntry:
        return CatalogEntry(
            manifest=make_manifest(license=self.license, tags=self.tags),
            digest=DIGEST,
            publisher="p",
            namespace=self.namespace,
        )

    @property
    def request(self) -> DownloadRequest:
        request = DownloadRequest(grants=self.grants, require_verified=self.require_verified)
        if self.allowed_licenses is None:
            return request
        return DownloadRequest(
            grants=self.grants,
            allowed_licenses=self.allowed_licenses,
            require_verified=self.require_verified,
        )


#: The conformance table (hub.md §9; ``LUNAR-SR-001``). Every dual-use tag Core gates appears here,
#: granted and ungranted, so the two engines are compared on the whole rule surface — not a sample.
POLICY_CASES: tuple[PolicyCase, ...] = (
    PolicyCase("permissive-license", allowed=True, code="allowed"),
    PolicyCase("mit-license", allowed=True, code="allowed", license="MIT"),
    PolicyCase("copyleft-denied", allowed=False, code="license_denied", license="GPL-3.0-only"),
    PolicyCase("no-license-denied", allowed=False, code="license_denied", license=None),
    PolicyCase(
        "license-not-in-requested-set",
        allowed=False,
        code="license_denied",
        license="Apache-2.0",
        allowed_licenses=frozenset({"MIT"}),
    ),
    PolicyCase(
        "unverified-namespace-denied",
        allowed=False,
        code="verification_required",
        require_verified=True,
    ),
    PolicyCase(
        "curated-namespace-allowed",
        allowed=True,
        code="allowed",
        namespace="curated",
        require_verified=True,
    ),
    PolicyCase(
        "verified-namespace-allowed",
        allowed=True,
        code="allowed",
        namespace="verified",
        require_verified=True,
    ),
    PolicyCase(
        "operational-targeting-gated",
        allowed=False,
        code="capability_gated",
        tags=("operational_targeting",),
    ),
    PolicyCase(
        "operational-targeting-granted",
        allowed=True,
        code="allowed",
        tags=("operational_targeting",),
        grants=frozenset({"operational_targeting"}),
    ),
    PolicyCase(
        "ground-truth-access-gated",
        allowed=False,
        code="capability_gated",
        tags=("ground_truth_access",),
    ),
    PolicyCase(
        "live-link-prediction-gated",
        allowed=False,
        code="capability_gated",
        tags=("comms.live_mission_link_prediction",),
    ),
    PolicyCase(
        "multiple-gated-tags-partially-granted",
        allowed=False,
        code="capability_gated",
        tags=("operational_targeting", "ground_truth_access"),
        grants=frozenset({"operational_targeting"}),
    ),
    PolicyCase(
        "multiple-gated-tags-fully-granted",
        allowed=True,
        code="allowed",
        tags=("operational_targeting", "ground_truth_access"),
        grants=frozenset({"operational_targeting", "ground_truth_access"}),
    ),
    PolicyCase(
        "ungated-tag-needs-no-grant",
        allowed=True,
        code="allowed",
        tags=("mobility.wheeled",),
    ),
    PolicyCase(
        "license-denied-takes-precedence-over-gated-tag",
        allowed=False,
        code="license_denied",
        license="GPL-3.0-only",
        tags=("operational_targeting",),
    ),
)
