# SPDX-License-Identifier: Apache-2.0
"""License + export-control download gating at the boundary (RM-P1-HUB-05; hub.md §9).

Hub evaluates **license** and **export-control/dual-use** policy at the **download boundary**,
against the artifact's Core manifest — **before returning any bytes** (hub.md §9; ``LUNAR-SR-001``).
The gate **fails closed**: :func:`gate` raises :class:`GatedDownload` (and audit-logs the denial) so
a denied download returns no bytes. It consumes **Core's** dual-use vocabulary
(:data:`~astro_mine.core.sadf.enums.GATED_CAPABILITY_TAGS`, including the reserved
``operational_targeting`` tag) — never a Hub-private taxonomy.

**Two engines, one rule, no drift.** hub.md §3/§11 specify OPA with "data-driven Rego bundles, so
governance rules evolve without code changes", while hub.md principle 7 requires the offline tier-1
path to work with nothing installed. So the rule is expressed **twice, over one set of data**:

- :class:`PythonPolicyEngine` — the **default**, and the offline path: pure Python, no OPA, no
  network (conventions.md §7 tier 1).
- :class:`~astro_mine.hub.policy.OpaPolicyEngine` — the hosted tier: the **versioned Rego bundle**
  in ``policy/rego/`` evaluated by OPA (embedded binary or sidecar), so a license or export-control
  change ships as a **bundle release**, not a code change plus a redeploy (hub.md §11).

Both read their rule *data* — allowed licenses, verified namespaces, the gated capability tags —
from the same bundle ``data.json`` (:func:`policy_data`), and both stamp the **bundle version** onto
every :class:`Decision` for the audit log (hub.md §5, §10). A shared conformance suite feeds
identical inputs to both and asserts identical outcomes, so they cannot silently diverge.

Operational maneuver targeting / guided EDL stay partitioned out via the reserved tag; the full
mission-architecture gating is P3 (hub.md §12) — this reserves the hook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from astro_mine.core.sadf.enums import GATED_CAPABILITY_TAGS
from astro_mine.hub.index import CatalogEntry
from astro_mine.hub.policy._audit import AuditLog, AuditRecord

__all__ = [
    "BUNDLE_DIR",
    "DEFAULT_ALLOWED_LICENSES",
    "Decision",
    "DownloadRequest",
    "GatedDownload",
    "PolicyData",
    "PolicyEngine",
    "PythonPolicyEngine",
    "evaluate",
    "gate",
    "policy_data",
    "policy_input",
]

#: The versioned Rego bundle: the policy *logic* (``download.rego``) and its *data* (``data.json``).
BUNDLE_DIR = Path(__file__).parent / "rego"


class GatedDownload(Exception):
    """A download is denied by policy — raised so no bytes are returned (fail closed)."""


@dataclass(frozen=True)
class PolicyData:
    """The bundle's rule **data** — the part that evolves without a code change (hub.md §11)."""

    version: str
    allowed_licenses: frozenset[str]
    verified_namespaces: frozenset[str]
    gated_capability_tags: frozenset[str]


@lru_cache(maxsize=4)
def policy_data(bundle_dir: Path = BUNDLE_DIR) -> PolicyData:
    """Load the Rego bundle's ``data.json`` — the single source of rule data for **both** engines.

    The gated capability tags are Core's (:data:`GATED_CAPABILITY_TAGS`); the bundle restates them
    so Rego can read them as data, and ``test_policy`` asserts the two sets are identical, so a new
    Core dual-use tag cannot land ungated.
    """
    document = json.loads((bundle_dir / "data.json").read_bytes())
    rules = document["astro_mine"]["hub"]["policy"]
    return PolicyData(
        version=str(rules["version"]),
        allowed_licenses=frozenset(rules["allowed_licenses"]),
        verified_namespaces=frozenset(rules["verified_namespaces"]),
        gated_capability_tags=frozenset(rules["gated_capability_tags"]),
    )


#: Permissive OSI licenses allowed by default (Apache-2.0 is the charter default, §10.4) — read from
#: the policy bundle, not hard-coded, so widening the set is a bundle release.
DEFAULT_ALLOWED_LICENSES: frozenset[str] = policy_data().allowed_licenses


@dataclass(frozen=True)
class Decision:
    """A gating verdict for one artifact: allowed or not, why, and under which policy version."""

    allowed: bool
    reference: str
    reason: str
    #: A stable machine-readable rule id — what the conformance suite compares across engines
    #: (prose is an engine's own; the *outcome* is the contract).
    code: str = ""
    #: The Rego bundle revision the decision was made under — the audit record (hub.md §5, §10).
    policy_version: str = ""
    #: Which engine decided: ``"python"`` (offline default) or ``"opa"``.
    engine: str = "python"


@dataclass(frozen=True)
class DownloadRequest:
    """What the requester presents at the download boundary."""

    grants: frozenset[str] = frozenset()  # capability grants the requester holds
    allowed_licenses: frozenset[str] = DEFAULT_ALLOWED_LICENSES
    require_verified: bool = False  # only accept curated/verified-namespace artifacts


@runtime_checkable
class PolicyEngine(Protocol):
    """The gating engine seam — Python (offline) or OPA/Rego (hosted), behind one call."""

    @property
    def name(self) -> str:
        """The engine's identity, recorded on the decision (``"python"`` / ``"opa"``)."""
        ...

    @property
    def version(self) -> str:
        """The policy-bundle revision this engine evaluates."""
        ...

    def evaluate(self, entry: CatalogEntry, request: DownloadRequest) -> Decision:
        """The gating verdict for ``entry`` under ``request``."""
        ...


def policy_input(entry: CatalogEntry, request: DownloadRequest) -> dict[str, object]:
    """The policy **input document** — identical for both engines (the conformance contract)."""
    return {
        "reference": entry.reference,
        "license": entry.license,
        "namespace": entry.namespace,
        "capability_tags": sorted(entry.capability_tags),
        "grants": sorted(request.grants),
        "allowed_licenses": sorted(request.allowed_licenses),
        "require_verified": request.require_verified,
    }


class PythonPolicyEngine:
    """The pure-Python evaluator — the **default**, and the offline tier-1 path (no OPA required).

    License must be present and permitted; if ``require_verified``, the artifact must be in a
    curated/verified namespace; and any **gated capability tag** the artifact declares must be
    covered by a matching grant. The first failing rule denies (fail closed) — the same order
    ``download.rego`` applies.
    """

    def __init__(self, data: PolicyData | None = None) -> None:
        self._data = data if data is not None else policy_data()

    @property
    def name(self) -> str:
        return "python"

    @property
    def version(self) -> str:
        return self._data.version

    def evaluate(self, entry: CatalogEntry, request: DownloadRequest) -> Decision:
        def decide(allowed: bool, code: str, reason: str) -> Decision:
            return Decision(
                allowed=allowed,
                reference=entry.reference,
                reason=reason,
                code=code,
                policy_version=self._data.version,
                engine=self.name,
            )

        if entry.license is None or entry.license not in request.allowed_licenses:
            return decide(False, "license_denied", f"license {entry.license!r} is not permitted")
        if request.require_verified and entry.namespace not in self._data.verified_namespaces:
            return decide(False, "verification_required", "a verified/curated artifact is required")
        ungranted = sorted(
            tag
            for tag in entry.capability_tags
            if tag in self._data.gated_capability_tags and tag not in request.grants
        )
        if ungranted:
            return decide(
                False,
                "capability_gated",
                f"gated capability {ungranted} requires an access grant",
            )
        return decide(True, "allowed", "allowed")


#: The Core dual-use tags gated at the download boundary (kept for consumers that import it).
_GATED_TAGS: frozenset[str] = frozenset(tag.value for tag in GATED_CAPABILITY_TAGS)

_DEFAULT_ENGINE = PythonPolicyEngine()


def evaluate(
    entry: CatalogEntry, request: DownloadRequest, *, engine: PolicyEngine | None = None
) -> Decision:
    """The gating verdict for ``entry`` under ``request`` — license, verification, then dual-use.

    ``engine`` selects the evaluator; it defaults to the pure-Python one, so the offline tier-1 path
    gates correctly with nothing installed. Pass an
    :class:`~astro_mine.hub.policy.OpaPolicyEngine` to evaluate the same rule from the versioned
    Rego bundle instead — the call signature is unchanged for every existing caller.
    """
    return (engine if engine is not None else _DEFAULT_ENGINE).evaluate(entry, request)


def gate(
    entry: CatalogEntry,
    request: DownloadRequest,
    *,
    audit: AuditLog | None = None,
    engine: PolicyEngine | None = None,
) -> Decision:
    """Enforce the download gate: audit the decision and **raise on denial** (no bytes returned).

    Returns the allow :class:`Decision` when permitted; raises :class:`GatedDownload` otherwise.
    Every decision (allow or deny) is written to ``audit`` when one is supplied, stamped with the
    **policy version** that produced it (hub.md §9, §10).
    """
    decision = evaluate(entry, request, engine=engine)
    if audit is not None:
        audit.record(
            AuditRecord(
                action="download",
                reference=decision.reference,
                allowed=decision.allowed,
                reason=decision.reason,
                policy_version=decision.policy_version,
                engine=decision.engine,
            )
        )
    if not decision.allowed:
        raise GatedDownload(decision.reason)
    return decision
