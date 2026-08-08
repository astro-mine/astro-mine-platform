"""License + export-control download gating + the audit log (RM-P1-HUB-05).

The download-boundary gate (hub.md §9; ``LUNAR-SR-001``): evaluate license + export-control/dual-use
policy against the Core manifest **before returning bytes**, failing closed on denial and consuming
Core's gated-capability vocabulary (``operational_targeting`` + the sealed set). Every decision is
written to the append-only :class:`AuditLog`, stamped with the **policy-bundle version** that
produced it.

Two engines evaluate **one** rule over **one** set of data (``policy/rego/``):

- :class:`PythonPolicyEngine` — the default; pure Python, so the offline tier-1 path gates with
  nothing installed (hub.md principle 7).
- :class:`OpaPolicyEngine` — OPA over the versioned **Rego bundle** (binary or sidecar), so
  governance rules evolve as a bundle release rather than a code change (hub.md §3, §11).

Backlog: RM-P1-HUB-05 — astro-mine-hub#5
"""

from __future__ import annotations

from astro_mine.hub.policy._audit import AuditLog, AuditRecord, InMemoryAuditLog
from astro_mine.hub.policy._opa import OpaPolicyEngine, OpaUnavailable, opa_engine_from_env
from astro_mine.hub.policy._policy import (
    BUNDLE_DIR,
    DEFAULT_ALLOWED_LICENSES,
    Decision,
    DownloadRequest,
    GatedDownload,
    PolicyData,
    PolicyEngine,
    PythonPolicyEngine,
    evaluate,
    gate,
    policy_data,
    policy_input,
)

__all__ = [
    "BUNDLE_DIR",
    "DEFAULT_ALLOWED_LICENSES",
    "AuditLog",
    "AuditRecord",
    "Decision",
    "DownloadRequest",
    "GatedDownload",
    "InMemoryAuditLog",
    "OpaPolicyEngine",
    "OpaUnavailable",
    "PolicyData",
    "PolicyEngine",
    "PythonPolicyEngine",
    "evaluate",
    "gate",
    "opa_engine_from_env",
    "policy_data",
    "policy_input",
]
