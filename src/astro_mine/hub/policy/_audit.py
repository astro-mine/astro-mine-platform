# SPDX-License-Identifier: Apache-2.0
"""Append-only audit log for gating + governance decisions (RM-P1-HUB-05; hub.md §5, §10).

Every download-gating decision and every curation action (yank / deprecate / promote) is written to
an **append-only** audit log — the compliance record hub.md §5/§10 requires. `InMemoryAuditLog`
is the tier-1 default; the hosted tier persists the same :class:`AuditRecord` stream to durable,
structured JSON storage behind the same :class:`AuditLog` seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = ["AuditLog", "AuditRecord", "InMemoryAuditLog"]


@dataclass(frozen=True)
class AuditRecord:
    """One audited decision: the action, the artifact, whether it was allowed, and why.

    A **gating** decision also records *under which policy it was made* — the Rego bundle revision
    and the engine that evaluated it — because "allowed in March" is only auditable if the rules of
    March are identifiable (hub.md §5, §10; RM-P1-HUB-05). Governance actions (yank/deprecate/
    promote) leave these empty.
    """

    action: str  # "download" | "yank" | "deprecate" | "promote"
    reference: str
    allowed: bool
    reason: str
    policy_version: str = ""  # the Rego bundle revision the decision was made under
    engine: str = ""  # "python" (offline default) | "opa"


class AuditLog(Protocol):
    """An append-only sink for :class:`AuditRecord`s."""

    def record(self, record: AuditRecord) -> None:
        """Append ``record`` (never mutate or delete existing records)."""
        ...


class InMemoryAuditLog:
    """A process-local append-only :class:`AuditLog` (tier-1 + tests)."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def record(self, record: AuditRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        """The audit trail, oldest first (a read-only snapshot)."""
        return tuple(self._records)
