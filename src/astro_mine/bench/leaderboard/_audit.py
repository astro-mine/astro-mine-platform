# SPDX-License-Identifier: Apache-2.0
"""The authN/authZ + verification audit trail (bench#29 AC4; bench.md §2 principle 7, §9).

bench.md §9 ends its anti-cheat section on the sentence this module implements: *"results carry full
provenance so **disputes are auditable**"* — and §2's seventh principle is *"Open and auditable …
Trust comes from transparency, not from a black box."* The
:class:`~astro_mine.bench.leaderboard._provenance.ProvenanceBundle` already makes the *score*
auditable. This makes the **decisions around it** auditable: who was authenticated, what they were
allowed or denied and why, whether a submission's attestations verified, and how a sandboxed run
ended.

An :class:`AuditEvent` is a structured, queryable record — not a log line to be grepped:
:meth:`AuditLog.query` filters by subject, action, decision, and resource, so "show me every denial
for this scenario" and "why was this entry flagged" are answerable questions.

Two backends behind one protocol, mirroring the leaderboard's other stores:
:class:`InMemoryAuditLog`
(the dependency-clean default; the local tier and tests) and
:class:`~astro_mine.bench.leaderboard._sql.SqlAuditLog` (SQLite/Postgres, the durable hosted
backend, behind the ``[leaderboard]`` extra).

Events are **append-only by construction** — frozen models, and neither backend exposes an update or
a delete. An audit trail an operator can quietly rewrite is not one.

Backlog: bench#29 — astro-mine-bench#29
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.bench.scenario._hash import content_hash

__all__ = [
    "AuditDecision",
    "AuditEvent",
    "AuditLog",
    "InMemoryAuditLog",
    "audit_event",
]


class AuditDecision(StrEnum):
    """What the trail records: an authorization outcome, or the outcome of a verification."""

    ALLOW = "allow"
    DENY = "deny"
    #: An attestation / integrity check that passed (supply-chain verify, sandboxed execution).
    VERIFIED = "verified"
    #: An attestation / integrity check that failed — the submission was rejected.
    REJECTED = "rejected"


class AuditEvent(BaseModel):
    """One decision, recorded. Frozen: the trail is append-only.

    ``event_id`` content-addresses the record, so the same decision logged twice is recognisably the
    same decision, and an event cannot be silently altered without changing its id.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    occurred_at: datetime
    action: str = Field(min_length=1)
    decision: AuditDecision
    #: The authenticated subject, or ``None`` when authentication itself is what failed.
    subject: str | None = None
    issuer: str | None = None
    #: What was acted on: a scenario id, a Hub reference, a submission id.
    resource: str = ""
    #: Why — the policy engine's reason, the verifier's failure, or the sandbox's rejection detail.
    reason: str = ""
    submission_id: str | None = None
    job_id: str | None = None
    #: The W3C trace id of the pipeline span this decision belongs to, so an audit record joins the
    #: OTel trace for the same submission (bench.md §10).
    trace_id: str | None = None
    #: Structured detail — e.g. the :class:`~._supply_chain.AttestationVerdict`, or a sandbox
    #: outcome's status/usage. Data only; never re-executed.
    detail: dict[str, Any] = Field(default_factory=dict)


def audit_event(
    *,
    action: str,
    decision: AuditDecision,
    subject: str | None = None,
    issuer: str | None = None,
    resource: str = "",
    reason: str = "",
    submission_id: str | None = None,
    job_id: str | None = None,
    trace_id: str | None = None,
    detail: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> AuditEvent:
    """Build a content-addressed :class:`AuditEvent`, stamping it with the current UTC time."""
    when = occurred_at if occurred_at is not None else datetime.now(UTC)
    body = {
        "action": action,
        "decision": str(decision),
        "issuer": issuer,
        "job_id": job_id,
        "occurred_at": when.isoformat(),
        "reason": reason,
        "resource": resource,
        "subject": subject,
        "submission_id": submission_id,
    }
    return AuditEvent(
        event_id=content_hash(body),
        occurred_at=when,
        action=action,
        decision=decision,
        subject=subject,
        issuer=issuer,
        resource=resource,
        reason=reason,
        submission_id=submission_id,
        job_id=job_id,
        trace_id=trace_id,
        detail=detail or {},
    )


class AuditLog(Protocol):
    """An append-only, queryable trail of authN/authZ and verification decisions (bench#29 AC4)."""

    def record(self, event: AuditEvent) -> None:
        """Append ``event``. Never updates or removes — the trail is immutable."""
        ...

    def query(
        self,
        *,
        subject: str | None = None,
        action: str | None = None,
        decision: AuditDecision | None = None,
        resource: str | None = None,
        submission_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """The most recent matching events, newest first; unset filters match everything."""
        ...


class InMemoryAuditLog:
    """A process-local :class:`AuditLog` — the dependency-clean default backend."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self._events.append(event)

    def query(
        self,
        *,
        subject: str | None = None,
        action: str | None = None,
        decision: AuditDecision | None = None,
        resource: str | None = None,
        submission_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        matched = [
            event
            for event in reversed(self._events)
            if (subject is None or event.subject == subject)
            and (action is None or event.action == action)
            and (decision is None or event.decision is decision)
            and (resource is None or event.resource == resource)
            and (submission_id is None or event.submission_id == submission_id)
        ]
        return matched[: max(0, limit)]

    def __len__(self) -> int:
        """How many events the trail holds — it only ever grows."""
        return len(self._events)

    @property
    def events(self) -> Sequence[AuditEvent]:
        """Every recorded event, oldest first."""
        return tuple(self._events)
