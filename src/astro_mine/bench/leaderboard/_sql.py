"""SQLAlchemy-backed leaderboard persistence: the submission catalog + the audit trail.

One store, two databases: the same SQLAlchemy Core code runs on **SQLite** (the tests + a
laptop) and on **PostgreSQL** (the ``docker compose`` deployment, bench.md §11) — selected only
by the connection URL, so the persistence path is verified locally yet deploys on Postgres. Each
:class:`~astro_mine.bench.leaderboard._models.Submission` is stored whole as a JSON body keyed by
its content-addressed id, so the schema stays trivial and forward-compatible.

:class:`SqlAuditLog` is the durable :class:`~astro_mine.bench.leaderboard._audit.AuditLog` — the
queryable authN/authZ + verification trail bench#29 requires, indexed on the columns an
investigation actually filters by (subject, action, decision, resource, submission). It is
**append-only**: it exposes an insert and a select, and no update or delete — an audit trail an
operator can quietly rewrite is not one (bench.md §5 puts "Leaderboard metadata & ranks" in
Postgres; the trail lives beside them).

Requires the ``[leaderboard]`` extra (SQLAlchemy; ``psycopg`` for the Postgres URL). Imported
lazily by :func:`astro_mine.bench.leaderboard.create_app`, so the base package stays dep-clean.

Backlog: RM-P0-BENCH-06 — astro-mine-bench#6;
bench#29 — astro-mine-bench#29
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from astro_mine.bench.leaderboard._audit import AuditDecision, AuditEvent
from astro_mine.bench.leaderboard._models import Submission

__all__ = ["SqlAuditLog", "SqlStore"]

_METADATA = MetaData()
_SUBMISSIONS = Table(
    "submissions",
    _METADATA,
    Column("submission_id", String, primary_key=True),
    Column("scenario_id", String, nullable=False, index=True),
    Column("body", JSON, nullable=False),
)
_AUDIT = Table(
    "audit_events",
    _METADATA,
    # A monotonic sequence, so "newest first" is well-defined even for events sharing a timestamp.
    Column("sequence", Integer, primary_key=True, autoincrement=True),
    Column("event_id", String, nullable=False, index=True),
    Column("occurred_at", DateTime(timezone=True), nullable=False, index=True),
    Column("action", String, nullable=False, index=True),
    Column("decision", String, nullable=False, index=True),
    Column("subject", String, nullable=True, index=True),
    Column("resource", String, nullable=True, index=True),
    Column("submission_id", String, nullable=True, index=True),
    Column("body", JSON, nullable=False),
)


class SqlStore:
    """A durable :class:`LeaderboardStore` over any SQLAlchemy engine (SQLite / PostgreSQL)."""

    def __init__(self, url: str | None = None, *, engine: Engine | None = None) -> None:
        """Open (or reuse) an engine for ``url`` and create the schema if absent."""
        if engine is None:
            if url is None:
                raise ValueError("SqlStore needs a database url or an engine")
            engine = create_engine(url)
        self._engine = engine
        _METADATA.create_all(engine)

    def add_submission(self, submission: Submission) -> None:
        body: dict[str, Any] = submission.model_dump(mode="json")
        with self._engine.begin() as connection:
            connection.execute(
                delete(_SUBMISSIONS).where(_SUBMISSIONS.c.submission_id == submission.submission_id)
            )
            connection.execute(
                insert(_SUBMISSIONS).values(
                    submission_id=submission.submission_id,
                    scenario_id=submission.scenario_id,
                    body=body,
                )
            )

    def get_submission(self, submission_id: str) -> Submission | None:
        statement = select(_SUBMISSIONS.c.body).where(_SUBMISSIONS.c.submission_id == submission_id)
        with self._engine.connect() as connection:
            row = connection.execute(statement).fetchone()
        return None if row is None else Submission.model_validate(row[0])

    def list_submissions(self, scenario_id: str) -> list[Submission]:
        statement = (
            select(_SUBMISSIONS.c.body)
            .where(_SUBMISSIONS.c.scenario_id == scenario_id)
            .order_by(_SUBMISSIONS.c.submission_id)
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement).fetchall()
        return [Submission.model_validate(row[0]) for row in rows]

    def remove_submission(self, submission_id: str) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                delete(_SUBMISSIONS).where(_SUBMISSIONS.c.submission_id == submission_id)
            )

    def dispose(self) -> None:
        """Close the engine's pooled connections (production keeps one long-lived engine)."""
        self._engine.dispose()


class SqlAuditLog:
    """A durable, queryable, **append-only** :class:`AuditLog` (SQLite / PostgreSQL).

    The hosted backing for bench#29's audit trail: every authentication, authorization,
    supply-chain verification, and sandboxed-execution decision, filterable by the dimensions an
    investigation uses. It shares the engine with :class:`SqlStore` in a deployment (one database,
    two tables).
    """

    def __init__(self, url: str | None = None, *, engine: Engine | None = None) -> None:
        """Open (or reuse) an engine for ``url`` and create the schema if absent."""
        if engine is None:
            if url is None:
                raise ValueError("SqlAuditLog needs a database url or an engine")
            engine = create_engine(url)
        self._engine = engine
        _METADATA.create_all(engine)

    def record(self, event: AuditEvent) -> None:
        body: dict[str, Any] = event.model_dump(mode="json")
        with self._engine.begin() as connection:
            connection.execute(
                insert(_AUDIT).values(
                    event_id=event.event_id,
                    occurred_at=event.occurred_at,
                    action=event.action,
                    decision=str(event.decision),
                    subject=event.subject,
                    resource=event.resource,
                    submission_id=event.submission_id,
                    body=body,
                )
            )

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
        statement = select(_AUDIT.c.body).order_by(_AUDIT.c.sequence.desc()).limit(max(0, limit))
        if subject is not None:
            statement = statement.where(_AUDIT.c.subject == subject)
        if action is not None:
            statement = statement.where(_AUDIT.c.action == action)
        if decision is not None:
            statement = statement.where(_AUDIT.c.decision == str(decision))
        if resource is not None:
            statement = statement.where(_AUDIT.c.resource == resource)
        if submission_id is not None:
            statement = statement.where(_AUDIT.c.submission_id == submission_id)
        with self._engine.connect() as connection:
            rows = connection.execute(statement).fetchall()
        return [AuditEvent.model_validate(row[0]) for row in rows]
