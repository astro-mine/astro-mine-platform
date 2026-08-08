"""Leaderboard persistence — the store contract + an in-memory backend (RM-P0-BENCH-06).

A :class:`LeaderboardStore` persists :class:`~astro_mine.bench.leaderboard._models.Submission`
records. :class:`InMemoryStore` is the dependency-clean default (the local tier and tests); the
SQLAlchemy-backed :class:`~astro_mine.bench.leaderboard._sql.SqlStore` (SQLite for tests,
PostgreSQL for the ``docker compose`` deployment) is the durable backend, behind the
``[leaderboard]`` extra.

Backlog: RM-P0-BENCH-06 — astro-mine-bench#6
"""

from __future__ import annotations

from typing import Protocol

from astro_mine.bench.leaderboard._models import Submission

__all__ = ["InMemoryStore", "LeaderboardStore"]


class LeaderboardStore(Protocol):
    """Persistence for leaderboard submissions — add, fetch by id, and list per scenario."""

    def add_submission(self, submission: Submission) -> None:
        """Persist ``submission`` (idempotent on ``submission_id`` — a re-submit replaces)."""
        ...

    def get_submission(self, submission_id: str) -> Submission | None:
        """Return the submission with ``submission_id``, or ``None`` if absent."""
        ...

    def list_submissions(self, scenario_id: str) -> list[Submission]:
        """Every submission for ``scenario_id``, in a stable order."""
        ...

    def remove_submission(self, submission_id: str) -> None:
        """Retract ``submission_id`` from the board (a no-op if absent).

        The only mutation the board allows, and it is privileged: retraction is an
        ``ranking:mutate`` action an admin performs through
        :meth:`~astro_mine.bench.leaderboard.LeaderboardService.retract`, and it is audit-logged
        (bench#29). The submission's :class:`ProvenanceBundle` is deliberately left in the object
        store, so a retracted entry stays *auditable* even once it is off the board.
        """
        ...


class InMemoryStore:
    """A process-local :class:`LeaderboardStore` — the dependency-clean default backend."""

    def __init__(self) -> None:
        self._by_id: dict[str, Submission] = {}

    def add_submission(self, submission: Submission) -> None:
        self._by_id[submission.submission_id] = submission

    def get_submission(self, submission_id: str) -> Submission | None:
        return self._by_id.get(submission_id)

    def list_submissions(self, scenario_id: str) -> list[Submission]:
        submissions = [s for s in self._by_id.values() if s.scenario_id == scenario_id]
        return sorted(submissions, key=lambda s: s.submission_id)

    def remove_submission(self, submission_id: str) -> None:
        self._by_id.pop(submission_id, None)
