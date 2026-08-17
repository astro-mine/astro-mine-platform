# SPDX-License-Identifier: Apache-2.0
"""``CostTable`` — cached per-(task, asset) duration/energy costs for the builders (RM-P1-ALLOC-03).

Allocate does not re-derive physics: "where durations/costs come from physics, they are sourced
via [Sim]/[Surrogate] rollouts or **cached cost tables**, not re-derived" (allocate.md §6). This is
that cached cost table — the quantitative input the comms (fit-in-window), power (energy budget),
and terrain (edge-weight) builders read, keyed by ``(task_id, asset_id)`` because a task's cost
depends on *which* asset does it (a tracked excavator and a wheeled rover traverse the same slope
differently).

The table is a frozen, content-addressed artifact: its :meth:`~CostTable.content_hash` is folded
into a plan's provenance alongside the request and config hashes so a plan pins the exact cost
inputs it was compiled against (conventions.md §5). A missing entry is **not** silently zero — a
builder that needs a cost it cannot find flags a *degraded* build (a declared fallback was used)
rather than fabricating a physical number.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.hashing import content_hash_json

__all__ = ["CostEntry", "CostTable"]


class _Model(BaseModel):
    """Base for the cost models: immutable, reject unknown fields loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CostEntry(_Model):
    """The cached cost of one asset performing one task.

    ``duration_s`` is the traversal/operation duration (SI seconds) the comms builder fits inside a
    contact window and the power builder multiplies by a mode power draw; ``energy_j`` is the direct
    energy cost (SI joules) the power builder sums against an asset's energy budget. Both are
    optional — a table may carry only what its producer measured — and both are non-negative.
    """

    duration_s: float | None = Field(default=None, ge=0.0)
    energy_j: float | None = Field(default=None, ge=0.0)


class CostTable(_Model):
    """A content-addressed table of cached ``(task_id, asset_id) → CostEntry`` costs.

    Stored as a flat map keyed by a ``(task_id, asset_id)`` string (a JSON-object key cannot be
    a tuple) so the wire form and content hash are byte-stable. Empty by default: with no entries
    builders fall back to their declared policy defaults and mark the build degraded.
    """

    entries: dict[str, CostEntry] = Field(default_factory=dict)

    @staticmethod
    def key(task_id: str, asset_id: str) -> str:
        """The flat lookup key for a ``(task_id, asset_id)`` pair."""
        return f"{task_id}\x1f{asset_id}"

    @classmethod
    def of(cls, costs: dict[tuple[str, str], CostEntry]) -> CostTable:
        """Build a table from a ``{(task_id, asset_id): CostEntry}`` mapping."""
        return cls(entries={cls.key(t, a): e for (t, a), e in costs.items()})

    def lookup(self, task_id: str, asset_id: str) -> CostEntry | None:
        """The cached cost for a pair, or ``None`` when the table has no entry for it."""
        return self.entries.get(self.key(task_id, asset_id))

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this cost table (its immutable identity)."""
        return content_hash_json(self.model_dump(mode="json"))
