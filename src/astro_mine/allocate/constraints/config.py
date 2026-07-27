"""``ConstraintConfig`` — Allocate's *declared modeling policy* for the builders (RM-P1-ALLOC-03).

The constraint builders lift **upstream truth** — Worlds traversability, Link contact windows,
Fleet SADF budgets, Prospect value — into the solver-neutral IR (allocate.md §5/§6). What they do
*not* do is invent physics: quantitative durations/energy costs come from cached cost tables or
Sim/Surrogate rollouts (:mod:`astro_mine.allocate.constraints.costs`), and the physical fields
(slope, illumination, contact windows, budgets) come from the Core contracts in the
:class:`~astro_mine.allocate.ConstraintContext`.

This module holds the small residue that is genuinely Allocate's own *modeling choice* — the
keep-out thresholds a builder compares upstream truth against, the SI budget/mode keys it reads,
and the episode↔TDB epoch anchor. It is **declared, versioned, and content-addressed**: every
field is explicit (no magic constant buried in a builder), the model is frozen and
``extra="forbid"``, and its ``content_hash``
is folded into the plan's provenance so a plan pins the exact policy it was compiled under
(conventions.md §5; allocate.md §8 determinism).
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.core.hashing import content_hash_json
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.units import Epoch

__all__ = [
    "CommsPolicy",
    "ConstraintConfig",
    "CostPolicy",
    "PowerPolicy",
    "TerrainPolicy",
]


class _Policy(BaseModel):
    """Base for every policy block: immutable and reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TerrainPolicy(_Policy):
    """Keep-out thresholds the terrain builder compares Worlds traversability against.

    None of these are physics — they are the *eligibility thresholds* Allocate applies to
    Worlds' truth (slope/illumination/bearing) and the asset's declared limits.
    ``default_max_slope_deg`` applies only when a SADF asset declares no
    ``mobility.contact[*].max_slope_deg`` of its own.
    ``require_illuminated`` forbids placing a task at a location in full shadow (a PSR keep-out for
    solar-dependent operations); ``ground_pressure_margin`` scales an asset's declared max ground
    pressure before comparing it against the regolith bearing capacity.
    """

    default_max_slope_deg: float = Field(default=25.0, gt=0.0, le=90.0)
    require_illuminated: bool = False
    enforce_bearing_capacity: bool = True
    ground_pressure_margin: float = Field(default=1.0, gt=0.0)


class PowerPolicy(_Policy):
    """How the power builder reads energy capacity and reserves the housekeeping floor.

    ``energy_budget_key`` names the SI budget in ``AssetRef.budgets`` that carries the deliverable
    energy (defaults to ``energy_j``); when absent, the builder falls back to the SADF
    ``power.storage[*].capacity_j`` sum. ``horizon_s`` is the planning horizon over which the SADF
    ``power.floor_w`` housekeeping draw is reserved from that capacity (a *feasibility* reservation,
    not the safety floor Guard independently enforces — allocate.md §6). ``default_mode_power_w`` is
    used to derive a task's energy from its duration only when neither a cost-table energy entry nor
    a SADF mode load is available (a declared, auditable fallback).
    """

    energy_budget_key: str = "energy_j"
    horizon_s: float = Field(default=0.0, ge=0.0)
    default_mode_power_w: float = Field(default=100.0, gt=0.0)
    reserve_floor: bool = True


class CommsPolicy(_Policy):
    """Which tasks must relay inside a contact window, and how episode time maps to TDB.

    A task is relay-gated when its ``task_id`` is in ``relay_required_task_ids`` **or** its Core
    :class:`~astro_mine.core.messages.enums.TaskKind` is in ``relay_required_kinds`` (both empty ⇒
    the comms builder is a no-op — Allocate presumes nothing). ``epoch0`` is the Core
    :class:`~astro_mine.core.units.Epoch` that episode-time second ``0`` maps to, so a task's
    episode-time window and Link's TDB contact intervals
    (:class:`~astro_mine.core.messages.model.ContactInterval`) are compared in one timeline
    (conventions.md §5: epochs are SPICE TDB/ET — RFC-0007). It is **required with no default**:
    an epoch origin carries its own time scale and must be stated, never smuggled as a bare
    ``float`` that silently anchors at J2000 (RM-P1-ALLOC-08). Declaring a comms policy therefore
    obliges the caller to state the episode↔TDB anchor; ``ConstraintConfig.comms`` is left
    ``None`` when no relay gating applies. ``node_id_for_asset`` overrides the default asset-id →
    contact-graph-node-id identity mapping.
    """

    relay_required_task_ids: frozenset[str] = Field(default_factory=frozenset)
    relay_required_kinds: frozenset[TaskKind] = Field(default_factory=frozenset)
    epoch0: Epoch
    node_id_for_asset: dict[str, str] = Field(default_factory=dict)

    def is_relay_gated(self, task_id: str, kind: TaskKind) -> bool:
        """Whether a task must execute inside a comms contact window."""
        return task_id in self.relay_required_task_ids or kind in self.relay_required_kinds

    def node_id(self, asset_id: str) -> str:
        """The contact-graph node id for an asset (identity unless overridden)."""
        return self.node_id_for_asset.get(asset_id, asset_id)


class CostPolicy(_Policy):
    """How a ``(task, asset)`` pair's SI cost is **priced** into the objective (RM-P1-ALLOC-04).

    The exchange rate between the SI cost of an assignment — the energy (J) and the time (s) *this*
    asset spends doing *this* task, resolved by the power and terrain builders from the cached
    :class:`~astro_mine.allocate.constraints.CostTable` — and the abstract value units a task's
    :class:`~astro_mine.allocate.ValueEstimate` is denominated in. It is what turns the objective
    from "value earned" into **"value earned minus what earning it cost"**, and therefore what makes
    two different feasible assignments of the same task score differently.

    There is deliberately **no default rate**: value units are abstract, so no joules-per-value
    constant is physically privileged, and a silently-defaulted price would put a number nobody
    chose into every plan's objective. Declaring the policy is the act of stating the rate — the
    same reason :attr:`CommsPolicy.epoch0` has no default. A policy that prices *nothing* (both
    rates zero) would contribute a uniformly-zero objective family, so it is rejected rather than
    silently ignored: leave ``ConstraintConfig.cost`` at ``None`` to opt out.
    """

    #: Value units charged per joule of the pair's energy cost.
    energy_price_per_j: float = Field(default=0.0, ge=0.0)
    #: Value units charged per second of the pair's duration (an opportunity cost of the asset's
    #: time, independent of the energy it draws).
    time_price_per_s: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _prices_something(self) -> Self:
        if self.energy_price_per_j == 0.0 and self.time_price_per_s == 0.0:
            raise ValueError(
                "a CostPolicy that prices neither energy nor time contributes nothing to the "
                "objective; set a price, or leave ConstraintConfig.cost as None to opt out"
            )
        return self


class ConstraintConfig(_Policy):
    """The complete declared modeling policy for the RM-P1-ALLOC-03 builders.

    Frozen, ``extra="forbid"``, and content-addressed: two identical configs
    :meth:`content_hash` identically across machines, and the hash is recorded in a plan's
    provenance so the policy a plan was compiled under is pinned and reproducible.
    """

    terrain: TerrainPolicy = Field(default_factory=TerrainPolicy)
    power: PowerPolicy = Field(default_factory=PowerPolicy)
    # None ⇒ no relay gating (the comms builder is skipped). A CommsPolicy is required whole —
    # and with it the episode↔TDB `epoch0` anchor — precisely so the epoch origin is never
    # silently defaulted (RM-P1-ALLOC-08; RFC-0007).
    comms: CommsPolicy | None = None
    # None ⇒ the objective carries no per-pair cost family: assignments are scored on task value
    # alone, so (under the exactly-one cover) every feasible plan ties. Declaring a CostPolicy is
    # what makes the objective — and hence the optimality gap — discriminate between plans.
    cost: CostPolicy | None = None

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this policy (its immutable identity)."""
        return content_hash_json(self.model_dump(mode="json"))
