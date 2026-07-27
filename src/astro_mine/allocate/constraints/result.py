"""Structured build artifacts the constraint builders emit alongside the IR (RM-P1-ALLOC-03).

Every constraint a builder derives is emitted **into the versioned IR** (the single contract a
solver lowers and :func:`~astro_mine.allocate.verify_feasible` re-checks). These types are the
*explanation* that rides beside it (allocate.md §10, the ``explain`` concern): the human-readable
findings behind each keep-out / infeasibility, the degraded-mode notes (a distribution used
deterministically, a cost fell back to a policy default), and the structured intermediate data
(per-pair keep-outs, comms windows, energy costs) a solver uses to *construct* a plan before the IR
independently *verifies* it. They are internal dataclasses, not wire types — the IR and the
``Allocation`` result are the serialized contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from astro_mine.allocate.model.ir.model import AllocationIR, Constraint, ObjectiveTerm

__all__ = [
    "ConstraintCompilation",
    "ConstraintFinding",
    "ConstraintReport",
    "PowerResult",
    "TerrainResult",
    "WindowResult",
]

#: A ``(task_id, asset_id)`` assignment pair a builder has forbidden (keep-out / no window).
Pair = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ConstraintFinding:
    """One human-readable reason a builder forbade a pair or found a task infeasible.

    ``code`` is a stable machine token (e.g. ``terrain.slope_keepout``, ``comms.no_window``);
    ``constraint_id`` names the IR constraint that carries the finding when there is one. Findings
    feed a plan's ``binding_constraints`` and, on infeasibility, its
    :class:`~astro_mine.allocate.InfeasibilityCertificate` explanation (allocate.md §10; the IIS
    is RM-P1-ALLOC-06).
    """

    code: str
    detail: str
    task_id: str | None = None
    asset_id: str | None = None
    constraint_id: str | None = None


@dataclass(frozen=True, slots=True)
class TerrainResult:
    """The terrain builder's contribution: keep-out constraints + resolved per-pair durations."""

    constraints: tuple[Constraint, ...] = ()
    forbidden: frozenset[Pair] = frozenset()
    findings: tuple[ConstraintFinding, ...] = ()
    degraded: tuple[str, ...] = ()
    #: Resolved traversal/operation duration (s) per ``(task_id, asset_id)`` (from the cost table
    #: or a declared fallback) — consumed by the comms and power builders.
    durations: Mapping[Pair, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WindowResult:
    """The comms builder's contribution: contact-window gating constraints + per-pair windows."""

    constraints: tuple[Constraint, ...] = ()
    forbidden: frozenset[Pair] = frozenset()
    findings: tuple[ConstraintFinding, ...] = ()
    degraded: tuple[str, ...] = ()
    #: Feasible ``[earliest, latest]`` start window (episode s) per allowed ``(task, asset)`` pair.
    pair_windows: Mapping[Pair, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PowerResult:
    """The power builder's contribution: per-asset energy-budget constraints + the cost data."""

    constraints: tuple[Constraint, ...] = ()
    findings: tuple[ConstraintFinding, ...] = ()
    degraded: tuple[str, ...] = ()
    #: Energy cost (J) per ``(task_id, asset_id)`` and available energy (J) per asset.
    energy_costs: Mapping[Pair, float] = field(default_factory=dict)
    energy_capacity: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    """The aggregate explanation of a constrained compile — everything but the IR itself.

    ``forbidden`` is every keep-out/no-window ``(task, asset)`` pair; ``pair_windows`` the comms
    windows a solver places starts within; ``durations``/``energy_costs``/``energy_capacity`` the
    cost data a solver honors; ``findings``/``degraded`` the explanation and honesty notes.
    """

    findings: tuple[ConstraintFinding, ...] = ()
    degraded: tuple[str, ...] = ()
    forbidden: frozenset[Pair] = frozenset()
    pair_windows: Mapping[Pair, tuple[float, float]] = field(default_factory=dict)
    durations: Mapping[Pair, float] = field(default_factory=dict)
    energy_costs: Mapping[Pair, float] = field(default_factory=dict)
    energy_capacity: Mapping[str, float] = field(default_factory=dict)
    #: What each pair's SI cost was **priced** at in objective value units (empty when no
    #: :class:`~astro_mine.allocate.constraints.CostPolicy` is declared) — the per-pair charge the
    #: objective's ``cost`` family subtracts, auditable beside the energy/duration it came from.
    pair_costs: Mapping[Pair, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConstraintCompilation:
    """The output of :func:`~astro_mine.allocate.constraints.compile_with_constraints`.

    ``ir`` is the augmented, byte-stable :class:`~astro_mine.allocate.AllocationIR` a solver lowers
    and :func:`~astro_mine.allocate.verify_feasible` re-checks; ``report`` is the explanation beside
    it; ``objective_terms`` mirrors ``ir.objective_terms`` for convenience.
    """

    ir: AllocationIR
    report: ConstraintReport
    objective_terms: tuple[ObjectiveTerm, ...] = ()
