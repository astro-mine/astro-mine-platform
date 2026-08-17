# SPDX-License-Identifier: Apache-2.0
"""Closed vocabularies for the allocation contracts (RM-P1-ALLOC-01).

Small, append-only ``StrEnum``\\ s — the platform idiom for a closed vocabulary
(:class:`astro_mine.core.registry.PluginKind`, the SADF enums). They grow only by adding
a member; members are never removed or repurposed, so the ``string`` wire form
(``allocation_ir.proto``) and every persisted :class:`~astro_mine.allocate.AllocationIR`
stay valid across versions (conventions.md §3).

The Core task and capability vocabularies are **reused, not redefined**: a
:class:`~astro_mine.allocate.Task` carries a Core
:class:`~astro_mine.core.messages.enums.TaskKind` and an
:class:`~astro_mine.allocate.AssetRef` carries Core
:class:`~astro_mine.core.sadf.CapabilityTag`\\ s, so a task/asset means exactly the same
thing whether Sim, Fleet, or Allocate names it. Only the *solver-neutral IR* vocabulary
— decision-variable / constraint / objective kinds and senses — is owned here.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "AllocationStatus",
    "ConstraintKind",
    "ConstraintSense",
    "ObjectiveSense",
    "VariableKind",
    "VariableSemantic",
]


class AllocationStatus(StrEnum):
    """Feasibility/optimality verdict of an :class:`~astro_mine.allocate.Allocation`.

    ``OPTIMAL``/``FEASIBLE`` carry a plan; ``INFEASIBLE`` carries no plan (and, from
    RM-P1-ALLOC-06, an infeasibility certificate). ``UNKNOWN``/``TIMEOUT`` are anytime
    outcomes that MAY carry a best-found incumbent plan but assert no optimality — a late
    optimal answer is a wrong answer in operations (allocate.md §2, principle 1).
    """

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


class VariableKind(StrEnum):
    """The domain type of an IR :class:`~astro_mine.allocate.DecisionVariable`.

    ``INTERVAL`` reserves the CP-SAT interval-variable idiom (start/size/end) for the
    scheduling encodings that land with the solver backends (RM-P1-ALLOC-02); the
    RM-P1-ALLOC-01 structural skeleton emits only ``BINARY`` assignment variables and
    ``CONTINUOUS`` start-time variables.
    """

    BINARY = "binary"
    INTEGER = "integer"
    CONTINUOUS = "continuous"
    INTERVAL = "interval"


class VariableSemantic(StrEnum):
    """What a decision variable *means* in allocation terms — the seam that lets a plan be
    mapped back onto IR variable values for verification (:mod:`astro_mine.allocate` verify).

    Solver-neutral but allocation-domain-typed: ``ASSIGNMENT`` is a binary "asset does
    task" indicator, ``START_TIME`` a task's continuous start epoch, and ``WINDOW_SELECT`` a
    binary "this task runs in *this* one of its disjoint availability windows" indicator — the
    selector that turns a task's plural ``time_windows`` into an exact **disjunction** rather
    than a single min/max envelope that would admit a start inside an availability gap
    (RM-P1-ALLOC-02). Physics-derived semantics (power/energy/comms-window variables) arrive with
    the constraint builders (RM-P1-ALLOC-03).
    """

    ASSIGNMENT = "assignment"
    START_TIME = "start_time"
    WINDOW_SELECT = "window_select"


class ConstraintKind(StrEnum):
    """The kind of an IR :class:`~astro_mine.allocate.Constraint` (allocate.md §3).

    ``ASSIGNMENT_COVER``/``PRECEDENCE``/``TIME_WINDOW`` are the structural families the
    RM-P1-ALLOC-01 compiler emits, and ``LINEAR`` the generic power/comms/terrain-budget family
    the RM-P1-ALLOC-03 builders add. ``NO_OVERLAP``/``CUMULATIVE`` are the **scheduling** families
    the compiler emits per capacity-bearing resource (one asset does one task at a time): unlike
    every other kind they are *not* the linear form ``sum(coefficient * variable) <sense> rhs`` —
    each term names a task's start variable and carries its **interval size** as the coefficient,
    and the family's semantics live in :mod:`astro_mine.allocate.model.ir.schedule` (the one place
    the CP-SAT lowering, the independent verifier, and the binding-constraint explanation all read
    them from, so they can never disagree). Every other kind is carried structurally (terms + sense
    + rhs) so any backend can lower it without knowing its physics.
    """

    ASSIGNMENT_COVER = "assignment_cover"
    PRECEDENCE = "precedence"
    TIME_WINDOW = "time_window"
    NO_OVERLAP = "no_overlap"
    CUMULATIVE = "cumulative"
    LINEAR = "linear"


class ConstraintSense(StrEnum):
    """The relational sense of a linear IR constraint (``lhs`` <sense> ``rhs``)."""

    LE = "le"  # <=
    EQ = "eq"  # ==
    GE = "ge"  # >=


class ObjectiveSense(StrEnum):
    """Whether the objective is minimized or maximized (allocate.md §3)."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
