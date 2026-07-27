"""IR → backend-encoding compilers (allocate.md §3, ``model/compile/``).

The lowerings that turn the solver-neutral :class:`~astro_mine.allocate.AllocationIR` into a
concrete solver encoding. RM-P1-ALLOC-02 ships the **CP-SAT** lowering
(:mod:`astro_mine.allocate.model.compile.cpsat`); the MILP (Pyomo) / auction / metaheuristic
encodings are later issues. Each lowering is a **pure function of the IR** — decoupled from
the search driver in :mod:`astro_mine.allocate.solvers` — so the encoding is independently
testable and a plan the driver returns is always re-checkable against the same IR by
:func:`~astro_mine.allocate.verify_feasible`.
"""

from __future__ import annotations

from astro_mine.allocate.model.compile.cpsat import (
    OBJECTIVE_SCALE,
    TIME_SCALE,
    CpSatModel,
    cumulative_constraint_id,
    lower_to_cpsat,
    no_overlap_constraint_id,
)

__all__ = [
    "OBJECTIVE_SCALE",
    "TIME_SCALE",
    "CpSatModel",
    "cumulative_constraint_id",
    "lower_to_cpsat",
    "no_overlap_constraint_id",
]
