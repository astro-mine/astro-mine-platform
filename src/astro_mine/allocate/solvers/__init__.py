"""Solver backends behind one ``Solver`` strategy (allocate.md §3, ``solvers/``).

The pluggable search backends of Allocate: each lowers the solver-neutral Allocation IR and
searches within a :class:`~astro_mine.allocate.SolveBudget`, streaming
:class:`Incumbent`\\ s behind the single :class:`Solver` strategy so swapping one for another
changes only the *path* to a solution, never the problem semantics (allocate.md §2, principle 3).

RM-P1-ALLOC-02 ships the **CP-SAT** primary engine (:class:`CpSatSolver`) and a no-dependency
greedy (:class:`TrivialStubSolver`) for the local no-solver path; the MILP / auction / metaheuristic
backends are later issues. :func:`resolve_solver` is the Allocate-internal registry
:class:`~astro_mine.allocate.AllocationPlanner` resolves a backend id through. ``CpSatSolver`` is
imported lazily via the registry, so importing this package never requires OR-Tools.
"""

from __future__ import annotations

from astro_mine.allocate.solvers.base import Incumbent, Solver
from astro_mine.allocate.solvers.registry import (
    CPSAT_BACKEND,
    SOLVER_ENTRY_POINT_GROUP,
    TRIVIAL_STUB_BACKEND,
    available_backends,
    backend_provider,
    known_backends,
    resolve_solver,
)
from astro_mine.allocate.solvers.trivial import TrivialStubSolver

__all__ = [
    "CPSAT_BACKEND",
    "SOLVER_ENTRY_POINT_GROUP",
    "TRIVIAL_STUB_BACKEND",
    "Incumbent",
    "Solver",
    "TrivialStubSolver",
    "available_backends",
    "backend_provider",
    "known_backends",
    "resolve_solver",
]
