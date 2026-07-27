"""The ``Solver`` strategy contract — ``Incumbent`` + the streaming protocol (RM-P1-ALLOC-02).

The single, backend-neutral seam every solver plugin sits behind (allocate.md §3): a
``solve(model, budget, hints?) -> stream[Incumbent]`` strategy that CP-SAT (RM-P1-ALLOC-02),
a MILP backend, or an auction all realize without changing problem semantics — only the path
to a solution. This is the **hard contract** the anytime stream (RM-P1-ALLOC-05), the
explainability decomposition (RM-P1-ALLOC-06), and the determinism gate (RM-P1-ALLOC-07)
build on, so both types here are deliberately small and stable.

An :class:`Incumbent` is one solution the search has found: its realized ``objective`` (always
recomputed from the float IR objective terms, never a scaled solver internal, so it agrees with
:func:`~astro_mine.allocate.verify_feasible`), the current dual ``bound`` and derived
``gap`` (the anytime optimality certificate), the feasible ``plan`` mapped back onto the
Core-typed schedule, and the solve ``status``. A solver yields incumbents with
**monotonically improving** bounds; the last one carries the terminal status.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from astro_mine.allocate.api.model import AssetSchedule, SolveBudget
from astro_mine.allocate.enums import AllocationStatus
from astro_mine.allocate.model.ir.model import AllocationIR

__all__ = ["Incumbent", "Solver"]


@dataclass(frozen=True, slots=True)
class Incumbent:
    """One solution found during search — a point on the anytime incumbent/bound trajectory.

    ``objective`` is the realized objective value recomputed from the IR objective terms;
    ``bound`` is the best dual bound proven so far (``None`` for a pure feasibility model with no
    objective); ``gap`` is the derived relative optimality gap (``None`` when there is no bound);
    ``plan`` is the feasible per-asset schedule this solution decodes to; ``status`` is the solve
    verdict at the moment this incumbent was recorded; ``is_feasible`` is ``True`` iff ``plan`` is
    a usable feasible plan (``False`` for a no-solution terminal incumbent).
    """

    objective: float
    bound: float | None
    gap: float | None
    plan: list[AssetSchedule]
    status: AllocationStatus
    is_feasible: bool = field(default=True)


@runtime_checkable
class Solver(Protocol):
    """A backend strategy: lower the IR, search within the budget, stream improving incumbents.

    ``solve`` compiles the solver-neutral :class:`~astro_mine.allocate.AllocationIR` to its native
    encoding, searches within the :class:`~astro_mine.allocate.SolveBudget` (deadline, target gap,
    seed, determinism, workers), and yields an :class:`Incumbent` per improving solution — the last
    one carrying the terminal status. ``hints`` is an optional warm start (IR variable id → value)
    the backend seeds and **verifies**, never trusts (allocate.md §9). Backends are constructed
    with whatever presentation context they need to decode a plan (task kinds, durations); the
    strategy signature itself stays IR-only so the registry can swap them freely.
    """

    def solve(
        self,
        ir: AllocationIR,
        budget: SolveBudget,
        *,
        hints: Mapping[str, float] | None = None,
    ) -> Iterator[Incumbent]: ...
