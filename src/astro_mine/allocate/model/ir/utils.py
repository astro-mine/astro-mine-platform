"""Read-only helpers over a compiled :class:`~astro_mine.allocate.AllocationIR` (RM-P1-ALLOC-03).

Small, pure accessors the constraint builders, the solver backends, and the verifier/explainer
share to read the structure :func:`~astro_mine.allocate.compile_request` emitted — the eligible
``(task, asset)`` assignment pairs, and the **window disjunction** behind a task's
``WINDOW_SELECT`` variables. Everything is read back out of the *IR itself*, never out of the
originating request: the IR is the single contract a backend lowers and the shield re-checks, so a
consumer that reasons about anything else is reasoning about a different model.
"""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.allocate.enums import ConstraintKind, ConstraintSense, VariableSemantic
from astro_mine.allocate.model.ir.model import AllocationIR

__all__ = ["WindowChoice", "assignment_pairs", "window_choices"]

#: Absolute floor for the window containment test (matches the IR verifier's epsilon).
_EPS = 1.0e-9


@dataclass(frozen=True, slots=True)
class WindowChoice:
    """One alternative window a task may run in: its selector variable and the window's bounds.

    ``var_id`` is the ``WINDOW_SELECT`` variable that is ``1`` iff the task runs in this window;
    ``start_s``/``end_s`` are the window's closed start bounds (SI seconds).
    """

    var_id: str
    start_s: float
    end_s: float

    def contains(self, start_s: float) -> bool:
        """Whether ``start_s`` falls inside this window (within a float epsilon)."""
        return self.start_s - _EPS <= start_s <= self.end_s + _EPS


def assignment_pairs(ir: AllocationIR) -> dict[str, list[str]]:
    """Map each task id to its eligible asset ids, read from the IR's assignment variables.

    The assignment variables :func:`~astro_mine.allocate.compile_request` emits *are* the eligible
    ``(task, asset)`` pairs (one per eligible pair). Asset ids are returned sorted so a builder
    iterating them is deterministic (allocate.md §8).
    """
    out: dict[str, list[str]] = {}
    for var in ir.variables:
        if var.semantic is VariableSemantic.ASSIGNMENT and var.task_ref and var.asset_ref:
            out.setdefault(var.task_ref, []).append(var.asset_ref)
    for task_id in out:
        out[task_id].sort()
    return out


def window_choices(ir: AllocationIR) -> dict[str, list[WindowChoice]]:
    """Map each task with **plural** windows to its alternative windows, read from the IR.

    Reconstructed from the disjunction the compiler emitted rather than carried redundantly on the
    variables: a ``TIME_WINDOW`` constraint whose terms include ``WINDOW_SELECT`` variables states
    ``start - sum(lo_k * y_k) >= 0`` (no earlier than the *selected* window's start) or
    ``start - sum(hi_k * y_k) <= 0`` (no later than its end), so each selector's coefficient is the
    negated bound. Reading them back out keeps the IR the *single* source of truth for the model —
    the verifier, the explainer, and the ``trivial-stub`` backend all reason about exactly the
    disjunction the solver saw.

    Tasks with one window (a plain start bound, no selector) and unwindowed tasks are absent. Each
    task's choices come back in selector-id order, which the compiler pins to chronological order.
    """
    task_of: dict[str, str] = {
        var.id: var.task_ref
        for var in ir.variables
        if var.semantic is VariableSemantic.WINDOW_SELECT and var.task_ref
    }
    if not task_of:
        return {}

    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for constraint in ir.constraints:
        if constraint.kind is not ConstraintKind.TIME_WINDOW:
            continue
        if constraint.sense is ConstraintSense.GE:
            bounds = lower
        elif constraint.sense is ConstraintSense.LE:
            bounds = upper
        else:
            continue  # the `twin_pick` exactly-one selects a window; it carries no bound
        for term in constraint.terms:
            if term.var_ref in task_of:
                bounds[term.var_ref] = -term.coefficient

    out: dict[str, list[WindowChoice]] = {}
    for var_id in sorted(task_of):
        start_s, end_s = lower.get(var_id), upper.get(var_id)
        if start_s is None or end_s is None:  # pragma: no cover - the compiler emits both bounds
            continue
        out.setdefault(task_of[var_id], []).append(
            WindowChoice(var_id=var_id, start_s=start_s, end_s=end_s)
        )
    return out
