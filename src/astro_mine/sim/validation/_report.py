"""The oracle-comparison primitive shared by the validation harnesses (RM-P0-SIM-10).

A :class:`OracleReport` is the verdict of comparing a simulated quantity against a reference within
an **explicit error budget** (sim.md §2.9, §10): the credibility backbone for the sim-to-real chasm
is that every engine result is admissible only against a stated tolerance, never "a single guess".
:func:`validate_against_oracle` is the one comparator the orbital and terramechanics harnesses build
on — it reduces two aligned trajectories to a worst-case error and checks it against the budget.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["OracleReport", "validate_against_oracle"]


@dataclass(frozen=True, slots=True)
class OracleReport:
    """The outcome of one oracle comparison: the worst error against a stated budget.

    ``passed`` is ``max_error <= budget``. ``name`` identifies the gate and ``detail`` records how
    the error was measured (sample count, relative vs absolute) — surfaced when a gate fails."""

    name: str
    max_error: float
    budget: float
    detail: str = ""

    @property
    def passed(self) -> bool:
        """Whether the worst error stayed within the error budget."""
        return self.max_error <= self.budget


def validate_against_oracle(
    actual: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    *,
    budget: float,
    name: str = "oracle",
    relative: bool = True,
) -> OracleReport:
    """Compare two aligned trajectories componentwise and report the worst error vs ``budget``.

    ``actual`` and ``reference`` are equal-length sequences of same-shape rows (e.g. position
    vectors at matching sample times). The per-row error is the Euclidean distance between the rows,
    divided by the reference magnitude when ``relative`` (so the budget is a fraction). The report's
    ``max_error`` is the worst row error. Raises ``ValueError`` on a negative budget, a length
    mismatch, or empty input.
    """
    if budget < 0.0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if len(actual) != len(reference):
        raise ValueError(
            f"actual and reference differ in length: {len(actual)} vs {len(reference)}"
        )
    if not reference:
        raise ValueError("need at least one sample to validate against an oracle")
    max_error = 0.0
    for a, r in zip(actual, reference, strict=True):
        error = _distance(a, r)
        if relative:
            magnitude = _magnitude(r)
            error = error / magnitude if magnitude > 0.0 else error
        max_error = max(max_error, error)
    measure = "relative" if relative else "absolute"
    return OracleReport(
        name=name,
        max_error=max_error,
        budget=budget,
        detail=f"{len(reference)} samples, {measure} error",
    )


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    """The Euclidean distance between two same-length component rows."""
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b, strict=True)))


def _magnitude(a: Sequence[float]) -> float:
    """The Euclidean magnitude of a component row."""
    return math.sqrt(sum(x * x for x in a))
