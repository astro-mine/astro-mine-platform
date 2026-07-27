"""``eval`` — validation against the oracle, error budgets, and the promotion gate (RM-P1-SURR-03).

Surrogate.md §3's ``eval`` module: the calibrated-coverage + error-budget predicate that gates a
retrained model's promotion into Sim (§10, §11). :func:`evaluate_promotion` reads the
:class:`~astro_mine.surrogate.report.ErrorReport` a surrogate already carries and returns a
first-class :class:`GateResult`.

Pure Core + Pydantic — importable without the ``[datasets]`` or ``[serve]`` extras; the gate needs
only the calibrated report, not numpy/torch/onnx.
"""

from __future__ import annotations

from astro_mine.surrogate.eval.gate import (
    BudgetVerdict,
    CoverageVerdict,
    GateResult,
    PromotionCriteria,
    evaluate_promotion,
)

__all__ = [
    "BudgetVerdict",
    "CoverageVerdict",
    "GateResult",
    "PromotionCriteria",
    "evaluate_promotion",
]
