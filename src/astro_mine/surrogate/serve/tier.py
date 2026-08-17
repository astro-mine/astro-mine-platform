# SPDX-License-Identifier: Apache-2.0
"""``ServedTier`` — the surrogate-side reference of the admit / fall-back contract (RM-P1-SURR-04).

Surrogate produces a fidelity tier; **Sim decides** when to use it (surrogate.md §1). This is the
thin seam that surfaces the *two* error channels a scheduler consumes (surrogate.md §11 "both"):
the **static** :class:`~astro_mine.surrogate.report.ErrorReport` admission budget (:meth:`admits`)
and the **live** per-query uncertainty / trust flag (:meth:`should_escalate`), over a wrapped
:class:`~astro_mine.surrogate.model.SurrogateModel`.

It is deliberately minimal and generic — the executable reference of the decision Sim's
multi-fidelity scheduler (RM-P1-SIM-03) makes over the same Core-visible artifacts, used here for
the phase-proof demonstration. It is **not** Sim's scheduler: it neither runs episodes nor tracks
per-tick deviation. Sim re-implements the decision on its side (the narrow waist — no import of this
package); ``ServedTier`` keeps the surrogate's own view honest and testable.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from astro_mine.surrogate.model import Prediction, SurrogateModel, SurrogateState
from astro_mine.surrogate.report import ErrorReport

__all__ = ["ServedTier"]


class ServedTier:
    """A wrapped surrogate exposing static admission + live fall-back decisions to a scheduler."""

    def __init__(self, model: SurrogateModel) -> None:
        self._model = model

    @property
    def error_report(self) -> ErrorReport:
        """The static, calibrated bound a scheduler admits the tier against."""
        return self._model.error_report

    def advance(self, state: SurrogateState, action: SurrogateState | None = None) -> Prediction:
        """One tier step — the surrogate prediction the scheduler substitutes for ground truth."""
        return self._model.predict(state, action)

    def admits(self, task_tolerance: Mapping[str, float]) -> bool:
        """Static admission: is the declared per-channel budget within the task's tolerance?

        Admits only if **every** requested channel has a recommended budget at or below the task's
        tolerance for it (surrogate.md §6). A channel with no declared budget is not admissible —
        the surrogate never claims accuracy it did not measure (principle 4, conservative).
        """
        budget = self._model.error_report.substitution_policy.recommended_error_budget
        return all(
            channel in budget and budget[channel] <= tolerance
            for channel, tolerance in task_tolerance.items()
        )

    def should_escalate(self, prediction: Prediction, max_uncertainty: float) -> bool:
        """Live fall-back: escalate to ground truth on an out-of-domain or over-tolerance query.

        Escalates if the query left the trust region (``in_domain`` is false — an OOD query never
        substitutes, surrogate.md principle 3) or if the prediction's representative calibrated
        uncertainty exceeds ``max_uncertainty``, a single accuracy tolerance for the in-loop
        decision (surrogate.md §11 "live per-query uncertainty for in-loop fallback"). The reduction
        is one field-layout-agnostic scalar; Sim's scheduler maps its per-channel ``FidelityPolicy``
        tolerances onto this on its side (the narrow waist — no import of this package).
        """
        if not prediction.in_domain:
            return True
        return _representative_uncertainty(prediction) > max_uncertainty


def _representative_uncertainty(prediction: Prediction) -> float:
    """A single live-uncertainty scalar comparable to the per-channel budget.

    The **mean** calibrated half-width across scalar channels and per-particle fields — aggregated
    the way the ``ErrorReport`` budget is (an RMSE-scale quantity), not a per-particle worst case,
    so it is comparable to ``recommended_error_budget``. OOD inflation (applied in-graph) still
    lifts it above tolerance, and Sim applies the exact per-channel mapping on its side.
    """
    values = [float(v) for v in prediction.uncertainty.values()]
    for array in prediction.field_uncertainty.values():
        if np.size(array):
            values.append(float(np.mean(array)))
    return max(values) if values else 0.0
