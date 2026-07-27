"""Explainability over the solver-neutral IR — *why this plan* / *why no plan* (RM-P1-ALLOC-06).

Principle 9 (allocate.md §2/§10): every plan ships with its objective decomposition, its binding
constraints, the optimality gap, and — on infeasibility — an irreducible infeasible set, so an
operator or reviewer learns why. An operability feature for delay-tolerant supervisory autonomy,
**not** a safety gate — Guard remains the independent runtime re-check.

Every explanation is computed over the solver-neutral
:class:`~astro_mine.allocate.AllocationIR` and returned on the Core-typed
:class:`~astro_mine.allocate.Allocation` — no backend-specific leakage into the contract — and
binding-constraint provenance traces back to its upstream source (Link window / Worlds terrain /
Fleet budget / the request), consumed via Core contracts, never a sibling import (``LUNAR-UX-004``).

- :func:`decompose_objective` — the per-family (roi / info_gain / value) objective breakdown.
- :func:`binding_constraints` — which constraint is tight at the optimum (with
  :func:`plan_variable_values`, the shared plan→IR value mapping).
- :func:`extract_iis` — the CP-SAT-assumptions irreducible infeasible set on infeasibility.
"""

from __future__ import annotations

from astro_mine.allocate.explain.binding import (
    binding_constraints,
    binding_source,
    plan_variable_values,
)
from astro_mine.allocate.explain.iis import extract_iis
from astro_mine.allocate.explain.objective import decompose_objective

__all__ = [
    "binding_constraints",
    "binding_source",
    "decompose_objective",
    "extract_iis",
    "plan_variable_values",
]
