"""Constraint builders — power, comms-window, terrain (RM-P1-ALLOC-03).

The builders that lift **upstream truth** into the solver-neutral Allocation IR so the model
reflects real physics rather than hardcoded assumptions (allocate.md §3, "constraints from upstream
truth"). Each builder consumes a Core contract carried in the
:class:`~astro_mine.allocate.ConstraintContext` — [Worlds](worlds.md) traversability
(:class:`~astro_mine.core.world.protocol.WorldProvider`), the [Link](link.md) contact graph
(:class:`~astro_mine.core.messages.model.ContactPlan`), Fleet SADF budgets
(:class:`~astro_mine.core.sadf.model.Asset`), and Prospect value
(:class:`~astro_mine.core.resource.protocol.ResourceField`) — plus a cached
:class:`~astro_mine.allocate.constraints.CostTable`; it invents no physics and imports no sibling
package. :func:`compile_with_constraints` composes them into one augmented, byte-stable IR.

The Phase-1 MVP set is **power, comms-window, terrain**; stochastic/robust formulations over
Prospect uncertainty and thermal-horizon constraints are P1-late/P2, and mission-level
Trajectory/Sizing constraint families are P3 (RFC-0001).
"""

from __future__ import annotations

from astro_mine.allocate.constraints.comms import build_comms_constraints
from astro_mine.allocate.constraints.compose import compile_with_constraints
from astro_mine.allocate.constraints.config import (
    CommsPolicy,
    ConstraintConfig,
    CostPolicy,
    PowerPolicy,
    TerrainPolicy,
)
from astro_mine.allocate.constraints.cost_objective import (
    CostObjectiveResult,
    refine_cost_objective,
)
from astro_mine.allocate.constraints.costs import CostEntry, CostTable
from astro_mine.allocate.constraints.infogain import InfoGainResult, refine_infogain_objective
from astro_mine.allocate.constraints.power import build_power_constraints
from astro_mine.allocate.constraints.result import (
    ConstraintCompilation,
    ConstraintFinding,
    ConstraintReport,
)
from astro_mine.allocate.constraints.terrain import build_terrain_constraints
from astro_mine.allocate.constraints.value import refine_value_objective

__all__ = [
    "CommsPolicy",
    "ConstraintCompilation",
    "ConstraintConfig",
    "ConstraintFinding",
    "ConstraintReport",
    "CostEntry",
    "CostObjectiveResult",
    "CostPolicy",
    "CostTable",
    "InfoGainResult",
    "PowerPolicy",
    "TerrainPolicy",
    "build_comms_constraints",
    "build_power_constraints",
    "build_terrain_constraints",
    "compile_with_constraints",
    "refine_cost_objective",
    "refine_infogain_objective",
    "refine_value_objective",
]
