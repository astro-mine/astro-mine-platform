"""Astro-Mine-Allocate — combinatorial task allocation and scheduling.

The specialist allocation-and-scheduling engine: given heterogeneous assets, a set of
tasks, and a thicket of coupled physical constraints, it decides **who does what, when, and
where** and returns a time-indexed, feasible-by-construction, Guard-recheckable plan
(allocate.md §1). It owns the **allocation sub-interface** of the Core Policy/Planner API
and a solver-neutral **Allocation IR** that lets solver backends be true plugins.

RM-P1-ALLOC-01 lands the foundation every other ALLOC issue builds on: the
``AllocationRequest → Allocation`` types, the versioned Allocation IR (JSON Schema +
Protobuf wire form) with its compiler and independent feasibility verifier, and the Core
manifest + planner registering Allocate as a Policy/Planner allocation sub-interface. The
CP-SAT backend (RM-P1-ALLOC-02) and the power/comms/terrain constraint builders
(RM-P1-ALLOC-03) land on top. See ``docs/architecture/allocate.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0"

# The RM-P1-ALLOC-05 anytime contract: monotone incumbent-bound streaming, the honest deadline
# status, and the warm-start seam for online re-solve.
from astro_mine.allocate.anytime import (
    BoundTracker,
    finalize_status,
    hints_from,
    stream_incumbents,
)

# The Core interface versions Allocate is built against — advertised here so consumers and
# the contract test cite one source of truth (defined in :mod:`astro_mine.allocate.api._core`).
from astro_mine.allocate.api._core import CORE_INTERFACES

# The foundational contracts (RM-P1-ALLOC-01). ``__version__`` is defined above before these
# imports so :mod:`astro_mine.allocate.api.manifest` / ``planner`` can read it without a cycle.
from astro_mine.allocate.api.manifest import AllocationAttributes, build_allocation_manifest
from astro_mine.allocate.api.model import (
    Allocation,
    AllocationProvenance,
    AllocationRequest,
    AssetRef,
    AssetSchedule,
    BindingConstraint,
    ConstraintContext,
    InfeasibilityCertificate,
    Objective,
    ObjectiveContribution,
    ObjectiveDecomposition,
    ScheduledTask,
    SolveBudget,
    Task,
    TimeWindow,
    ValueEstimate,
)
from astro_mine.allocate.api.planner import (
    CONSTRAINT_CONFIG_KEY,
    CONSTRAINT_CONTEXT_KEY,
    COST_TABLE_KEY,
    REQUEST_KEY,
    AllocationPlanner,
)

# The RM-P1-ALLOC-03 constraint builders (power, comms-window, terrain) and their inputs. The full
# builder set lives under :mod:`astro_mine.allocate.constraints`; the entry point and its declared
# modeling policy / cost-table inputs are re-exported here for ergonomics.
from astro_mine.allocate.constraints import (
    CommsPolicy,
    ConstraintCompilation,
    ConstraintConfig,
    ConstraintReport,
    CostEntry,
    CostObjectiveResult,
    CostPolicy,
    CostTable,
    InfoGainResult,
    PowerPolicy,
    TerrainPolicy,
    compile_with_constraints,
    refine_cost_objective,
    refine_infogain_objective,
)
from astro_mine.allocate.enums import (
    AllocationStatus,
    ConstraintKind,
    ConstraintSense,
    ObjectiveSense,
    VariableKind,
    VariableSemantic,
)

# The RM-P1-ALLOC-06 explainability layer over the solver-neutral IR: objective decomposition,
# binding-constraint reporting, and the CP-SAT-assumptions IIS on infeasibility.
from astro_mine.allocate.explain import (
    binding_constraints,
    decompose_objective,
    extract_iis,
    plan_variable_values,
)
from astro_mine.allocate.model.ir.compile import (
    compile_request,
    earliest_start_in_windows,
    no_overlap_constraints,
    task_windows,
    window_envelope,
)
from astro_mine.allocate.model.ir.model import (
    IR_VERSION,
    AllocationIR,
    Constraint,
    ConstraintTerm,
    DecisionVariable,
    ObjectiveTerm,
)
from astro_mine.allocate.model.ir.schedule import (
    cumulative_slack,
    no_overlap_slack,
    satisfies_scheduling,
    scheduling_slack,
)
from astro_mine.allocate.model.ir.utils import WindowChoice, assignment_pairs, window_choices
from astro_mine.allocate.model.ir.verify import verify_feasible
from astro_mine.allocate.model.ir.wire import ir_from_wire, ir_to_proto, ir_to_wire
from astro_mine.allocate.solvers import (
    CPSAT_BACKEND,
    SOLVER_ENTRY_POINT_GROUP,
    TRIVIAL_STUB_BACKEND,
    Incumbent,
    Solver,
    TrivialStubSolver,
    available_backends,
    backend_provider,
    known_backends,
    resolve_solver,
)

__all__ = [
    "CONSTRAINT_CONFIG_KEY",
    "CONSTRAINT_CONTEXT_KEY",
    "CORE_INTERFACES",
    "COST_TABLE_KEY",
    "CPSAT_BACKEND",
    "IR_VERSION",
    "REQUEST_KEY",
    "SOLVER_ENTRY_POINT_GROUP",
    "TRIVIAL_STUB_BACKEND",
    "Allocation",
    "AllocationAttributes",
    "AllocationIR",
    "AllocationPlanner",
    "AllocationProvenance",
    "AllocationRequest",
    "AllocationStatus",
    "AssetRef",
    "AssetSchedule",
    "BindingConstraint",
    "BoundTracker",
    "CommsPolicy",
    "Constraint",
    "ConstraintCompilation",
    "ConstraintConfig",
    "ConstraintContext",
    "ConstraintKind",
    "ConstraintReport",
    "ConstraintSense",
    "ConstraintTerm",
    "CostEntry",
    "CostObjectiveResult",
    "CostPolicy",
    "CostTable",
    "DecisionVariable",
    "Incumbent",
    "InfeasibilityCertificate",
    "InfoGainResult",
    "Objective",
    "ObjectiveContribution",
    "ObjectiveDecomposition",
    "ObjectiveSense",
    "ObjectiveTerm",
    "PowerPolicy",
    "ScheduledTask",
    "SolveBudget",
    "Solver",
    "Task",
    "TerrainPolicy",
    "TimeWindow",
    "TrivialStubSolver",
    "ValueEstimate",
    "VariableKind",
    "VariableSemantic",
    "WindowChoice",
    "__version__",
    "assignment_pairs",
    "available_backends",
    "backend_provider",
    "binding_constraints",
    "build_allocation_manifest",
    "compile_request",
    "compile_with_constraints",
    "cumulative_slack",
    "decompose_objective",
    "earliest_start_in_windows",
    "extract_iis",
    "finalize_status",
    "hints_from",
    "ir_from_wire",
    "ir_to_proto",
    "ir_to_wire",
    "known_backends",
    "no_overlap_constraints",
    "no_overlap_slack",
    "plan_variable_values",
    "refine_cost_objective",
    "refine_infogain_objective",
    "resolve_solver",
    "satisfies_scheduling",
    "scheduling_slack",
    "stream_incumbents",
    "task_windows",
    "verify_feasible",
    "window_choices",
    "window_envelope",
]
