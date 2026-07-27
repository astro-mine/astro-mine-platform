"""Anchor lunar-polar-ice ``AllocationRequest`` / IR fixtures for the ALLOC-01 tests.

``anchor_request`` is the flagship scenario (LUNAR): a prospect → excavate → haul chain of
tasks over a heterogeneous rover / excavator / hauler swarm, with capability-tag eligibility,
time windows, and precedence. ``unwindowed_request`` exercises the optional-absence cases
(no time windows, no variance, no location). Together they cover the request→IR→plan path and
the contract's structural generality.
"""

from __future__ import annotations

from typing import Any

from astro_mine.allocate import (
    Allocation,
    AllocationIR,
    AllocationPlanner,
    AllocationRequest,
    AssetRef,
    Objective,
    ObjectiveSense,
    SolveBudget,
    Task,
    TimeWindow,
    ValueEstimate,
    compile_request,
)
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.messages.model import Vec3, Volume
from astro_mine.core.objective.enums import MetricDirection
from astro_mine.core.objective.model import (
    MetricBinding,
    ObjectiveSpec,
    SuccessCriterion,
)
from astro_mine.core.sadf import CapabilityTag


def _ice_region() -> Volume:
    return Volume(
        frame="MOON_ME",
        center_m=Vec3(x=1500.0, y=-2200.0, z=0.0),
        dimensions_m=Vec3(x=50.0, y=50.0, z=2.0),
    )


def anchor_objective() -> ObjectiveSpec:
    """A minimal Core ObjectiveSpec the request optimizes toward (water yield)."""
    return ObjectiveSpec(
        id="lunar-polar-ice",
        name="Maximize water-ice yield",
        success_criteria=[
            SuccessCriterion(
                id="water-yield",
                binding=MetricBinding(
                    metric="water_kg",
                    unit="kg",
                    direction=MetricDirection.HIGHER_BETTER,
                    target=1000.0,
                    tolerance=50.0,
                ),
            )
        ],
    )


def anchor_request(**overrides: Any) -> AllocationRequest:
    """A lunar-polar-ice prospect→excavate→haul request over a heterogeneous swarm."""
    kwargs: dict[str, Any] = dict(
        request_id="lunar-polar-ice-001",
        tasks=[
            Task(
                task_id="prospect-crater-a",
                kind=TaskKind.PROSPECT,
                location=_ice_region(),
                resource_target_ref="sha256:" + "ab" * 32,
                required_capabilities=[CapabilityTag.PROSPECTING_NEUTRON],
                time_windows=[TimeWindow(start_s=0.0, end_s=3600.0)],
                value=ValueEstimate(mean=10.0, variance=2.0),
            ),
            Task(
                task_id="excavate-crater-a",
                kind=TaskKind.EXCAVATE,
                location=_ice_region(),
                required_capabilities=[CapabilityTag.EXCAVATION_BUCKET],
                precedence=["prospect-crater-a"],
                time_windows=[TimeWindow(start_s=1800.0, end_s=9000.0)],
                value=ValueEstimate(mean=40.0, variance=9.0),
            ),
            Task(
                task_id="haul-to-plant",
                kind=TaskKind.HAUL,
                required_capabilities=[CapabilityTag.MOBILITY_WHEELED],
                precedence=["excavate-crater-a"],
                time_windows=[TimeWindow(start_s=3600.0, end_s=18000.0)],
                value=ValueEstimate(mean=15.0),
            ),
        ],
        assets=[
            AssetRef(
                asset_id="prospector-rover-1",
                capability_tags=[
                    CapabilityTag.PROSPECTING_NEUTRON,
                    CapabilityTag.MOBILITY_WHEELED,
                ],
                budgets={"energy_j": 5.0e6, "time_s": 18000.0},
            ),
            AssetRef(
                asset_id="excavator-1",
                capability_tags=[CapabilityTag.EXCAVATION_BUCKET, CapabilityTag.MOBILITY_TRACKED],
                budgets={"energy_j": 1.2e7},
            ),
            AssetRef(
                asset_id="hauler-1",
                capability_tags=[CapabilityTag.MOBILITY_WHEELED, CapabilityTag.RETURN_BULK_HAULER],
                budgets={"energy_j": 8.0e6},
            ),
        ],
        objective=Objective(sense=ObjectiveSense.MAXIMIZE, spec=anchor_objective()),
        budget=SolveBudget(wall_clock_deadline_s=5.0, target_gap=0.01, deterministic=True, seed=7),
    )
    kwargs.update(overrides)
    return AllocationRequest(**kwargs)


def unwindowed_request(**overrides: Any) -> AllocationRequest:
    """A request exercising optional absence: no windows, no precedence, no location/variance."""
    kwargs: dict[str, Any] = dict(
        request_id="unwindowed-001",
        tasks=[
            Task(task_id="survey-1", kind=TaskKind.PROSPECT, value=ValueEstimate(mean=3.0)),
            Task(task_id="survey-2", kind=TaskKind.PROSPECT, value=ValueEstimate(mean=7.0)),
        ],
        assets=[
            AssetRef(asset_id="rover-a"),
            AssetRef(asset_id="rover-b"),
        ],
    )
    kwargs.update(overrides)
    return AllocationRequest(**kwargs)


def infeasible_request() -> AllocationRequest:
    """A request whose task requires a capability no asset declares (unassignable)."""
    return AllocationRequest(
        request_id="infeasible-001",
        tasks=[
            Task(
                task_id="drill-1",
                kind=TaskKind.SAMPLE,
                required_capabilities=[CapabilityTag.SAMPLE_COLLECTION_DRILL],
                value=ValueEstimate(mean=1.0),
            )
        ],
        assets=[AssetRef(asset_id="rover-a", capability_tags=[CapabilityTag.MOBILITY_WHEELED])],
    )


def solved(request: AllocationRequest) -> tuple[AllocationIR, Allocation]:
    """Compile + solve a request, returning the (IR, Allocation) pair for verification."""
    ir = compile_request(request)
    allocation = AllocationPlanner().solve(request)
    return ir, allocation
