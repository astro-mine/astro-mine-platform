"""Property test: constrained plans are always feasible against the model (RM-P1-ALLOC-03).

The acceptance invariant (allocate.md §10, "returned plans are always feasible against the model"):
for an arbitrary request compiled with terrain/comms/power constraints, the constraint-aware solve
returns either

* a ``FEASIBLE`` plan that **re-checks** against the augmented IR with the independent verifier
  (:func:`~astro_mine.allocate.verify_feasible` — the same oracle Guard/Bench use), or
* an ``INFEASIBLE`` result carrying no plan and an explicit certificate.

Never a plan that violates a window / power / precedence / keep-out constraint in the model it
claims to solve.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from astro_mine.allocate import (
    AllocationPlanner,
    AllocationRequest,
    AllocationStatus,
    AssetRef,
    ConstraintConfig,
    ConstraintContext,
    Objective,
    Task,
    TimeWindow,
    ValueEstimate,
    compile_with_constraints,
    verify_feasible,
)
from astro_mine.allocate.constraints import CostTable
from astro_mine.allocate.constraints.config import CommsPolicy, PowerPolicy
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.messages.model import Vec3, Volume
from astro_mine.core.sadf import CapabilityTag
from astro_mine.core.units import J2000_EPOCH
from tests.allocate import constraint_factories as F

_CAPS = [
    CapabilityTag.PROSPECTING_NEUTRON,
    CapabilityTag.EXCAVATION_BUCKET,
    CapabilityTag.MOBILITY_WHEELED,
    CapabilityTag.MOBILITY_TRACKED,
]


def _volume() -> Volume:
    return Volume(
        frame="MOON_ME",
        center_m=Vec3(x=10.0, y=-5.0, z=0.0),
        dimensions_m=Vec3(x=1.0, y=1.0, z=1.0),
    )


@st.composite
def _requests(draw: st.DrawFn) -> AllocationRequest:
    n_tasks = draw(st.integers(min_value=1, max_value=4))
    n_assets = draw(st.integers(min_value=1, max_value=3))

    tasks = []
    for i in range(n_tasks):
        caps = draw(st.lists(st.sampled_from(_CAPS), max_size=2, unique=True))
        w0 = draw(st.floats(min_value=0.0, max_value=3600.0))
        wlen = draw(st.floats(min_value=600.0, max_value=12000.0))
        precedence = draw(
            st.lists(
                st.sampled_from([f"t{j}" for j in range(i)]) if i else st.nothing(), unique=True
            )
        )
        tasks.append(
            Task(
                task_id=f"t{i}",
                kind=TaskKind.PROSPECT,
                location=_volume() if draw(st.booleans()) else None,
                resource_target_ref="sha256:" + "cd" * 32 if draw(st.booleans()) else None,
                required_capabilities=caps,
                time_windows=[TimeWindow(start_s=w0, end_s=w0 + wlen)],
                precedence=precedence,
                value=ValueEstimate(mean=draw(st.floats(min_value=1.0, max_value=100.0))),
            )
        )

    assets = []
    for j in range(n_assets):
        caps = draw(st.lists(st.sampled_from(_CAPS), min_size=1, max_size=4, unique=True))
        assets.append(
            AssetRef(
                asset_id=f"a{j}",
                capability_tags=caps,
                budgets={"energy_j": draw(st.floats(min_value=1.0e5, max_value=2.0e7))},
            )
        )

    return AllocationRequest(request_id="prop", tasks=tasks, assets=assets, objective=Objective())


@st.composite
def _contexts_and_costs(
    draw: st.DrawFn, request: AllocationRequest
) -> tuple[ConstraintContext, CostTable, ConstraintConfig]:
    asset_ids = [a.asset_id for a in request.assets]
    task_ids = [t.task_id for t in request.tasks]

    costs = {
        (tid, aid): (
            draw(st.floats(min_value=0.0, max_value=1200.0)),
            draw(st.floats(min_value=0.0, max_value=5.0e6)),
        )
        for tid in task_ids
        for aid in asset_ids
    }
    windows = {
        aid: (
            draw(st.floats(min_value=0.0, max_value=4000.0)),
            draw(st.floats(min_value=4000.0, max_value=16000.0)),
        )
        for aid in asset_ids
    }
    gated = draw(st.lists(st.sampled_from(task_ids), unique=True))
    config = ConstraintConfig(
        comms=CommsPolicy(relay_required_task_ids=frozenset(gated), epoch0=J2000_EPOCH),
        power=PowerPolicy(horizon_s=draw(st.floats(min_value=0.0, max_value=1000.0))),
    )
    ctx = F.context(
        world=F.FakeWorld(slope_deg=draw(st.floats(min_value=0.0, max_value=45.0)))
        if draw(st.booleans())
        else None,
        contacts=F.contact_plan(windows) if gated else None,
        resource=F.FakeField() if draw(st.booleans()) else None,
    )
    return ctx, F.cost_table(costs), config


@given(data=st.data())
def test_constrained_plan_is_feasible_or_an_explicit_certificate(data: st.DataObject) -> None:
    request = data.draw(_requests())
    ctx, costs, config = data.draw(_contexts_and_costs(request))

    alloc = AllocationPlanner().solve(request, context=ctx, config=config, costs=costs)

    if alloc.status is AllocationStatus.FEASIBLE:
        assert alloc.plan is not None
        # The returned plan verifies against the model it solves (load-bearing property).
        comp = compile_with_constraints(request, ctx, config=config, costs=costs)
        assert verify_feasible(alloc, comp.ir)
    else:
        assert alloc.status is AllocationStatus.INFEASIBLE
        assert alloc.plan is None
        assert alloc.infeasibility_certificate is not None
