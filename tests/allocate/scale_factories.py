"""A seeded **scale** instance of the anchor scenario — tens of robots, hundreds of tasks.

The Phase-1 exit criterion is stated in instance sizes, not in toys: "near-optimal (≤ a few % gap)
plans for **tens of robots / hundreds of tasks** over a multi-day horizon within seconds to a few
minutes" (allocate.md §8). Every other fixture in this suite is a 3-task/3-asset anchor; this module
builds the instance that criterion actually names, and :mod:`tests.test_scale_benchmark` measures
against it.

The instance is the anchor scenario *scaled*, not a synthetic random graph: ``sites`` polar craters,
each with a ``prospect → excavate → haul`` precedence chain, over a heterogeneous swarm of
prospector rovers, tracked excavators, and bulk haulers. Every RM-P1-ALLOC-03 constraint family is
live and biting:

- **terrain** — :class:`RidgedWorld` makes every third crater steep, so only the *high-clearance*
  half of the swarm (the assets whose SADF declares the higher ``max_slope_deg``) may work them:
  real keep-outs, not a uniform world that forbids nothing;
- **comms** — the hauls are relay-gated, and a quarter of the swarm has a *late* contact window
  only, so their early hauls are forbidden for want of a downlink;
- **power** — each asset's energy budget admits roughly its fair share of tasks and no more;
- **schedule** — every task has a real duration, so an asset's tasks must serialize
  (``NO_OVERLAP``);
- **windows** — a slice of the tasks carry two **disjoint** windows (a crater lit on two separate
  passes), exercising the disjunction rather than an envelope;
- **objective** — every pair is charged the energy *this* asset spends on *this* task, priced into
  the task's value units (:class:`~astro_mine.allocate.CostPolicy`), so two feasible assignments
  score differently and there is something for the solver to actually optimize (issue #22).

Everything derives from ``seed`` through one :class:`random.Random`, so the instance — and therefore
the benchmark number — is reproducible across machines (allocate.md §8; conventions.md §11).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from astro_mine.allocate import (
    AllocationRequest,
    AssetRef,
    CommsPolicy,
    ConstraintConfig,
    ConstraintContext,
    CostEntry,
    CostPolicy,
    CostTable,
    Objective,
    ObjectiveSense,
    PowerPolicy,
    SolveBudget,
    Task,
    TerrainPolicy,
    TimeWindow,
    ValueEstimate,
)
from astro_mine.core.messages.enums import ContactConfidence, NodeRole, TaskKind
from astro_mine.core.messages.model import (
    ContactInterval,
    ContactNode,
    ContactPlan,
    Vec3,
    Volume,
)
from astro_mine.core.sadf import CapabilityTag
from astro_mine.core.sadf.model import Asset
from astro_mine.core.units import J2000_EPOCH, Epoch
from astro_mine.core.world.model import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
    Vector,
)
from tests.allocate.constraint_factories import MOON_FRAME, FakeField, sadf_asset

__all__ = [
    "CONFLICT_SITE",
    "DEFAULT_ASSETS",
    "DEFAULT_SITES",
    "HORIZON_S",
    "RidgedWorld",
    "ScaleInstance",
    "infeasible_scale_instance",
    "scale_instance",
]

#: The default swarm size — "tens of robots" (allocate.md §8).
DEFAULT_ASSETS = 25

#: Craters, each contributing a prospect → excavate → haul chain: 84 * 3 = 252 tasks
#: ("hundreds of tasks").
DEFAULT_SITES = 84

#: A three-day planning horizon in SI seconds ("a multi-day horizon").
HORIZON_S = 3 * 24 * 3600.0

#: The steep craters only a tracked excavator may enter — every third site.
_STEEP_EVERY = 3
_STEEP_SLOPE_DEG = 22.0
_GENTLE_SLOPE_DEG = 6.0

#: Slope ceilings the swarm declares in SADF (the terrain builder compares Worlds' truth against
#: these): the low-clearance half refuses the steep craters, the high-clearance half takes them.
#: Every capability class has members of both, so a steep crater loses candidates without ever
#: losing *all* of them — the keep-outs bite, but the instance stays feasible.
_LOW_CLEARANCE_MAX_SLOPE_DEG = 15.0
_HIGH_CLEARANCE_MAX_SLOPE_DEG = 30.0

#: Nominal task durations (s) by kind, jittered per (task, asset) into the cost table.
_DURATION_S = {TaskKind.PROSPECT: 900.0, TaskKind.EXCAVATE: 1800.0, TaskKind.HAUL: 1200.0}

#: Mean energy draw (W) used to derive each pair's energy cost from its duration.
_MODE_POWER_W = 220.0

#: Housekeeping draw (W) the power builder reserves over the horizon before any task energy.
_FLOOR_W = 12.0

#: What a joule of an assignment's energy costs, in the abstract value units a task's
#: ``ValueEstimate`` is denominated in (:class:`~astro_mine.allocate.CostPolicy`). It is a *modeling
#: choice*, not a physical constant, and this is the choice it is making: a pair costs ~10-20 units
#: against tasks worth 5-50, so the swarm's ~40% gross margin is real money and picking the wrong
#: asset for a task is a *measurably* worse plan. Price the energy far below this and the cost
#: family shrinks into the noise — the value every plan earns under the exactly-one cover swamps it,
#: the best and worst plans land within a couple of percent of each other, and the optimality-gap
#: gate goes back to being unable to tell them apart (issue #22).
_ENERGY_PRICE_PER_J = 5.0e-5


class RidgedWorld:
    """A :class:`~astro_mine.core.world.protocol.WorldProvider` whose slope varies with position.

    Unlike the uniform ``FakeWorld``, this one has *terrain*: craters on the ridge (every
    ``_STEEP_EVERY``-th site, encoded in the sampled y-coordinate) are steep enough to keep a
    low-clearance rover out, while the rest are gentle. That is what makes the terrain builder emit
    real keep-outs at scale instead of forbidding nothing.
    """

    def __init__(
        self, *, steep_deg: float = _STEEP_SLOPE_DEG, gentle_deg: float = _GENTLE_SLOPE_DEG
    ) -> None:
        self._steep_deg = steep_deg
        self._gentle_deg = gentle_deg

    @property
    def frame(self):  # type: ignore[no-untyped-def]
        return MOON_FRAME

    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        rad = math.radians(self._steep_deg if _is_steep_position(position) else self._gentle_deg)
        return SurfacePoint(
            frame=MOON_FRAME,
            elevation_m=0.0,
            surface_normal=(math.sin(rad), 0.0, math.cos(rad)),
            gravity=(0.0, 0.0, -1.62),
            illumination=Illumination(state=IlluminationState.LIT, solar_flux_w_m2=1361.0),
            temperature_k=100.0,
            regolith=RegolithParams(bearing_capacity_pa=5.0e4),
        )

    def ray_intersect(self, origin: Vector, direction: Vector) -> Vector | None:
        return None

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        return True


def _is_steep_position(position: Vector) -> bool:
    """Whether a sampled position belongs to a ridge crater (encoded in its site index)."""
    return round(position[1] / 100.0) % _STEEP_EVERY == 0


def _site_volume(site: int) -> Volume:
    """The crater's keep-in volume — its y-coordinate carries the site index (see
    :func:`_is_steep_position`)."""
    return Volume(
        frame="MOON_ME",
        center_m=Vec3(x=float(site) * 250.0, y=float(site) * 100.0, z=0.0),
        dimensions_m=Vec3(x=40.0, y=40.0, z=2.0),
    )


@dataclass(frozen=True, slots=True)
class ScaleInstance:
    """A complete, seeded scale problem: the request plus every input a constrained solve needs."""

    request: AllocationRequest
    context: ConstraintContext
    config: ConstraintConfig
    costs: CostTable

    @property
    def assets(self) -> int:
        return len(self.request.assets)

    @property
    def tasks(self) -> int:
        return len(self.request.tasks)


def _asset_class(index: int) -> str:
    """Round-robin the swarm across the three roles the anchor scenario needs."""
    return ("prospector", "excavator", "hauler")[index % 3]


def _asset_ref(index: int, energy_j: float) -> AssetRef:
    kind = _asset_class(index)
    tags = {
        "prospector": [CapabilityTag.PROSPECTING_NEUTRON, CapabilityTag.MOBILITY_WHEELED],
        "excavator": [CapabilityTag.EXCAVATION_BUCKET, CapabilityTag.MOBILITY_TRACKED],
        "hauler": [CapabilityTag.MOBILITY_WHEELED, CapabilityTag.RETURN_BULK_HAULER],
    }[kind]
    return AssetRef(
        asset_id=f"{kind}-{index:02d}", capability_tags=tags, budgets={"energy_j": energy_j}
    )


def _sadf(index: int) -> Asset:
    """The SADF handle whose declared slope ceiling the terrain builder gates keep-outs on.

    Clearance alternates with the index while the capability class rotates every three, so each
    class holds both high- and low-clearance members: a steep crater keeps half the candidates out
    and still leaves the task assignable.
    """
    max_slope = _HIGH_CLEARANCE_MAX_SLOPE_DEG if index % 2 == 0 else _LOW_CLEARANCE_MAX_SLOPE_DEG
    return sadf_asset(
        f"{_asset_class(index)}-{index:02d}",
        max_slope_deg=max_slope,
        floor_w=_FLOOR_W,
        storage_j=1.0e7,
    )


#: Each of a site's three chained tasks gets a window this wide (SI seconds).
_WINDOW_SPAN_S = 6 * 3600.0


def _site_windows(site_open_s: float, site: int) -> dict[TaskKind, list[TimeWindow]]:
    """The site chain's three availability windows, **staggered in precedence order**.

    ``prospect → excavate → haul`` are consecutive ``_WINDOW_SPAN_S`` slices from the site's opening
    epoch, so the chain's precedence edges (a task starts no earlier than its predecessor) are
    satisfiable by construction — windows drawn independently per task would routinely open a haul
    before its own excavate and make the instance vacuously infeasible.

    Every sixth site's **excavate** window is split into two **disjoint** halves — a crater lit on
    two separate passes with a shadow between them. That is the exact shape a ``[min, max]``
    envelope would flatten into one span *across* the gap (RM-P1-ALLOC-02), so the scale instance
    exercises the window disjunction rather than assuming it away.
    """
    prospect = site_open_s
    excavate = site_open_s + _WINDOW_SPAN_S
    haul = site_open_s + 2 * _WINDOW_SPAN_S

    excavate_windows = [TimeWindow(start_s=excavate, end_s=excavate + _WINDOW_SPAN_S)]
    if site % 6 == 0:
        excavate_windows = [
            TimeWindow(start_s=excavate, end_s=excavate + _WINDOW_SPAN_S * 0.25),
            TimeWindow(start_s=excavate + _WINDOW_SPAN_S * 0.75, end_s=excavate + _WINDOW_SPAN_S),
        ]
    return {
        TaskKind.PROSPECT: [TimeWindow(start_s=prospect, end_s=prospect + _WINDOW_SPAN_S)],
        TaskKind.EXCAVATE: excavate_windows,
        TaskKind.HAUL: [TimeWindow(start_s=haul, end_s=haul + _WINDOW_SPAN_S)],
    }


def _contact_plan(assets: list[AssetRef], rng: random.Random) -> ContactPlan:
    """A relay + one contact interval per asset; a quarter of the swarm gets a **late** window only.

    Those assets cannot downlink an early haul, so the comms builder forbids those pairs: the gate
    bites instead of degrading to "any contact window". The stagger is on a different modulus than
    the capability rotation, so it never strips a whole class of its downlink.
    """
    nodes = [ContactNode(id="relay-1", role=NodeRole.GROUND, kind="ground_station")]
    intervals: list[ContactInterval] = []
    for index, asset in enumerate(assets):
        nodes.append(ContactNode(id=asset.asset_id, role=NodeRole.SPACE, kind="surface_agent"))
        start_s = HORIZON_S / 2.0 if index % 4 == 0 else 0.0
        intervals.append(
            ContactInterval(
                node_a=asset.asset_id,
                node_b="relay-1",
                start_tdb_s=start_s + J2000_EPOCH.tdb_seconds,
                end_tdb_s=HORIZON_S + J2000_EPOCH.tdb_seconds,
                max_rate_bps=1.0e6,
                confidence=ContactConfidence.HIGH,
            )
        )
    return ContactPlan(nodes=nodes, intervals=intervals)


def scale_instance(
    *,
    seed: int = 19,
    assets: int = DEFAULT_ASSETS,
    sites: int = DEFAULT_SITES,
    energy_per_task: float = 16.0,
    deadline_s: float = 60.0,
    target_gap: float | None = None,
) -> ScaleInstance:
    """Build the seeded instance: ``assets`` robots over ``sites`` prospect→excavate→haul chains.

    ``energy_per_task`` is how many tasks' worth of energy each asset's budget carries — it sets how
    hard the power constraint bites (the default leaves real slack over the ~10 tasks an asset
    averages, so the instance stays feasible while the budgets still bind on the busiest assets).

    ``target_gap`` defaults to **None** — no ``relative_gap_limit``, so CP-SAT must genuinely
    *prove* its optimum rather than stopping the moment it is provably within a few percent. CP-SAT
    reports ``OPTIMAL`` for **both** outcomes, so a gap limit would quietly turn the benchmark's
    "we proved an optimum" assertion into "we stopped when we were near one". Proving it costs a
    fraction of a second here, so the instance buys the stronger claim.
    """
    rng = random.Random(seed)

    asset_refs = [_asset_ref(i, energy_j=0.0) for i in range(assets)]
    tasks: list[Task] = []
    costs: dict[tuple[str, str], CostEntry] = {}

    for site in range(sites):
        volume = _site_volume(site)
        windows = _site_windows(rng.uniform(0.0, HORIZON_S - 3 * _WINDOW_SPAN_S), site)
        # The haul carries no crater location — it hauls *to the plant* — so it is gated by comms,
        # not terrain (exactly as the anchor request models it).
        chain: list[tuple[str, TaskKind, list[str], list[CapabilityTag], Volume | None]] = [
            (
                f"prospect-{site:03d}",
                TaskKind.PROSPECT,
                [],
                [CapabilityTag.PROSPECTING_NEUTRON],
                volume,
            ),
            (
                f"excavate-{site:03d}",
                TaskKind.EXCAVATE,
                [f"prospect-{site:03d}"],
                [CapabilityTag.EXCAVATION_BUCKET],
                volume,
            ),
            (
                f"haul-{site:03d}",
                TaskKind.HAUL,
                [f"excavate-{site:03d}"],
                [CapabilityTag.MOBILITY_WHEELED],
                None,
            ),
        ]
        for task_id, kind, precedence, required, location in chain:
            nominal = _DURATION_S[kind]
            tasks.append(
                Task(
                    task_id=task_id,
                    kind=kind,
                    location=location,
                    required_capabilities=required,
                    time_windows=windows[kind],
                    precedence=precedence,
                    duration_s=nominal,
                    value=ValueEstimate(mean=rng.uniform(5.0, 50.0), variance=2.0),
                )
            )
            # A measured, asset-specific cost per eligible pair — a tracked excavator and a wheeled
            # rover do not cross the same crater in the same time (allocate.md §6).
            for asset in asset_refs:
                if not set(required) <= set(asset.capability_tags):
                    continue
                duration = nominal * rng.uniform(0.85, 1.15)
                costs[(task_id, asset.asset_id)] = CostEntry(
                    duration_s=duration, energy_j=duration * _MODE_POWER_W
                )

    # Size each asset's energy budget off the mean pair cost it will actually see, *on top of* the
    # housekeeping energy the power builder reserves over the horizon — so `energy_per_task` means
    # what it says (the number of tasks the deliverable energy admits) rather than being silently
    # eaten by the floor reservation.
    mean_energy = sum(e.energy_j or 0.0 for e in costs.values()) / max(len(costs), 1)
    reserved = _FLOOR_W * HORIZON_S
    asset_refs = [
        _asset_ref(i, energy_j=reserved + mean_energy * energy_per_task) for i in range(assets)
    ]

    request = AllocationRequest(
        request_id=f"scale-{assets}x{len(tasks)}-seed{seed}",
        tasks=tasks,
        assets=asset_refs,
        # Value earned *minus what earning it cost* — the per-pair cost family (issue #22).
        # Without it every feasible plan scores identically (the cover is exactly-one and value is
        # per-task), and the optimality gap is zero by construction rather than by quality.
        objective=Objective(sense=ObjectiveSense.MAXIMIZE, weights={"cost": 1.0}),
        budget=SolveBudget(
            wall_clock_deadline_s=deadline_s,
            target_gap=target_gap,
            # A wall-clock deadline is only honored in non-deterministic mode (deterministic mode
            # deliberately trades the clock for byte-reproducibility, cpsat.py `_configure`) — and a
            # wall-clock deadline is exactly what this benchmark measures.
            deterministic=False,
            workers=8,
            seed=seed,
        ),
    )
    context = ConstraintContext(
        world=RidgedWorld(),
        resource=FakeField(mean=0.6, variance=0.02),
        contacts=_contact_plan(asset_refs, rng),
        assets={a.asset_id: _sadf(i) for i, a in enumerate(asset_refs)},
    )
    config = ConstraintConfig(
        terrain=TerrainPolicy(default_max_slope_deg=_LOW_CLEARANCE_MAX_SLOPE_DEG),
        power=PowerPolicy(horizon_s=HORIZON_S, default_mode_power_w=_MODE_POWER_W),
        comms=CommsPolicy(relay_required_kinds=frozenset({TaskKind.HAUL}), epoch0=J2000_EPOCH),
        cost=CostPolicy(energy_price_per_j=_ENERGY_PRICE_PER_J),
    )
    return ScaleInstance(request=request, context=context, config=config, costs=CostTable.of(costs))


#: The site whose chain is deliberately broken to create the localized conflict
#: (:func:`infeasible_scale_instance`).
CONFLICT_SITE = 40


def infeasible_scale_instance(**kwargs: object) -> ScaleInstance:
    """The same scale instance with **one localized, deliberate conflict** — the IIS test case.

    One site's ``excavate`` is pushed *after* its own ``haul``: the chain's precedence edge
    (``excavate → haul``) then cannot hold inside either task's window. Exactly one site is broken,
    so the conflict is small and *localized* — which is the operational question an
    infeasibility certificate has to answer ("why can this one haul not run?"), and the case a
    reported irreducible set must stay small for even when the surrounding instance is large.

    A *globally* infeasible instance (say, every asset starved of energy) is deliberately **not**
    used here: its conflict set is genuinely large — the whole point of an IIS is that it is
    irreducible, not that it is small — and deletion-filtering it costs more than it explains. The
    :data:`~astro_mine.allocate.explain.iis.DEFAULT_MAX_REFINEMENT_SOLVES` cap is what bounds that
    case; this fixture is what proves the *useful* case is fast and tight.
    """
    base = scale_instance(**kwargs)  # type: ignore[arg-type]
    tasks: list[Task] = []
    for task in base.request.tasks:
        if task.task_id == f"excavate-{CONFLICT_SITE:03d}":
            haul_window = next(
                t for t in base.request.tasks if t.task_id == f"haul-{CONFLICT_SITE:03d}"
            ).time_windows[0]
            # Force this excavate to start strictly *after* its own haul's latest start: the
            # precedence edge `start[haul] >= start[excavate]` is then unsatisfiable.
            task = task.model_copy(
                update={
                    "time_windows": [
                        TimeWindow(
                            start_s=haul_window.end_s + _WINDOW_SPAN_S,
                            end_s=haul_window.end_s + 2 * _WINDOW_SPAN_S,
                        )
                    ]
                }
            )
        tasks.append(task)

    return ScaleInstance(
        request=base.request.model_copy(update={"tasks": tasks}),
        context=base.context,
        config=base.config,
        costs=base.costs,
    )
