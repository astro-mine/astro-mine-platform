"""The real planner backends behind the Core contracts (RM-P1-MIND-03).

Marker-gated: ``pddl`` needs the ``[pddl]`` extra (unified-planning + the up-fast-downward
engine), ``native`` needs ``[native]`` (OMPL + FCL). Both are deselected in CI — the engines
bundle native binaries we do not install on runners — so these tests are the local/opt-in gate
that the adapters really drive their engines, not a mock. The pure-Python reference backends in
``test_backends.py`` remain the CI-tested default and the golden-trace baseline.

Each test skips (rather than fails) when its extra is absent, so a base checkout can still run
the whole file.
"""

from __future__ import annotations

import math
from random import Random

import pytest

from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.compose import compose
from astro_mine.mind.exec import Executive
from astro_mine.mind.spec.enums import TierRole
from astro_mine.mind.spec.model import ShieldBinding, StackSpec, StackSpecDocument, TierBinding
from astro_mine.mind.tamp.motion.protocol import MotionPlanner
from astro_mine.mind.tamp.motion.reference import Obstacle, ReferenceMotionPlanner
from tests.mind.support.harness import reference_registry
from tests.mind.support.toy_env import ToyProspectingEnv

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_KEEP_OUT = Obstacle(center=(5.0, 0.0), radius=2.0)


def _observations():  # type: ignore[no-untyped-def]
    return ToyProspectingEnv(horizon=4).reset().observations


def _clear_of(path, start, obstacle) -> bool:  # type: ignore[no-untyped-def]
    """Whether every leg of ``start`` → ``path`` stays outside ``obstacle``."""
    from astro_mine.mind.tamp.motion.reference import _segment_clear_of

    return all(
        _segment_clear_of(a, b, obstacle) for a, b in zip([start, *path], path, strict=False)
    )


# --- the reference planner satisfies the contract it declares -------------------------


def test_reference_motion_planner_satisfies_the_protocol() -> None:
    # The seam both backends fill (checked in CI, unmarked): if this breaks, the OMPL adapter
    # is no longer swappable for the reference one.
    assert isinstance(ReferenceMotionPlanner(), MotionPlanner)


# --- unified-planning mission backend ([pddl]) ----------------------------------------


@pytest.mark.pddl
def test_unified_planning_solves_the_generated_problem() -> None:
    pytest.importorskip("unified_planning")
    from astro_mine.mind.mission.planner.native import UnifiedPlanningMissionPlanner

    planner = UnifiedPlanningMissionPlanner(engine_name="fast-downward")
    observations = _observations()
    assignment = planner.solve(observations)

    # The ENGINE derived the decomposition from the goal — every agent bound to a distinct
    # region, covering all of them (not an assignment Mind handed it).
    assert set(assignment) == set(observations)
    assert sorted(assignment.values()) == list(range(len(observations)))


@pytest.mark.pddl
def test_unified_planning_emits_prospect_tasks() -> None:
    pytest.importorskip("unified_planning")
    from astro_mine.mind.mission.planner.native import UnifiedPlanningMissionPlanner

    batch = UnifiedPlanningMissionPlanner().decide(_observations(), DecisionContext(seed=7))
    assert len(batch.actions) == 2
    for action in batch.actions:
        assert action.task is not None and action.task.prospect is not None
    # regions are the shared geometry the reference planner uses (spacing_m * (index + 1))
    centers = sorted(a.task.prospect.region.center_m.x for a in batch.actions)
    assert centers == [10.0, 20.0]


@pytest.mark.pddl
def test_unified_planning_drops_into_the_mission_tier() -> None:
    pytest.importorskip("unified_planning")

    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="up-stack",
            name="unified-planning mission tier",
            tiers=[
                TierBinding(
                    role=TierRole.MISSION,
                    plugin="mind.mission.up",
                    params={"engine_name": "fast-downward"},
                ),
                TierBinding(role=TierRole.TAMP, plugin="mind.tamp.sampling"),
                TierBinding(role=TierRole.CONTROL, plugin="mind.reference.control"),
            ],
            shield=ShieldBinding(plugin="mind.reference.constraint_shield"),
        ),
    )
    result = Executive(compose(doc, reference_registry(), seed=7)).run(
        ToyProspectingEnv(horizon=5), max_ticks=5, seed=7
    )
    # a real PDDL engine drove the swarm: every agent moved out toward its derived region
    for obs in result.final_observations.values():
        assert obs.self_state.pose.translation_m.x > 0.0


# --- OMPL + FCL motion backend ([native]) ---------------------------------------------


@pytest.mark.native
@pytest.mark.parametrize("algorithm", ["rrtstar", "prmstar", "bitstar"])
def test_ompl_routes_around_a_keep_out(algorithm: str) -> None:
    pytest.importorskip("ompl")
    pytest.importorskip("fcl")
    from astro_mine.mind.tamp.motion.native import OmplMotionPlanner

    planner = OmplMotionPlanner(planner=algorithm, solve_time_s=1.0, agent_radius_m=0.1)
    start, goal = (0.0, 0.0), (10.0, 0.0)
    path = planner.plan(start, goal, (_KEEP_OUT,), Random(7))

    assert path, "OMPL returned no waypoints"
    assert math.isclose(path[-1][0], goal[0], abs_tol=1e-6)
    assert math.isclose(path[-1][1], goal[1], abs_tol=1e-6)
    # the straight line start->goal passes through the keep-out, so a real detour was planned
    assert not _clear_of([goal], start, _KEEP_OUT)
    assert _clear_of(path, start, _KEEP_OUT), "OMPL path clipped the keep-out"


@pytest.mark.native
def test_ompl_satisfies_the_motion_planner_contract() -> None:
    pytest.importorskip("ompl")
    from astro_mine.mind.tamp.motion.native import OmplMotionPlanner

    # structurally interchangeable with the reference RRT (the whole point of the seam)
    assert isinstance(OmplMotionPlanner(), MotionPlanner)


@pytest.mark.native
def test_ompl_rejects_an_unknown_algorithm() -> None:
    pytest.importorskip("ompl")
    from astro_mine.mind.tamp.motion.native import OmplMotionPlanner

    with pytest.raises(ValueError, match="unknown OMPL planner"):
        OmplMotionPlanner(planner="a-star-ish")


@pytest.mark.native
def test_ompl_drops_into_the_tamp_tier() -> None:
    pytest.importorskip("ompl")
    pytest.importorskip("fcl")

    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="ompl-stack",
            name="OMPL motion behind the TAMP tier",
            tiers=[
                TierBinding(role=TierRole.MISSION, plugin="mind.mission.pddl"),
                TierBinding(
                    role=TierRole.TAMP,
                    plugin="mind.tamp.ompl",
                    params={"planner": "rrtstar", "solve_time_s": 0.5},
                ),
                TierBinding(role=TierRole.CONTROL, plugin="mind.reference.control"),
            ],
            shield=ShieldBinding(plugin="mind.reference.constraint_shield"),
        ),
    )
    result = Executive(compose(doc, reference_registry(), seed=7)).run(
        ToyProspectingEnv(horizon=5), max_ticks=5, seed=7
    )
    for obs in result.final_observations.values():
        assert obs.self_state.pose.translation_m.x > 0.0
    # every action still left through the shield, clamped onto the ceiling
    speeds = [
        math.hypot(*a.actuator.setpoint)
        for t in result.trace.ticks
        for a in t.action_batch.actions
        if a.actuator
    ]
    assert speeds and all(s <= 1.5 + 1e-9 for s in speeds)
