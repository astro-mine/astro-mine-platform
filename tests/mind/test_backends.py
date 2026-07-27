"""Pluggable mission / TAMP / control backends (RM-P1-MIND-03)."""

from __future__ import annotations

import math
from random import Random

import pytest

from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.compose import compose
from astro_mine.mind.control.reference import MpcController, PidController
from astro_mine.mind.exec import Executive
from astro_mine.mind.mission.planner.pddl import PddlProblem, prospecting_domain, render_problem
from astro_mine.mind.mission.planner.reference import ReferenceMissionPlanner
from astro_mine.mind.spec.enums import TierRole
from astro_mine.mind.spec.model import ShieldBinding, StackSpec, StackSpecDocument, TierBinding
from astro_mine.mind.tamp.motion.reference import Obstacle, ReferenceMotionPlanner
from astro_mine.mind.tamp.reference import ReferenceTampPlanner
from astro_mine.mind.tamp.task.reference import ReferenceTaskPlanner
from tests.mind.support.harness import (
    assert_deterministic_trace,
    policy_plugin,
    reference_registry,
    run_stack,
)
from tests.mind.support.toy_env import ToyProspectingEnv


def _observations():  # type: ignore[no-untyped-def]
    return ToyProspectingEnv(horizon=4).reset().observations


# --- mission planner (PDDL) -----------------------------------------------------------


def test_pddl_problem_is_canonical() -> None:
    problem = PddlProblem(name="p", agents=("b", "a"), regions=("r1", "r0"))
    text = render_problem(problem)
    assert render_problem(problem) == text  # deterministic
    assert "(:objects a b - agent r0 r1 - region)" in text
    assert "(and (prospected r0) (prospected r1))" in text
    assert ":action prospect" in prospecting_domain()


def test_pddl_problem_is_solvable_not_a_rubber_stamp() -> None:
    # The generated problem must be reachable from its own init, or a real engine ([pddl]) gets
    # an unsolvable goal: `prospect` needs an `assigned` fact, which only `assign` establishes,
    # and `assign` needs `free`/`unassigned` in the init. The engine derives the decomposition.
    problem = PddlProblem(name="p", agents=("a", "b"), regions=("r0", "r1"))
    text = render_problem(problem)
    assert "(:init (free a) (free b) (unassigned r0) (unassigned r1))" in text
    assert ":action assign" in prospecting_domain()


def test_reference_mission_generates_problem_and_assigns() -> None:
    planner = ReferenceMissionPlanner(spacing_m=5.0)
    obs = _observations()
    problem = planner.problem(obs)
    assert problem.agents == ("rover-0", "rover-1") and problem.regions == ("r0", "r1")
    assert "lunar-prospecting" in planner.pddl(obs)
    batch = planner.decide(obs, DecisionContext())
    assert [a.task.prospect.region.center_m.x for a in batch.actions] == [5.0, 10.0]


# --- TAMP (task + sampling motion) ----------------------------------------------------


def test_task_planner_reads_mission_targets() -> None:
    mission = ReferenceMissionPlanner().decide(_observations(), DecisionContext())
    targets = ReferenceTaskPlanner().targets(_observations(), mission)
    assert {a: t.x for a, t in targets.items()} == {"rover-0": 10.0, "rover-1": 20.0}


def test_task_planner_holds_when_unassigned() -> None:
    from astro_mine.core.messages.model import ActionBatch

    targets = ReferenceTaskPlanner().targets(_observations(), ActionBatch())
    assert all(t.x == 0.0 for t in targets.values())  # hold at current pose


def test_motion_planner_straight_line_when_clear() -> None:
    path = ReferenceMotionPlanner().plan((0.0, 0.0), (5.0, 0.0), (), Random(0))
    assert path == [(5.0, 0.0)]


def test_motion_planner_routes_around_obstacle() -> None:
    obstacle = Obstacle(center=(5.0, 0.0), radius=2.0)
    planner = ReferenceMotionPlanner(step_m=1.5, max_samples=2000)
    path = planner.plan((0.0, 0.0), (10.0, 0.0), (obstacle,), Random(7))
    assert path and path[-1] == (10.0, 0.0)
    # every leg stays clear of the keep-out
    prev = (0.0, 0.0)
    for point in path:
        mid = ((prev[0] + point[0]) / 2, (prev[1] + point[1]) / 2)
        assert math.hypot(mid[0] - 5.0, mid[1]) >= 0.0  # path exists; detailed clearance below
        prev = point
    assert all(
        _segment_clear((a[0], a[1]), (b[0], b[1]), obstacle)
        for a, b in zip([(0.0, 0.0), *path], path, strict=False)
    )


def test_motion_planner_is_seed_deterministic() -> None:
    obstacle = Obstacle(center=(5.0, 0.0), radius=2.0)
    args = ((0.0, 0.0), (10.0, 0.0), (obstacle,))
    a = ReferenceMotionPlanner().plan(*args, Random(3))
    b = ReferenceMotionPlanner().plan(*args, Random(3))
    assert a == b


def test_tamp_emits_goto_toward_region() -> None:
    mission = ReferenceMissionPlanner().decide(_observations(), DecisionContext())
    batch = ReferenceTampPlanner().decide(
        _observations(), DecisionContext(upstream=mission, seed=7)
    )
    assert all(a.task is not None and a.task.goto is not None for a in batch.actions)
    assert [a.task.goto.target_pose.translation_m.x for a in batch.actions] == [10.0, 20.0]


# --- controllers ----------------------------------------------------------------------


@pytest.mark.parametrize("controller", [PidController(kp=0.5), MpcController(horizon_s=2.0)])
def test_controllers_drive_toward_target(controller) -> None:  # type: ignore[no-untyped-def]
    mission = ReferenceMissionPlanner().decide(_observations(), DecisionContext())
    tamp = ReferenceTampPlanner().decide(_observations(), DecisionContext(upstream=mission, seed=7))
    batch = controller.decide(_observations(), DecisionContext(upstream=tamp))
    for action in batch.actions:
        assert action.actuator is not None
        assert action.actuator.setpoint[0] > 0.0  # moving toward +x regions


def test_controllers_clamp_to_speed_limit() -> None:
    mission = ReferenceMissionPlanner().decide(_observations(), DecisionContext())
    tamp = ReferenceTampPlanner().decide(_observations(), DecisionContext(upstream=mission, seed=7))
    batch = MpcController(horizon_s=0.1, max_speed_mps=1.0).decide(
        _observations(), DecisionContext(upstream=tamp)
    )
    assert all(abs(v) <= 1.0 for a in batch.actions for v in a.actuator.setpoint)


# --- backend stack --------------------------------------------------------------------


def test_backend_stack_runs_anchor_scenario() -> None:
    result = run_stack("lunar_prospecting_backends.yaml", horizon=6, max_ticks=6)
    assert result.ticks_run == 6
    for obs in result.final_observations.values():
        assert obs.self_state.pose.translation_m.x > 0.0


def test_backend_stack_is_deterministic() -> None:
    assert_deterministic_trace(
        lambda: run_stack("lunar_prospecting_backends.yaml", horizon=6, max_ticks=6)
    )


# --- ONNX learned controller drop-in --------------------------------------------------


def _identity_onnx_bytes() -> bytes:
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "controller",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model.SerializeToString()  # type: ignore[no-any-return]


def test_onnx_controller_drops_into_the_control_tier() -> None:
    pytest.importorskip("onnxruntime")
    onnx_module = pytest.importorskip("astro_mine.core.policy.onnx")  # noqa: F841 (post-v0.1.0 Core)
    from astro_mine.core.policy.model import ModelRef, PolicyPackage
    from astro_mine.mind.control.policy.onnx_tier import content_digest, onnx_controller

    onnx_bytes = _identity_onnx_bytes()
    package = PolicyPackage(
        name="test.onnx",
        version="0.1.0",
        onnx_model=ModelRef(digest=content_digest(onnx_bytes)),
        core_interfaces={"policy": "0.1.0", "messages": "0.1.0"},
    )
    controller = onnx_controller(package, onnx_bytes, max_speed_mps=2.0)

    registry = reference_registry()
    registry.register(policy_plugin("test.onnx-control", lambda params: controller, tier="control"))
    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="onnx-stack",
            name="onnx",
            tiers=[
                TierBinding(role=TierRole.MISSION, plugin="mind.mission.pddl"),
                TierBinding(role=TierRole.TAMP, plugin="mind.tamp.sampling"),
                TierBinding(role=TierRole.CONTROL, plugin="test.onnx-control"),
            ],
            shield=ShieldBinding(plugin="mind.reference.constraint_shield"),
        ),
    )
    result = Executive(compose(doc, registry, seed=7)).run(
        ToyProspectingEnv(horizon=5), max_ticks=5, seed=7
    )
    # the learned controller drives the agents (identity graph => velocity = position error)
    for obs in result.final_observations.values():
        assert obs.self_state.pose.translation_m.x > 0.0
    # and every learned action is still Guard-wrapped (clamped onto the ceiling)
    speeds = [
        math.hypot(*a.actuator.setpoint)
        for t in result.trace.ticks
        for a in t.action_batch.actions
        if a.actuator
    ]
    assert speeds and all(s <= 1.5 + 1e-9 for s in speeds)


def test_onnx_factory_requires_a_model_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("onnxruntime")
    pytest.importorskip("astro_mine.core.policy.onnx")
    from astro_mine.mind.control.policy.onnx_tier import onnx_control_plugin

    plugin = onnx_control_plugin()
    with pytest.raises(ValueError, match="model_path"):
        plugin.factory({})
    model_file = tmp_path / "m.onnx"
    model_file.write_bytes(_identity_onnx_bytes())
    controller = plugin.factory({"model_path": str(model_file)})
    assert hasattr(controller, "decide")


def _segment_clear(a, b, obstacle) -> bool:  # type: ignore[no-untyped-def]
    from astro_mine.mind.tamp.motion.reference import _segment_clear_of

    return _segment_clear_of(a, b, obstacle)
