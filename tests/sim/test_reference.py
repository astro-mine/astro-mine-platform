"""The offline reference environment — the always-works quickstart tier (#75).

These tests pin the properties a consumer's quickstart depends on: the environment builds with no
content, store, registry or network; it honours the Core Environment contract; **actions change the
trajectory** (the property that makes it trainable rather than a fixed replay); it reproduces at a
fixed seed; and the packaged scenario and the SADF that describes it agree on their agent ids.
"""

from __future__ import annotations

from time import perf_counter

from astro_mine.core.env.conformance import check_environment
from astro_mine.core.messages.enums import ActionKind, ControlMode, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    GotoTask,
    Quat,
    TaskDirective,
    Transform,
    Vec3,
)
from astro_mine.sim.reference import (
    load_reference_scenario,
    make_reference_env,
    make_reference_env_and_assets,
    reference_assets,
)


def _drive(agent_id: str, setpoint: list[float]) -> ActionBatch:
    return ActionBatch(
        actions=[
            Action(
                agent_id=agent_id,
                kind=ActionKind.ACTUATOR,
                actuator=ActuatorCommand(
                    target="base", control_mode=ControlMode.VELOCITY, setpoint=setpoint
                ),
            )
        ]
    )


def _goto(agent_id: str, target: tuple[float, float, float]) -> ActionBatch:
    """What an RL adapter's mobility modality encodes to — a ``TASK``, not a velocity setpoint."""
    x, y, z = target
    return ActionBatch(
        actions=[
            Action(
                agent_id=agent_id,
                kind=ActionKind.TASK,
                task=TaskDirective(
                    task_kind=TaskKind.GOTO,
                    goto=GotoTask(
                        target_frame="MOON_ME",
                        target_pose=Transform(
                            translation_m=Vec3(x=x, y=y, z=z),
                            rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                        ),
                    ),
                ),
            )
        ]
    )


def _positions(env: object, batches: list[ActionBatch]) -> list[tuple[float, float, float]]:
    """Step ``env`` through ``batches`` and return the rover's per-tick position."""
    trail: list[tuple[float, float, float]] = []
    for batch in batches:
        result = env.step(batch)  # type: ignore[attr-defined]
        pose = result.observations["rover"].self_state.pose.translation_m
        trail.append((pose.x, pose.y, pose.z))
    return trail


def test_the_reference_env_builds_offline_with_no_store_or_registry() -> None:
    """AC1 — the five-minute path: import, construct, reset. No content, no network."""
    env = make_reference_env()
    result = env.reset()
    assert set(result.observations) == {"rover", "excavator", "relay"}


def test_the_reference_env_honours_the_core_environment_contract() -> None:
    """AC2 — conformance, including the contract's own full-trace determinism check."""
    check_environment(make_reference_env())


def test_actions_change_the_trajectory() -> None:
    """AC3 — the policy can affect the outcome.

    Learn's only environment is a test fake whose *"dynamics are a pure function of the tick"*, so
    a policy cannot influence the return and it is unusable for training. This asserts the
    reference environment does not repeat that: two different action sequences from the same seed
    must diverge observably.
    """
    east, north = make_reference_env(), make_reference_env()
    east.reset()
    north.reset()

    east_trail = _positions(east, [_drive("rover", [1.0, 0.0, 0.0])] * 4)
    north_trail = _positions(north, [_drive("rover", [0.0, 1.0, 0.0])] * 4)

    assert east_trail != north_trail, "the rover ignored its velocity setpoint"
    assert east_trail[-1][0] > north_trail[-1][0]
    assert north_trail[-1][1] > east_trail[-1][1]


def test_a_goto_task_moves_the_agent() -> None:
    """The half that was missing, and the reason the first cut of this module was untrainable.

    An RL adapter encodes its mobility modality as ``ActionKind.TASK`` with a ``GotoTask`` and
    never emits ``VELOCITY``. Sim's default kinematic engine honours only ``MODE`` and
    ``VELOCITY``, so on it such a policy moves nothing, every pose-derived reward is flat with
    respect to the policy, and training runs while learning nothing. Asserting divergence under
    ``VELOCITY`` alone does **not** catch that — it exercises an action kind the consumer never
    sends. This pins the kind it does.
    """
    env = make_reference_env()
    env.reset()
    trail = _positions(env, [_goto("rover", (40.0, 0.0, 0.0))] * 10)
    assert trail[-1][0] > 1.0, f"the rover ignored its goto task: {trail[-1]}"
    assert trail[-1][0] > trail[0][0], "the rover did not make progress toward the target"


def test_a_goto_task_steers_toward_the_target_not_merely_away_from_rest() -> None:
    """Two different targets must produce two different trajectories — the trainable property."""
    east, north = make_reference_env(), make_reference_env()
    east.reset()
    north.reset()
    east_trail = _positions(east, [_goto("rover", (40.0, 0.0, 0.0))] * 10)
    north_trail = _positions(north, [_goto("rover", (0.0, 40.0, 0.0))] * 10)
    assert east_trail[-1][0] > north_trail[-1][0]
    assert north_trail[-1][1] > east_trail[-1][1]


def test_the_relay_advertises_no_control_it_cannot_execute() -> None:
    """It declares comms and no mobility, so no consumer derives a `goto` it cannot perform."""
    relay = reference_assets()["relay"]
    mobility = {tag for tag in relay.capabilities if tag.value.startswith("mobility.")}
    assert not mobility, f"the relay has no mobility dynamics but advertises {mobility}"


def test_an_unactuated_rollout_is_not_mistaken_for_a_divergent_one() -> None:
    """The negative half of AC3: identical action sequences must *not* diverge."""
    a, b = make_reference_env(), make_reference_env()
    a.reset()
    b.reset()
    drive = [_drive("rover", [1.0, 0.0, 0.0])] * 4
    assert _positions(a, drive) == _positions(b, drive)


def test_the_same_seed_reproduces_the_same_trace() -> None:
    """AC6 — determinism (CX-REPRO)."""
    a, b = make_reference_env(), make_reference_env()
    first, second = a.reset(), b.reset()
    assert first.observations.keys() == second.observations.keys()
    empty = [ActionBatch(actions=[])] * 6
    assert _positions(a, empty) == _positions(b, empty)


def test_rewards_stay_the_consumers_job() -> None:
    """Sim renders physics, not training signal — reward shaping belongs to the consumer."""
    env = make_reference_env()
    env.reset()
    assert not env.step(ActionBatch(actions=[])).rewards


def test_every_scenario_agent_has_a_matching_sadf_asset() -> None:
    """The two halves are coupled by convention; nothing but this catches a drift."""
    scenario = load_reference_scenario()
    assert {a.agent_id for a in scenario.agents} == set(reference_assets())


def test_the_pair_factory_hands_over_both_halves() -> None:
    """The single symbol a consumer's ``module:attr`` env-factory seam resolves."""
    env, assets = make_reference_env_and_assets()
    assert set(env.possible_agents) == set(assets)


def test_the_assets_give_the_three_agents_different_capability_sets() -> None:
    """Heterogeneity is by declared capability — it is what yields per-agent action spaces."""
    assets = reference_assets()
    caps = {agent: set(asset.capabilities) for agent, asset in assets.items()}
    assert caps["rover"] != caps["excavator"] != caps["relay"]
    assert assets["relay"].comms, "the relay declares the only radio"
    assert not any(asset.sensors for asset in assets.values()), (
        "a declared sensor Sim cannot fill would render invalid on every tick"
    )


def test_construction_is_fast_enough_for_a_quickstart() -> None:
    """AC4 — sub-second construction, so a quickstart feels immediate."""
    start = perf_counter()
    make_reference_env().reset()
    assert perf_counter() - start < 1.0
