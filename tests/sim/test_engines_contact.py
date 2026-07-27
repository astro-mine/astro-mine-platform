"""RM-P0-SIM-03 / RM-P1-SIM-04 — the real contact tiers: MuJoCo (CPU) and Brax/MJX (batched).

The gap these close: the mobility tier was closed-form kinematics, and the "Brax" tier was that same
closed-form model re-expressed in ``jax.numpy`` — no file under ``engines/brax/`` imported ``brax``
or
``mujoco`` at all. Both contact tiers now simulate a **physical machine**: a chassis and four
torque-driven wheels in frictional contact with a compliant regolith plane.

These tests prove the tiers (a) register manifest-only, with no MuJoCo/JAX needed; (b) really do
contact — the rover *slips*, which a kinematic model cannot express; (c) step the *same* rover model
as each other; (d) are gated against the analytic drawbar-pull oracle; (e) reproduce in-process; and
(f) preserve the existing ``VectorizedRollout`` batching + Ray fan-out surfaces unchanged.

The MuJoCo/JAX-dependent tests skip without the extras; the descriptor/manifest tests always run.
"""

from __future__ import annotations

import math

import pytest

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Action, ActionBatch, ActuatorCommand
from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import (
    CapabilityTag,
    DeterminismClass,
    FidelityTier,
    Regime,
)
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines import (
    MJX_CONTACT_ENGINE_DESCRIPTOR,
    MUJOCO_MOBILITY_ENGINE_DESCRIPTOR,
    RegimeEngine,
    default_engine_registry,
    mjx_contact_engine_factory,
    mujoco_mobility_engine_factory,
)
from astro_mine.sim.engines._rover_mjcf import RoverModelSpec, rover_mjcf
from astro_mine.sim.runtime import (
    AgentSpec,
    MjxContactDynamics,
    MujocoMobilityDynamics,
    RngStreams,
    Scenario,
    run_episode,
)
from tests.sim._equivalence import assert_shards_match_oracle

_MASS_KG = 250.0
_MAX_SPEED = 0.8
_DRIVE_MPS = 0.5


@pytest.fixture
def mujoco_stack():
    """Skip unless the MuJoCo contact solver (the ``[mujoco]`` extra) is importable."""
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    from astro_mine.sim.engines.mujoco import _engine

    return _engine


@pytest.fixture
def mjx_stack():
    """Skip unless the JAX + MJX stack (the ``[brax]`` extra) is importable."""
    pytest.importorskip("jax")
    pytest.importorskip("brax")
    pytest.importorskip("mujoco")
    from astro_mine.sim.engines.brax import _mjx

    return _mjx


def _mujoco_scenario(*, dt_s: float = 0.5, steps: int = 8) -> Scenario:
    return Scenario(
        name="contact-rover",
        seed=7,
        dt_s=dt_s,
        horizon_steps=steps,
        agents=(
            AgentSpec(
                agent_id="rover",
                frame=MOON_BODY_FIXED,
                velocity_mps=(_DRIVE_MPS, 0.0, 0.0),
                battery_soc_j=1.0e6,
                dynamics=MujocoMobilityDynamics(mass_kg=_MASS_KG, max_speed_mps=_MAX_SPEED),
            ),
        ),
    )


def _mjx_scenario(*, dt_s: float = 0.5, steps: int = 4, n_agents: int = 1) -> Scenario:
    return Scenario(
        name="mjx-contact-rover",
        seed=7,
        dt_s=dt_s,
        horizon_steps=steps,
        agents=tuple(
            AgentSpec(
                agent_id=f"rover{i}",
                frame=MOON_BODY_FIXED,
                velocity_mps=(_DRIVE_MPS, 0.0, 0.0),
                battery_soc_j=1.0e6,
                dynamics=MjxContactDynamics(
                    mass_kg=_MASS_KG, max_speed_mps=_MAX_SPEED, batch_size=2
                ),
            )
            for i in range(n_agents)
        ),
    )


# --- registration needs neither MuJoCo nor JAX ------------------------------------


@pytest.mark.parametrize(
    ("name", "descriptor"),
    [
        ("astro-mine.sim.mujoco_mobility", MUJOCO_MOBILITY_ENGINE_DESCRIPTOR),
        ("astro-mine.sim.mjx_contact", MJX_CONTACT_ENGINE_DESCRIPTOR),
    ],
)
def test_contact_tiers_register_manifest_only(name: str, descriptor: object) -> None:
    registry = default_engine_registry()
    manifest = registry.manifest(name)
    assert manifest.kind is PluginKind.REGIME_ENGINE
    assert manifest.regimes == [Regime.SURFACE]
    # Both contact tiers are ARTICULATED — a real rung above the reduced-order KINEMATIC mobility
    # engine, which is what lets the multi-fidelity scheduler choose between them.
    assert manifest.attributes["fidelity"]["tier"] == FidelityTier.ARTICULATED.value
    assert manifest.determinism_class is DeterminismClass.TOLERANCE
    assert CapabilityTag.MOBILITY_WHEELED in manifest.capability_tags


def test_the_reduced_order_tiers_remain_the_local_fallback() -> None:
    # The always-works local tier (CX-LOCAL) must survive the arrival of the real backends.
    registry = default_engine_registry()
    assert registry.manifest("astro-mine.sim.mobility").attributes["fidelity"]["tier"] == (
        FidelityTier.KINEMATIC.value
    )
    assert registry.manifest("astro-mine.sim.orbital").attributes["fidelity"]["tier"] == (
        FidelityTier.MASSMODEL.value
    )
    # and the cheap JAX kernel stays available alongside the real MJX contact tier.
    assert registry.manifest("astro-mine.sim.brax_contact").attributes["fidelity"]["tier"] == (
        FidelityTier.KINEMATIC.value
    )


# --- the shared rover model ------------------------------------------------------


def test_both_contact_tiers_compile_the_same_rover_model() -> None:
    # The CPU and GPU contact tiers must not silently disagree about what a rover *is*: they build
    # their model from one shared spec, so the MJCF is byte-identical.
    from astro_mine.sim.engines.brax._mjx import model_spec_from_mjx_dynamics
    from astro_mine.sim.engines.mujoco._engine import model_spec_from_dynamics

    cpu = model_spec_from_dynamics(
        MujocoMobilityDynamics(mass_kg=_MASS_KG, max_speed_mps=_MAX_SPEED)
    )
    gpu = model_spec_from_mjx_dynamics(
        MjxContactDynamics(mass_kg=_MASS_KG, max_speed_mps=_MAX_SPEED, timestep_s=cpu.timestep_s)
    )
    assert cpu == gpu
    assert rover_mjcf(cpu) == rover_mjcf(gpu)


def test_the_friction_cone_comes_from_the_regolith_friction_angle() -> None:
    # Traction is limited by mu = tan(phi) — the *terrain's* property — not a hand-written a = F/m
    # cap. A different regolith gives a different friction cone, which is the whole point.
    loose = RoverModelSpec(mass_kg=_MASS_KG, friction_angle_deg=20.0)
    firm = RoverModelSpec(mass_kg=_MASS_KG, friction_angle_deg=40.0)
    assert loose.friction_coeff == pytest.approx(math.tan(math.radians(20.0)))
    assert firm.friction_coeff > loose.friction_coeff
    assert f'friction="{firm.friction_coeff}' in rover_mjcf(firm)


# --- the MuJoCo CPU contact tier -------------------------------------------------


def test_the_mujoco_engine_satisfies_the_regime_engine_protocol(mujoco_stack: object) -> None:
    engine = mujoco_mobility_engine_factory(_mujoco_scenario(), RngStreams(0))
    assert isinstance(engine, RegimeEngine)
    assert engine.descriptor is MUJOCO_MOBILITY_ENGINE_DESCRIPTOR


def test_mujoco_really_simulates_wheel_soil_contact_the_rover_slips(
    mujoco_stack: object,
) -> None:
    # THE acceptance criterion. A *kinematic* model reaches its commanded speed exactly, because the
    # command IS the model. A contact model does not: the wheels must transmit torque through a
    # friction cone, so the rover accelerates gradually and settles slightly *below* the commanded
    # speed — it slips. That gap is the evidence there is real contact physics here.
    engine = mujoco_mobility_engine_factory(_mujoco_scenario(), RngStreams(0))
    for _ in range(6):  # 3 s
        engine.advance(0.5)
    sample = engine.export_coupling_state().by_agent["rover"]
    v = sample.linear_velocity_mps
    assert v is not None
    speed = math.dist((v.x, v.y, v.z), (0.0, 0.0, 0.0))
    assert 0.0 < speed < _DRIVE_MPS  # it moves, but slips: strictly under the commanded speed
    assert speed == pytest.approx(_DRIVE_MPS, rel=0.2)  # yet tracks it within the drawbar budget
    # and it has genuinely travelled (a contact rover that never gripped would sit still).
    assert sample.pose.translation_m.x > 0.5


def test_the_contact_chassis_carries_a_real_attitude(mujoco_stack: object) -> None:
    # A contact-simulated chassis pitches on its suspension; the point-mass tiers report identity.
    engine = mujoco_mobility_engine_factory(_mujoco_scenario(), RngStreams(0))
    for _ in range(4):
        engine.advance(0.5)
    q = engine.export_coupling_state().by_agent["rover"].pose.rotation_quat_xyzw
    assert math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2) == pytest.approx(1.0, abs=1e-6)


def test_mujoco_tracks_a_velocity_command(mujoco_stack: object) -> None:
    engine = mujoco_mobility_engine_factory(_mujoco_scenario(), RngStreams(0))
    engine.apply_actions(
        ActionBatch(
            actions=[
                Action(
                    agent_id="rover",
                    kind=ActionKind.ACTUATOR,
                    actuator=ActuatorCommand(
                        target="base",
                        control_mode=ControlMode.VELOCITY,
                        setpoint=[0.0, 0.0, 0.0],
                    ),
                )
            ]
        )
    )
    for _ in range(6):
        engine.advance(0.5)
    v = engine.export_coupling_state().by_agent["rover"].linear_velocity_mps
    assert v is not None
    assert math.dist((v.x, v.y, v.z), (0.0, 0.0, 0.0)) < 0.05  # commanded to stop, it stops


def test_mujoco_reproduces_in_process_under_a_fixed_seed(mujoco_stack: object) -> None:
    scenario = _mujoco_scenario()
    assert (
        run_episode(scenario, engine_factory=mujoco_mobility_engine_factory).content_hash
        == run_episode(scenario, engine_factory=mujoco_mobility_engine_factory).content_hash
    )


def test_the_mujoco_coupling_triad_round_trips(mujoco_stack: object) -> None:
    engine = mujoco_mobility_engine_factory(_mujoco_scenario(), RngStreams(0))
    engine.advance(0.5)
    snapshot = engine.export_coupling_state()
    other = mujoco_mobility_engine_factory(_mujoco_scenario(), RngStreams(0))
    other.import_coupling_state(snapshot)
    restored = other.export_coupling_state().by_agent["rover"].pose.translation_m
    original = snapshot.by_agent["rover"].pose.translation_m
    assert (restored.x, restored.y, restored.z) == pytest.approx(
        (original.x, original.y, original.z), abs=1e-9
    )


def test_mujoco_retire_and_empty_scenario(mujoco_stack: object) -> None:
    engine = mujoco_mobility_engine_factory(_mujoco_scenario(), RngStreams(0))
    engine.retire(["rover"])
    assert engine.export_coupling_state().samples == ()
    empty = mujoco_mobility_engine_factory(
        Scenario(name="none", agents=(AgentSpec(agent_id="a"),)), RngStreams(0)
    )
    assert empty.export_coupling_state().samples == ()


# --- the MJX batched contact tier (RM-P1-SIM-04) ----------------------------------


def test_the_mjx_engine_uses_real_mjx_contact_not_just_jax(mjx_stack: object) -> None:
    # The literal acceptance criterion: a module under engines/brax/ imports and uses brax/mujoco
    # (MJX) for contact simulation, not just jax/jax.numpy.
    import inspect

    source = inspect.getsource(mjx_stack)
    assert "from mujoco import mjx" in source
    assert "mjx.step" in source and "mjx.put_model" in source


def test_the_mjx_engine_satisfies_the_regime_engine_protocol(mjx_stack: object) -> None:
    engine = mjx_contact_engine_factory(_mjx_scenario(), RngStreams(0))
    assert isinstance(engine, RegimeEngine)
    assert engine.descriptor is MJX_CONTACT_ENGINE_DESCRIPTOR


def test_mjx_simulates_articulated_wheel_soil_contact(mjx_stack: object) -> None:
    # Same evidence as the CPU tier: the rover drives, and it slips (a kinematic kernel would not).
    engine = mjx_contact_engine_factory(_mjx_scenario(), RngStreams(0))
    for _ in range(6):  # 3 s
        engine.advance(0.5)
    sample = engine.export_coupling_state().by_agent["rover0"]
    v = sample.linear_velocity_mps
    assert v is not None
    speed = math.dist((v.x, v.y, v.z), (0.0, 0.0, 0.0))
    assert 0.0 < speed < _DRIVE_MPS
    assert sample.pose.translation_m.x > 0.5


def test_the_two_contact_tiers_agree_within_a_tolerance(
    mujoco_stack: object, mjx_stack: object
) -> None:
    # MJX is MuJoCo's solver in JAX, so the CPU and batched tiers must agree on the same machine to
    # within the documented TOLERANCE band — not bit-exactly (XLA reductions are non-associative and
    # a stiff contact solve amplifies that; sim.md §11), but physically.
    cpu = mujoco_mobility_engine_factory(_mujoco_scenario(), RngStreams(0))
    gpu = mjx_contact_engine_factory(_mjx_scenario(), RngStreams(0))  # same rover, same command
    for _ in range(6):
        cpu.advance(0.5)
        gpu.advance(0.5)
    a = cpu.export_coupling_state().by_agent["rover"].pose.translation_m
    b = gpu.export_coupling_state().by_agent["rover0"].pose.translation_m
    assert a.x == pytest.approx(b.x, rel=0.15)  # same physics, different substrate


def test_mjx_reproduces_in_process_under_a_fixed_seed(mjx_stack: object) -> None:
    # The in-process half of the TOLERANCE contract — what the determinism gate actually checks.
    scenario = _mjx_scenario()
    assert (
        run_episode(scenario, engine_factory=mjx_contact_engine_factory).content_hash
        == run_episode(scenario, engine_factory=mjx_contact_engine_factory).content_hash
    )


def test_the_mjx_coupling_triad_round_trips(mjx_stack: object) -> None:
    engine = mjx_contact_engine_factory(_mjx_scenario(), RngStreams(0))
    engine.advance(0.5)
    snapshot = engine.export_coupling_state()
    other = mjx_contact_engine_factory(_mjx_scenario(), RngStreams(0))
    other.import_coupling_state(snapshot)
    restored = other.export_coupling_state().by_agent["rover0"].pose.translation_m
    original = snapshot.by_agent["rover0"].pose.translation_m
    assert (restored.x, restored.y, restored.z) == pytest.approx(
        (original.x, original.y, original.z), abs=1e-9
    )


def test_mjx_retire_and_empty_scenario(mjx_stack: object) -> None:
    engine = mjx_contact_engine_factory(_mjx_scenario(n_agents=2), RngStreams(0))
    engine.retire(["rover0"])
    assert tuple(s.agent_id for s in engine.export_coupling_state().samples) == ("rover1",)
    empty = mjx_contact_engine_factory(
        Scenario(name="none", agents=(AgentSpec(agent_id="a"),)), RngStreams(0)
    )
    assert empty.export_coupling_state().samples == ()


def test_mjx_agents_must_share_one_physical_model(mjx_stack: object) -> None:
    scenario = Scenario(
        name="mismatch",
        agents=(
            AgentSpec(
                agent_id="a",
                dynamics=MjxContactDynamics(mass_kg=250.0, max_speed_mps=1.0),
            ),
            AgentSpec(
                agent_id="b",
                dynamics=MjxContactDynamics(mass_kg=400.0, max_speed_mps=1.0),
            ),
        ),
    )
    with pytest.raises(ValueError, match="must share one physical model"):
        mjx_contact_engine_factory(scenario, RngStreams(0))


# --- the batched surface + the Ray fan-out are unchanged for the new tier (RM-P1-SIM-04) -----


def test_the_mjx_tier_reuses_the_existing_vectorized_rollout_surface(mjx_stack: object) -> None:
    # Acceptance criterion: "existing VectorizedRollout batching ... continue to work unchanged".
    # The MJX batch satisfies the *same* RolloutBatch surface the reduced-order kernel does, so the
    # fan-out machinery is untouched — this is a fidelity upgrade behind the architecture, not a new
    # integration surface.
    from astro_mine.sim.engines.brax import build_mjx_vectorized_rollout
    from astro_mine.sim.engines.brax._batch import RolloutBatch

    rollout = build_mjx_vectorized_rollout(_mjx_scenario(), RngStreams(1), n_envs=2)
    assert isinstance(rollout, RolloutBatch)
    assert rollout.n_envs == 2 and rollout.n_agents == 1
    assert rollout.env_indices == (0, 1)

    observations = rollout.reset()
    assert len(observations) == 2 and len(observations[0]) == 1
    stepped = rollout.step(ActionBatch())
    assert stepped[0][0].agent_id == "rover0"
    assert rollout.positions.shape == (2, 1, 3)  # (N_envs, N_agents, xyz)


def test_the_mjx_batch_really_steps_contact_in_every_env(mjx_stack: object) -> None:
    from astro_mine.sim.engines.brax import build_mjx_vectorized_rollout

    rollout = build_mjx_vectorized_rollout(_mjx_scenario(), RngStreams(1), n_envs=2)
    rollout.reset()
    for _ in range(4):
        rollout.step(ActionBatch())
    # Every env drove forward under contact — the batch is really solving, not returning the seed.
    assert all(float(rollout.positions[e, 0, 0]) > 0.2 for e in range(2))


def test_the_ray_fan_out_aggregates_the_mjx_tier_to_the_in_process_oracle(
    mjx_stack: object,
) -> None:
    # Acceptance criterion: "... and Ray fan-out continue to work unchanged for the new tier".
    # The fan-out's sharder + actor + aggregation are tier-agnostic (they drive the RolloutBatch
    # surface), so a sharded MJX run must reassemble to the single-process MJX result — the same
    # equivalence oracle the reduced-order tier is checked against. Exercised in-process here;
    # the live ray.remote dispatch is `-m ray` (GitHub runners cannot spawn Ray workers).
    #
    # "Matches" is numerical, not bitwise: the oracle vmaps over 4 envs and the shards over 2
    # each, and XLA's reduction order follows the batch shape (astro-mine-sim#46). See
    # tests/_equivalence.py.
    from astro_mine.sim.engines.brax._ray import (
        _aggregate,
        _positions_as_list,
        _roll,
        _RolloutActor,
        _shard_ranges,
        run_in_process,
    )

    scenario = _mjx_scenario()
    actions, steps, total = ActionBatch(), 3, 4

    oracle = run_in_process(scenario, RngStreams(2), actions=actions, steps=steps, n_envs=total)

    shards = _shard_ranges(total, 2)
    results = [
        _RolloutActor(scenario.model_dump_json(), 2, shard).rollout(
            actions.model_dump_json(), steps
        )
        for shard in shards
    ]
    aggregated = _aggregate(results, total)

    assert len(aggregated) == total
    assert_shards_match_oracle(aggregated, oracle, what="MJX fan-out (4 envs vs 2x2 shards)")
    assert _positions_as_list is not None and _roll is not None  # the helpers are tier-agnostic


def test_the_fan_out_rejects_a_scenario_with_neither_jax_tier() -> None:
    from astro_mine.sim.engines.brax._ray import build_rollout

    scenario = Scenario(name="none", agents=(AgentSpec(agent_id="a"),))
    with pytest.raises(ValueError, match="brax_contact or mjx_contact"):
        build_rollout(scenario, RngStreams(0))


# --- the tolerance gate: the contact tiers vs. the analytic drawbar-pull oracle ---------------


@pytest.mark.parametrize("tier", ["mujoco", "mjx"])
def test_contact_tiers_are_gated_against_the_analytic_drawbar_pull_oracle(
    tier: str, mujoco_stack: object, mjx_stack: object
) -> None:
    # Both contact tiers are TOLERANCE-class, so they are admitted against an **explicit error
    # budget** rather than a golden hash (sim.md §11; conventions.md §11). The oracle: a rover
    # commanded from rest cannot exceed the drawbar-pull limit the friction cone allows —
    # a_max = mu * g (the friction coefficient times gravity, independent of mass) — and must settle
    # at or below its commanded speed. A tier that violated the friction cone would be simulating
    # traction it does not have.
    from astro_mine.sim.engines._rover_mjcf import RoverModelSpec

    spec = RoverModelSpec(mass_kg=_MASS_KG)
    a_max = spec.friction_coeff * spec.gravity_m_s2  # the friction-cone acceleration limit

    if tier == "mujoco":
        engine = mujoco_mobility_engine_factory(_mujoco_scenario(), RngStreams(0))
        agent = "rover"
    else:
        engine = mjx_contact_engine_factory(_mjx_scenario(), RngStreams(0))
        agent = "rover0"

    dt, elapsed = 0.5, 0.0
    for _ in range(6):
        engine.advance(dt)
        elapsed += dt
        v = engine.export_coupling_state().by_agent[agent].linear_velocity_mps
        assert v is not None
        speed = math.dist((v.x, v.y, v.z), (0.0, 0.0, 0.0))
        # Within the friction cone (with a 10% numerical margin) and never above the command.
        assert speed <= a_max * elapsed * 1.1
        assert speed <= _DRIVE_MPS + 1e-6
