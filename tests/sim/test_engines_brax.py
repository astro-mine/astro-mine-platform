"""RM-P1-SIM-04 — the Brax/MJX GPU-vectorizable contact/mobility engine.

A JAX-native low-fidelity mobility tier behind the ``RegimeEngine`` waist: it registers
**manifest-only** (no JAX needed to register), satisfies the engine Protocol, round-trips the
coupling triad, and is gated by a documented **tolerance** — same-seed runs reproduce in-process and
its velocity profile cross-checks the reduced-order mobility oracle within an explicit budget
(``TOLERANCE``, not bit-exact: XLA reductions are non-associative / not bit-portable across builds,
sim.md §11). The JAX-dependent tests skip without ``astro-mine-sim[brax]``; the descriptor/manifest
tests run regardless (proving registration needs no JAX).
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages.enums import ActionKind, ControlMode, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    GotoTask,
    ModeCommand,
    Quat,
    TaskDirective,
    Transform,
    Vec3,
)
from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import (
    GATED_CAPABILITY_TAGS,
    CapabilityTag,
    DeterminismClass,
    FidelityTier,
    Regime,
)
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines import RegimeEngine, default_engine_registry
from astro_mine.sim.engines.brax import BRAX_CONTACT_ENGINE_DESCRIPTOR, brax_contact_engine_factory
from astro_mine.sim.runtime import AgentSpec, BraxContactDynamics, RngStreams, Scenario

_ENGINE_NAME = "astro-mine.sim.brax_contact"


@pytest.fixture
def brax_engine():
    """Skip unless the JAX stack (``[brax]`` extra) is importable; return the JAX engine module."""
    pytest.importorskip("jax")
    pytest.importorskip("brax")
    pytest.importorskip("mujoco")
    from astro_mine.sim.engines.brax import _engine

    return _engine


def _scenario(*, agents=None, velocity=(0.0, 0.0, 0.0), **dyn) -> Scenario:
    params: dict[str, object] = {
        "mass_kg": 800.0,
        "max_speed_mps": 1.0,
        "max_traction_n": 400.0,
    }
    params.update(dyn)
    default = (
        AgentSpec(
            agent_id="rover",
            battery_soc_j=1.0e7,
            battery_floor_j=0.0,
            velocity_mps=velocity,
            dynamics=BraxContactDynamics(**params),  # type: ignore[arg-type]
        ),
    )
    return Scenario(name="brax-mob", horizon_steps=1, dt_s=0.5, agents=agents or default)


def _velocity_cmd(vec: tuple[float, float, float]) -> ActionBatch:
    return ActionBatch(
        actions=[
            Action(
                agent_id="rover",
                kind=ActionKind.ACTUATOR,
                actuator=ActuatorCommand(
                    target="base", control_mode=ControlMode.VELOCITY, setpoint=list(vec)
                ),
            )
        ]
    )


def _goto_cmd(point: tuple[float, float, float]) -> ActionBatch:
    return ActionBatch(
        actions=[
            Action(
                agent_id="rover",
                kind=ActionKind.TASK,
                task=TaskDirective(
                    task_kind=TaskKind.GOTO,
                    goto=GotoTask(
                        target_frame="MOON_ME",
                        target_pose=Transform(
                            translation_m=Vec3(x=point[0], y=point[1], z=point[2]),
                            rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                        ),
                    ),
                ),
            )
        ]
    )


# -- registration / descriptor (no JAX) -----------------------------------------------------


def test_descriptor_declares_surface_kinematic_tolerance() -> None:
    d = BRAX_CONTACT_ENGINE_DESCRIPTOR
    assert d.name == _ENGINE_NAME
    assert d.regimes == (Regime.SURFACE,)
    assert d.frames == (MOON_BODY_FIXED,)
    assert d.fidelity.tier is FidelityTier.KINEMATIC
    assert d.determinism_class is DeterminismClass.TOLERANCE
    assert CapabilityTag.MOBILITY_WHEELED in d.capability_tags
    assert d.to_manifest().kind is PluginKind.REGIME_ENGINE


def test_mobility_wheeled_is_not_a_gated_tag() -> None:
    # The engine advertises MOBILITY_WHEELED; an open-commons asset/plugin may declare it, so
    # registration (which gates gated tags) does not reject the engine.
    assert CapabilityTag.MOBILITY_WHEELED not in GATED_CAPABILITY_TAGS


def test_registration_is_manifest_only_and_resolves() -> None:
    # The descriptor registers via the default registry (manifest-only) without the factory ever
    # importing JAX — the [brax] extra is only needed to build an engine (mirrors the DEM engine).
    registry = default_engine_registry()
    assert _ENGINE_NAME in registry.names
    assert registry.manifest(_ENGINE_NAME).kind is PluginKind.REGIME_ENGINE


def test_missing_jax_stack_raises_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the JAX stack being absent: each lazy factory must re-raise a clean
    # ModuleNotFoundError naming the [brax] extra (never a raw JAX ImportError). Runs without JAX.
    import builtins

    from astro_mine.sim.engines.brax import build_vectorized_rollout, fan_out

    real_import = builtins.__import__
    guarded = {
        "astro_mine.sim.engines.brax._engine",
        "astro_mine.sim.engines.brax._batch",
        "astro_mine.sim.engines.brax._ray",
    }

    def fake_import(name: str, *args: object, **kwargs: object):
        if name in guarded:
            raise ModuleNotFoundError("No module named 'jax'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    scenario = _scenario()
    for call in (
        lambda: brax_contact_engine_factory(scenario, RngStreams(0)),
        lambda: build_vectorized_rollout(scenario, RngStreams(0), n_envs=4),
        lambda: fan_out(scenario, RngStreams(0), actions=ActionBatch(), steps=1),
    ):
        with pytest.raises(ModuleNotFoundError, match=r"astro-mine-platform\[sim-brax\]"):
            call()


# -- Protocol conformance + coupling triad (needs [brax]) -----------------------------------


def test_engine_satisfies_regime_engine_protocol(brax_engine) -> None:
    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    assert isinstance(engine, RegimeEngine)
    assert isinstance(engine, brax_engine.BraxContactEngine)
    assert engine.descriptor is BRAX_CONTACT_ENGINE_DESCRIPTOR


def test_factory_builds_only_brax_agents(brax_engine) -> None:
    scenario = _scenario(
        agents=(
            AgentSpec(
                agent_id="rover",
                battery_soc_j=1.0e7,
                dynamics=BraxContactDynamics(
                    mass_kg=800.0, max_speed_mps=1.0, max_traction_n=400.0
                ),
            ),
            AgentSpec(agent_id="kin"),  # kinematic default — skipped
        )
    )
    engine = brax_contact_engine_factory(scenario, RngStreams(0))
    assert set(engine.export_coupling_state().by_agent) == {"rover"}


def test_velocity_command_moves_the_rover(brax_engine) -> None:
    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(_velocity_cmd((0.8, 0.0, 0.0)))
    engine.advance(0.5)
    sample = engine.export_coupling_state().by_agent["rover"]
    assert sample.pose.translation_m.x > 0.0
    assert sample.linear_velocity_mps is not None and sample.linear_velocity_mps.x > 0.0


def test_goto_task_drives_toward_target(brax_engine) -> None:
    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(_goto_cmd((100.0, 0.0, 0.0)))
    for _ in range(4):
        engine.advance(0.5)
    assert engine.export_coupling_state().by_agent["rover"].pose.translation_m.x > 0.0


def test_goto_to_the_current_position_holds_still(brax_engine) -> None:
    # A goto target equal to the current position resolves to zero desired velocity (no overshoot).
    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(_goto_cmd((0.0, 0.0, 0.0)))
    engine.advance(0.5)
    pose = engine.export_coupling_state().by_agent["rover"].pose.translation_m
    assert pose.x == 0.0 and pose.y == 0.0 and pose.z == 0.0


def test_mismatched_brax_params_raise(brax_engine) -> None:
    scenario = Scenario(
        name="mismatch",
        agents=(
            AgentSpec(
                agent_id="r1",
                dynamics=BraxContactDynamics(
                    mass_kg=800.0, max_speed_mps=1.0, max_traction_n=400.0
                ),
            ),
            AgentSpec(
                agent_id="r2",
                dynamics=BraxContactDynamics(
                    mass_kg=900.0, max_speed_mps=1.0, max_traction_n=400.0
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="share one parameter set"):
        brax_contact_engine_factory(scenario, RngStreams(0))


def test_engine_with_no_brax_agents_is_an_empty_no_op(brax_engine) -> None:
    # The heterogeneous co-step (RM-P0-SIM-04): built on a non-brax scenario, the engine owns
    # nothing — export is empty and advance/import just track the clock without a kernel call.
    from astro_mine.sim.engines.adapter import CouplingState

    scenario = Scenario(name="kin-only", agents=(AgentSpec(agent_id="kin"),))
    engine = brax_contact_engine_factory(scenario, RngStreams(0))
    assert engine.export_coupling_state().by_agent == {}
    engine.advance(0.5)
    engine.import_coupling_state(CouplingState(sim_time_s=1.0, samples=()))
    assert engine.export_coupling_state().by_agent == {}


def test_mode_command_sets_mode(brax_engine) -> None:
    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(
        ActionBatch(
            actions=[
                Action(agent_id="rover", kind=ActionKind.MODE, mode=ModeCommand(mode="driving"))
            ]
        )
    )
    assert engine.export_coupling_state().by_agent["rover"].mode == "driving"


def test_action_for_an_unowned_agent_is_ignored(brax_engine) -> None:
    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(
        ActionBatch(
            actions=[Action(agent_id="ghost", kind=ActionKind.MODE, mode=ModeCommand(mode="x"))]
        )
    )
    assert set(engine.export_coupling_state().by_agent) == {"rover"}


def test_retire_drops_the_agent(brax_engine) -> None:
    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    engine.retire(["rover"])
    assert engine.export_coupling_state().by_agent == {}


def test_retire_of_an_unowned_agent_is_a_no_op(brax_engine) -> None:
    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    engine.retire(["ghost"])
    assert set(engine.export_coupling_state().by_agent) == {"rover"}


def test_import_coupling_ignores_snapshots_without_this_agent(brax_engine) -> None:
    from astro_mine.sim.engines.adapter import CouplingState

    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    before = engine.export_coupling_state().by_agent["rover"].pose
    engine.import_coupling_state(CouplingState(sim_time_s=0.0, samples=()))
    assert engine.export_coupling_state().by_agent["rover"].pose == before


def test_coupling_state_round_trips(brax_engine) -> None:
    engine = brax_contact_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(_velocity_cmd((0.8, 0.2, 0.0)))
    engine.advance(0.5)
    snapshot = engine.export_coupling_state()
    engine.advance(0.5)
    engine.import_coupling_state(snapshot)
    restored = engine.export_coupling_state().by_agent["rover"]
    original = snapshot.by_agent["rover"]
    assert restored.pose == original.pose
    assert restored.battery_soc_j == original.battery_soc_j
    assert restored.linear_velocity_mps == original.linear_velocity_mps


def test_battery_is_drawn_while_driving(brax_engine) -> None:
    engine = brax_contact_engine_factory(
        _scenario(idle_power_w=10.0, drive_power_w_per_mps=100.0), RngStreams(0)
    )
    soc0 = engine.export_coupling_state().by_agent["rover"].battery_soc_j
    engine.apply_actions(_velocity_cmd((1.0, 0.0, 0.0)))
    for _ in range(6):
        engine.advance(0.5)
    soc1 = engine.export_coupling_state().by_agent["rover"].battery_soc_j
    assert soc0 is not None and soc1 is not None and soc1 < soc0


# -- determinism (TOLERANCE: in-process reproducibility + oracle cross-check) ----------------


def test_cpu_same_seed_reproduces_within_tolerance(brax_engine) -> None:
    # Same seed twice must agree within a documented tolerance — NOT bit-exactness (the descriptor's
    # TOLERANCE rationale). Domain-randomization jitter (seeded) is on, so the seed is load-bearing.
    def run():
        engine = brax_contact_engine_factory(_scenario(init_speed_jitter_mps=0.5), RngStreams(11))
        engine.apply_actions(_velocity_cmd((0.8, 0.3, 0.0)))
        for _ in range(6):
            engine.advance(0.5)
        s = engine.export_coupling_state().by_agent["rover"]
        return s.pose.translation_m, s.linear_velocity_mps, s.battery_soc_j

    a_pos, a_vel, a_soc = run()
    b_pos, b_vel, b_soc = run()
    tol = 1e-9  # documented absolute tolerance; in-process on one build this is 0
    assert abs(a_pos.x - b_pos.x) <= tol and abs(a_pos.y - b_pos.y) <= tol
    assert a_vel is not None and b_vel is not None and abs(a_vel.x - b_vel.x) <= tol
    assert a_soc is not None and b_soc is not None and abs(a_soc - b_soc) <= tol


def test_seeded_jitter_varies_across_seeds(brax_engine) -> None:
    def initial_velocity(seed: int):
        engine = brax_contact_engine_factory(_scenario(init_speed_jitter_mps=1.0), RngStreams(seed))
        v = engine.export_coupling_state().by_agent["rover"].linear_velocity_mps
        assert v is not None
        return (v.x, v.y, v.z)

    assert initial_velocity(1) != initial_velocity(2)


def test_cpu_tolerance_gate_cross_checks_mobility_oracle(brax_engine) -> None:
    # The reduced-order MobilityEngine is the oracle: identical params + command, compare the
    # velocity trajectories within an explicit budget (sim.md §11). The Brax tier is the same model
    # in JAX, so the budget is tight — it documents the tolerance, not a loose escape hatch.
    from astro_mine.sim.engines.mobility import mobility_engine_factory
    from astro_mine.sim.runtime import MobilityDynamics
    from astro_mine.sim.validation import validate_against_oracle

    dyn = {
        "mass_kg": 800.0,
        "max_speed_mps": 1.0,
        "max_traction_n": 400.0,
        "idle_power_w": 10.0,
        "drive_power_w_per_mps": 100.0,
    }
    brax = brax_contact_engine_factory(_scenario(**dyn), RngStreams(7))
    mob_scenario = Scenario(
        name="mob",
        horizon_steps=1,
        dt_s=0.5,
        agents=(
            AgentSpec(
                agent_id="rover",
                battery_soc_j=1.0e7,
                dynamics=MobilityDynamics(**dyn),  # type: ignore[arg-type]
            ),
        ),
    )
    mob = mobility_engine_factory(mob_scenario, RngStreams(7))

    command = _velocity_cmd((2.0, 0.0, 0.0))  # above the cap → full ramp then hold
    brax.apply_actions(command)
    mob.apply_actions(command)
    brax_v: list[tuple[float, float, float]] = []
    mob_v: list[tuple[float, float, float]] = []
    for _ in range(8):
        brax.advance(0.5)
        mob.advance(0.5)
        bv = brax.export_coupling_state().by_agent["rover"].linear_velocity_mps
        mv = mob.export_coupling_state().by_agent["rover"].linear_velocity_mps
        assert bv is not None and mv is not None
        brax_v.append((bv.x, bv.y, bv.z))
        mob_v.append((mv.x, mv.y, mv.z))

    report = validate_against_oracle(
        brax_v, mob_v, budget=1e-6, name="brax-vs-mobility", relative=True
    )
    assert report.passed, f"brax diverged from the mobility oracle: {report}"


def test_runs_in_a_real_episode_and_reproduces(brax_engine) -> None:
    # Drive the Brax engine through the full episode loop (reset/step/Trace) — exercises the
    # scheduler / provenance kind-routing and the TOLERANCE reproducibility gate.
    from astro_mine.sim.runtime import run_episode

    scenario = _scenario()
    t1 = run_episode(scenario, seed=1, engine_factory=brax_contact_engine_factory)
    t2 = run_episode(scenario, seed=1, engine_factory=brax_contact_engine_factory)
    assert t1.content_hash == t2.content_hash
