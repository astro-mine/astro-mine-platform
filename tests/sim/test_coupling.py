"""RM-P0-SIM-04 — the multi-domain coupler (explicit co-simulation).

Covers the acceptance criteria: the orbital + surface-mobility + excavation engines co-step
across named boundaries with residuals tracked and bounded, and frame/unit bridging is
explicit (a cross-frame boundary fails loudly). Also pins the reproducibility invariant the
refactor must not break — a coupler over a single kinematic engine reproduces the reference
stepping core byte-for-byte (RM-P0-SIM-01).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    ModeCommand,
    Observation,
    Quat,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import INERTIAL_J2000, MOON_BODY_FIXED
from astro_mine.sim.coupling import (
    COUPLED_ENGINE_NAME,
    CoupledEngine,
    CouplingBoundary,
    CouplingResidual,
    CouplingSchedule,
    FrameBridgeError,
    coupled_engine_factory,
)
from astro_mine.sim.engines import (
    CouplingState,
    EngineDescriptor,
    FidelityDescriptor,
    RegimeEngine,
    granular_engine_factory,
    mobility_engine_factory,
    orbital_engine_factory,
)
from astro_mine.sim.runtime import (
    AgentSpec,
    GranularDynamics,
    MobilityDynamics,
    OrbitalDynamics,
    RngStreams,
    Scenario,
    Simulator,
    load_scenario,
)

DATA = Path(__file__).parent / "data"


# --- helpers ---------------------------------------------------------------------


def _episode_observation_dumps(
    scenario: Scenario, factory: object | None = None
) -> list[dict[str, object]]:
    """Roll a whole episode and return the per-frame observation dumps — the byte-level
    surface a determinism comparison checks."""
    sim = Simulator(scenario, engine_factory=factory) if factory else Simulator(scenario)  # type: ignore[arg-type]

    def dump(obs: Mapping[str, Observation]) -> dict[str, object]:
        return {a: o.model_dump(mode="json") for a, o in obs.items()}

    frames = [dump(sim.reset().observations)]
    for _ in range(scenario.horizon_steps):
        frames.append(dump(sim.step(ActionBatch()).observations))
    return frames


def _mobility_excavator(agent_id: str, *, frame: object | None = None) -> Scenario:
    return Scenario(
        name="mover",
        agents=(
            AgentSpec(
                agent_id=agent_id,
                initial_position_m=(0.0, 0.0, 0.0),
                velocity_mps=(0.5, 0.0, 0.0),
                battery_soc_j=300.0,
                frame=frame,  # type: ignore[arg-type]
                dynamics=MobilityDynamics(mass_kg=200.0, max_speed_mps=0.5, max_traction_n=600.0),
            ),
        ),
    )


def _granular_excavator(agent_id: str) -> Scenario:
    return Scenario(
        name="digger",
        agents=(
            AgentSpec(
                agent_id=agent_id,
                initial_position_m=(0.0, 0.0, 0.0),
                battery_soc_j=400.0,
                dynamics=GranularDynamics(max_dig_rate_m3_s=0.01),
            ),
        ),
    )


_SPY_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.spy",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED,),
    determinism_class=DeterminismClass.BIT_EXACT,
    fidelity=FidelityDescriptor(tier=FidelityTier.KINEMATIC),
)


class _SpyEngine:
    """A minimal :class:`RegimeEngine` that records the ``dt`` of each advance call — so a
    test can prove the coupler's multi-rate sub-stepping drives it at the scheduled rate."""

    def __init__(self) -> None:
        self.advance_calls: list[float] = []
        self._sample = StateSample(
            agent_id="spy",
            frame=MOON_BODY_FIXED,
            pose=Transform(
                translation_m=Vec3(x=0.0, y=0.0, z=0.0),
                rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

    @property
    def descriptor(self) -> EngineDescriptor:
        return _SPY_DESCRIPTOR

    def apply_actions(self, actions: ActionBatch) -> None:
        pass

    def advance(self, dt_s: float) -> None:
        self.advance_calls.append(dt_s)

    def export_coupling_state(self) -> CouplingState:
        return CouplingState(sim_time_s=sum(self.advance_calls), samples=(self._sample,))

    def import_coupling_state(self, state: CouplingState) -> None:
        pass

    def retire(self, agent_ids: Iterable[str]) -> None:
        pass


# --- the composite is a RegimeEngine behind the waist ----------------------------


def test_coupled_engine_satisfies_the_regime_engine_protocol() -> None:
    factory = coupled_engine_factory()
    engine = factory(load_scenario(DATA / "heterogeneous.json"), RngStreams(11))
    assert isinstance(engine, RegimeEngine)  # routes behind the Core Environment waist
    assert isinstance(engine, CoupledEngine)
    assert engine.descriptor.name == COUPLED_ENGINE_NAME


# --- co-stepping the heterogeneous anchor engine set -----------------------------


def test_factory_co_steps_orbital_mobility_and_granular_engines() -> None:
    scenario = load_scenario(DATA / "heterogeneous.json")
    sim = Simulator(scenario, engine_factory=coupled_engine_factory())
    start = sim.reset()
    assert set(start.observations) == {"relay", "rover", "digger"}

    results = [sim.step(ActionBatch()) for _ in range(scenario.horizon_steps)]
    last = results[-1].observations
    assert set(last) == {"relay", "rover", "digger"}
    # the relay propagated orbitally (inertial position swept out an arc),
    relay_x0 = start.observations["relay"].self_state.pose.translation_m
    relay_x1 = last["relay"].self_state.pose.translation_m
    assert (relay_x1.x, relay_x1.y) != (relay_x0.x, relay_x0.y)
    # the rover drove along +x,
    assert last["rover"].self_state.pose.translation_m.x > 0.0
    # and the digger is co-stepped (present and frame-explicit) all the way through.
    assert last["digger"].self_state.frame == MOON_BODY_FIXED


def test_co_step_is_deterministic_under_a_fixed_seed() -> None:
    scenario = load_scenario(DATA / "heterogeneous.json")
    assert _episode_observation_dumps(
        scenario, coupled_engine_factory()
    ) == _episode_observation_dumps(scenario, coupled_engine_factory())


# --- reproducibility: the coupler never perturbs an existing trace ----------------


def test_coupler_over_a_single_kinematic_engine_reproduces_the_baseline() -> None:
    # An all-kinematic scenario yields a single kinematic sub-engine; the coupled path must
    # match the reference stepping core's default engine byte-for-byte (CX-REPRO).
    scenario = load_scenario(DATA / "scenario.json")
    assert _episode_observation_dumps(scenario, coupled_engine_factory()) == (
        _episode_observation_dumps(scenario)
    )


# --- multi-rate sub-stepping -----------------------------------------------------


def test_schedule_sub_steps_each_engine_at_its_own_rate() -> None:
    spy = _SpyEngine()
    coupled = CoupledEngine({"spy": spy}, schedule=CouplingSchedule(substeps={"spy": 3}))
    coupled.advance(1.5)
    assert spy.advance_calls == [0.5, 0.5, 0.5]  # 3 equal sub-steps of dt/3


def test_default_schedule_advances_once_per_macro_step() -> None:
    spy = _SpyEngine()
    CoupledEngine({"spy": spy}).advance(2.0)
    assert spy.advance_calls == [2.0]


# --- named coupling boundaries + tracked residuals -------------------------------


def test_boundary_hands_pose_to_the_consumer_and_tracks_a_bounded_residual() -> None:
    mover = mobility_engine_factory(_mobility_excavator("exc"), RngStreams(0))
    digger = granular_engine_factory(_granular_excavator("exc"), RngStreams(0))
    coupled = CoupledEngine(
        {"mover": mover, "digger": digger},
        boundaries=(CouplingBoundary("dig-site", "mover", "digger", ("exc",)),),
    )
    coupled.advance(2.0)

    mover_pose = mover.export_coupling_state().by_agent["exc"].pose
    digger_pose = digger.export_coupling_state().by_agent["exc"].pose
    # the dig site followed the moving excavator: the consumer adopted the producer's pose.
    assert digger_pose == mover_pose
    assert mover_pose.translation_m.x > 0.0  # the mover actually moved

    (residual,) = coupled.residuals
    assert isinstance(residual, CouplingResidual)
    assert residual.boundary == "dig-site" and residual.agent_id == "exc"
    # the residual is the correction applied (the digger was stale at the origin) and is
    # bounded by the rover's max single-step displacement (max_speed * dt = 1.0 m).
    assert residual.position_residual_m == pytest.approx(mover_pose.translation_m.x)
    assert 0.0 < residual.position_residual_m <= 0.5 * 2.0 + 1e-9


def test_three_anchor_engines_co_step_across_a_named_boundary_with_a_bounded_residual() -> None:
    # The full RM-P0-SIM-04 acceptance shape in one scenario: the orbital + surface-mobility +
    # excavation engines co-step, and a named mobility->granular boundary on the shared excavator
    # tracks a bounded coupling residual (the dig site follows the moving excavator).
    relay = Scenario(
        name="relay",
        agents=(
            AgentSpec(
                agent_id="relay",
                initial_position_m=(1.8e6, 0.0, 0.0),
                velocity_mps=(0.0, 1633.0, 0.0),
                frame=INERTIAL_J2000,
                dynamics=OrbitalDynamics(),
            ),
        ),
    )
    coupled = CoupledEngine(
        {
            "orbital": orbital_engine_factory(relay, RngStreams(0)),
            "mobility": mobility_engine_factory(_mobility_excavator("excavator"), RngStreams(0)),
            "granular": granular_engine_factory(_granular_excavator("excavator"), RngStreams(0)),
        },
        boundaries=(CouplingBoundary("dig-site", "mobility", "granular", ("excavator",)),),
    )
    coupled.advance(2.0)

    by_agent = coupled.export_coupling_state().by_agent
    # all three regimes advanced: the relay swept an orbital arc, the excavator drove off origin.
    assert by_agent["relay"].pose.translation_m.y != 0.0
    assert by_agent["excavator"].pose.translation_m.x > 0.0
    # the named boundary tracked a residual, bounded by the rover's max single-step displacement.
    (residual,) = coupled.residuals
    assert residual.boundary == "dig-site" and residual.agent_id == "excavator"
    assert 0.0 < residual.position_residual_m <= 0.5 * 2.0 + 1e-9


def test_boundary_keeps_each_engines_own_battery_and_mode() -> None:
    mover = mobility_engine_factory(_mobility_excavator("exc"), RngStreams(0))
    digger = granular_engine_factory(_granular_excavator("exc"), RngStreams(0))
    coupled = CoupledEngine(
        {"mover": mover, "digger": digger},
        boundaries=(CouplingBoundary("dig-site", "mover", "digger", ("exc",)),),
    )
    coupled.advance(1.0)
    # only the pose crosses the boundary — the digger keeps its own (granular) battery, not
    # the mover's, so a handoff never silently overwrites unrelated state.
    assert digger.export_coupling_state().by_agent["exc"].battery_soc_j == pytest.approx(400.0)


def test_no_residual_when_a_boundary_agent_is_absent_from_an_engine() -> None:
    mover = mobility_engine_factory(_mobility_excavator("exc"), RngStreams(0))
    digger = granular_engine_factory(_granular_excavator("exc"), RngStreams(0))
    coupled = CoupledEngine(
        {"mover": mover, "digger": digger},
        boundaries=(CouplingBoundary("ghost", "mover", "digger", ("not-here",)),),
    )
    coupled.advance(1.0)
    assert coupled.residuals == ()


def test_residuals_are_empty_before_any_advance() -> None:
    mover = mobility_engine_factory(_mobility_excavator("exc"), RngStreams(0))
    assert CoupledEngine({"mover": mover}).residuals == ()


# --- explicit frame bridging: mismatches fail loudly -----------------------------


def test_cross_frame_boundary_fails_loudly() -> None:
    # producer resolves the agent in the inertial frame, consumer in the body-fixed frame:
    # bridging needs a SPICE rotation Sim does not yet carry, so it must not silently mix.
    mover = mobility_engine_factory(_mobility_excavator("exc", frame=INERTIAL_J2000), RngStreams(0))
    digger = granular_engine_factory(_granular_excavator("exc"), RngStreams(0))
    coupled = CoupledEngine(
        {"mover": mover, "digger": digger},
        boundaries=(CouplingBoundary("dig-site", "mover", "digger", ("exc",)),),
    )
    with pytest.raises(FrameBridgeError, match=r"J2000.*MOON_ME"):
        coupled.advance(1.0)


# --- export deduplication + lifecycle fan-out ------------------------------------


def test_export_deduplicates_a_shared_agent() -> None:
    mover = mobility_engine_factory(_mobility_excavator("exc"), RngStreams(0))
    digger = granular_engine_factory(_granular_excavator("exc"), RngStreams(0))
    coupled = CoupledEngine({"mover": mover, "digger": digger})
    samples = coupled.export_coupling_state().samples
    assert tuple(s.agent_id for s in samples) == ("exc",)  # one sample, first engine wins


def test_apply_actions_and_retire_fan_out_to_sub_engines() -> None:
    scenario = load_scenario(DATA / "heterogeneous.json")
    coupled = coupled_engine_factory()(scenario, RngStreams(11))
    # a mode command addressed to the rover reaches the mobility sub-engine,
    coupled.apply_actions(
        ActionBatch(
            actions=[Action(agent_id="rover", kind=ActionKind.MODE, mode=ModeCommand(mode="drive"))]
        )
    )
    coupled.advance(1.0)
    assert coupled.export_coupling_state().by_agent["rover"].mode == "drive"
    # and retiring an agent drops it from whichever sub-engine owned it.
    coupled.retire(["digger"])
    assert "digger" not in coupled.export_coupling_state().by_agent


def test_actuation_through_the_coupler_retargets_a_rover() -> None:
    scenario = load_scenario(DATA / "heterogeneous.json")
    sim = Simulator(scenario, engine_factory=coupled_engine_factory())
    sim.reset()
    drive_y = Action(
        agent_id="rover",
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="base", control_mode=ControlMode.VELOCITY, setpoint=[0.0, 0.5, 0.0]
        ),
    )
    sim.step(ActionBatch(actions=[drive_y]))
    result = sim.step(ActionBatch(actions=[drive_y]))
    assert result.observations["rover"].self_state.pose.translation_m.y > 0.0


# --- the synthesized composite descriptor ----------------------------------------


def test_descriptor_unions_regimes_frames_and_takes_the_weakest_determinism() -> None:
    relay = Scenario(
        name="relay",
        agents=(AgentSpec(agent_id="relay", frame=INERTIAL_J2000, dynamics=OrbitalDynamics()),),
    )
    digger = _granular_excavator("exc")
    coupled = CoupledEngine(
        {
            "orbital": orbital_engine_factory(relay, RngStreams(0)),
            "granular": granular_engine_factory(digger, RngStreams(0)),
        }
    )
    d = coupled.descriptor
    assert d.regimes == (Regime.PROXIMITY_ORBIT, Regime.SURFACE)  # union, value-sorted
    assert tuple(f.name for f in d.frames) == ("J2000", "MOON_ME")  # union, name-sorted
    # orbital is TOLERANCE, granular is BIT_EXACT -> the composite is the weaker class.
    assert d.determinism_class is DeterminismClass.TOLERANCE
    assert d.fidelity.tier is FidelityTier.MASSMODEL  # coarsest rung


# --- construction-time validation (fail loud) ------------------------------------


def test_rejects_an_empty_sub_engine_set() -> None:
    with pytest.raises(ValueError, match="at least one sub-engine"):
        CoupledEngine({})


def test_rejects_a_boundary_referencing_an_unknown_engine() -> None:
    mover = mobility_engine_factory(_mobility_excavator("exc"), RngStreams(0))
    with pytest.raises(ValueError, match="unknown sub-engine 'ghost'"):
        CoupledEngine(
            {"mover": mover},
            boundaries=(CouplingBoundary("b", "mover", "ghost", ("exc",)),),
        )


def test_rejects_a_non_positive_sub_step_count() -> None:
    mover = mobility_engine_factory(_mobility_excavator("exc"), RngStreams(0))
    with pytest.raises(ValueError, match="must be >= 1"):
        CoupledEngine({"mover": mover}, schedule=CouplingSchedule(substeps={"mover": 0}))


def test_rejects_a_schedule_referencing_an_unknown_engine() -> None:
    mover = mobility_engine_factory(_mobility_excavator("exc"), RngStreams(0))
    with pytest.raises(ValueError, match="schedule references unknown sub-engine 'ghost'"):
        CoupledEngine({"mover": mover}, schedule=CouplingSchedule(substeps={"ghost": 2}))


# --- coupling-state round-trip through the composite -----------------------------


def test_import_coupling_state_fans_out_and_restores_sub_engines() -> None:
    coupled = coupled_engine_factory()(load_scenario(DATA / "heterogeneous.json"), RngStreams(11))
    snapshot = coupled.export_coupling_state()
    coupled.advance(1.0)  # diverge from the snapshot
    assert coupled.export_coupling_state().by_agent["rover"].pose != snapshot.by_agent["rover"].pose
    coupled.import_coupling_state(snapshot)  # fan the snapshot back to every sub-engine
    restored = coupled.export_coupling_state()
    assert restored.by_agent["rover"].pose == snapshot.by_agent["rover"].pose
    assert restored.by_agent["relay"].pose == snapshot.by_agent["relay"].pose
    assert restored.sim_time_s == snapshot.sim_time_s
