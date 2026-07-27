"""sim.md §3, §6 — the service skin: gRPC ``EnvironmentService`` + the generic Ray-actor wrapper.

The gap this closes: sim.md §3's package layout names a ``service/`` module and §6 names its shape
("as a gRPC EnvironmentService (server-streaming step) wrapped in a Ray actor"), but the package had
no gRPC surface anywhere and no generic Environment-as-actor wrapper — only the Brax tier's
engine-internal fan-out, which parallelizes *inside* one engine rather than exposing environments as
actors for Cloud-level fan-out.

The load-bearing test here is
:func:`test_the_served_environment_honours_cores_environment_contract`: it runs Core's **own**
``check_environment`` against a live served environment, so the served path is *proved* to honour
the declared Environment API — not merely asserted to.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from astro_mine.core.env import Environment, check_environment
from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    Observation,
)
from astro_mine.core.sadf.enums import SensorKind
from astro_mine.core.sadf.model import ObservationModel, ResourceTarget, Sensor
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines import kinematic_engine_factory
from astro_mine.sim.runtime import (
    AgentSpec,
    MobilityDynamics,
    RngStreams,
    Scenario,
    Simulator,
)
from astro_mine.sim.runtime.episode import CORE_INTERFACES
from astro_mine.sim.runtime.provenance import engines_that_ran
from astro_mine.sim.sensors import ReferenceResourceField
from astro_mine.sim.service import (
    EnvironmentActor,
    RemoteEnvironment,
    connect,
    decode_frame,
    encode_frame,
    run_episode_in_process,
    serve,
)

_SPECIES = "water_equivalent_hydrogen"


def _scenario(*, agents: int = 2, horizon: int = 6) -> Scenario:
    return Scenario(
        name="served-anchor",
        seed=11,
        dt_s=60.0,
        horizon_steps=horizon,
        agents=tuple(
            AgentSpec(
                agent_id=f"rover{i}",
                frame=MOON_BODY_FIXED,
                initial_position_m=(float(i), 0.0, 0.0),
                velocity_mps=(0.2, 0.0, 0.0),
                battery_soc_j=1.0e6,
                dynamics=MobilityDynamics(mass_kg=250.0, max_speed_mps=1.0, max_traction_n=200.0),
                sensors=(
                    Sensor(
                        name="neutron",
                        kind=SensorKind.NEUTRON_SPECTROMETER,
                        frame="body",
                        observation_model=ObservationModel(noise_sigma=0.01),
                        resource=ResourceTarget(species=_SPECIES, si_unit="mass_fraction"),
                    ),
                ),
            )
            for i in range(agents)
        ),
    )


@pytest.fixture
def served() -> Iterator[tuple[RemoteEnvironment, list[tuple[int, bytes]]]]:
    """A live gRPC-served environment plus the frames its ``on_frame`` seam pushed out."""
    scenario = _scenario()
    frames: list[tuple[int, bytes]] = []
    server, address = serve(
        lambda: Simulator(scenario, resource_field=ReferenceResourceField()),
        scenario=scenario,
        on_frame=lambda tick, payload: frames.append((tick, payload)),
    )
    channel, env = connect(address)
    try:
        yield env, frames
    finally:
        channel.close()
        server.stop(grace=None)


# --- the FlatBuffers per-tick payload (conventions.md §3) -------------------------


def test_the_per_tick_payload_round_trips_through_flatbuffers() -> None:
    # Protobuf is the control plane; the per-tick sensor/telemetry payload is FlatBuffers, read by
    # pointer offset rather than parsed (conventions.md §3; sim.md §11).
    sim = Simulator(_scenario(), resource_field=ReferenceResourceField())
    observations = sim.reset().observations

    decoded = decode_frame(encode_frame(observations))

    assert set(decoded) == set(observations)
    for agent_id, original in observations.items():
        back = decoded[agent_id]
        assert back.self_state.pose == original.self_state.pose
        assert back.self_state.battery_soc_j == original.self_state.battery_soc_j
        assert back.self_state.mode == original.self_state.mode
        assert back.tick == original.tick and back.sim_time_s == original.sim_time_s
        # The sensor readings survive intact — this is the *telemetry* payload, after all.
        assert [r.values for r in back.sensors] == [r.values for r in original.sensors]
        assert back.sensors[0].resource_species == _SPECIES
        assert back.sensors[0].valid is True


def test_the_payload_encoding_is_deterministic() -> None:
    # A served run must stay as reproducible as an in-process one (conventions.md §11), so the wire
    # encoding cannot introduce nondeterminism (e.g. map ordering).
    observations = Simulator(_scenario()).reset().observations
    assert encode_frame(observations) == encode_frame(observations)


def test_an_empty_frame_encodes_and_decodes() -> None:
    assert decode_frame(encode_frame({})) == {}


# --- the served Environment IS a Core Environment --------------------------------


def test_the_served_environment_honours_cores_environment_contract() -> None:
    # THE acceptance criterion, and the whole point of the skin: the served environment is proved to
    # honour the declared Environment API by Core's *own* conformance utility — the same check the
    # in-process Simulator passes. check_environment drives a seeded multi-step rollout, validates
    # the
    # reset/step shapes and the monotonic agent attrition, applies the Gym/PettingZoo adapters, and
    # enforces determinism by a full-trace hash across two same-seed rollouts.
    scenario = _scenario()
    server, address = serve(lambda: Simulator(scenario), scenario=scenario)
    channel, env = connect(address)
    try:
        assert isinstance(env, Environment)  # structurally a Core Environment
        check_environment(env, seed=3, steps=4)  # ... and behaviourally one
    finally:
        channel.close()
        server.stop(grace=None)


def test_the_client_learns_the_environments_static_facts_from_describe(
    served: tuple[RemoteEnvironment, list[tuple[int, bytes]]],
) -> None:
    # The static description does not ride in every high-rate frame — that separation is the point
    # of
    # the two-plane split.
    env, _frames = served
    assert env.possible_agents == ("rover0", "rover1")
    assert env.core_interfaces == CORE_INTERFACES
    assert env.frame == MOON_BODY_FIXED


def test_a_served_rollout_matches_the_in_process_rollout(
    served: tuple[RemoteEnvironment, list[tuple[int, bytes]]],
) -> None:
    # Same physics, same seed, same answers: the transport is not part of the model.
    env, _frames = served
    local = Simulator(_scenario(), resource_field=ReferenceResourceField())

    remote_obs = env.reset(seed=5).observations
    local_obs = local.reset(seed=5).observations
    _assert_same(remote_obs, local_obs)

    for _ in range(3):
        remote = env.step(ActionBatch())
        local_result = local.step(ActionBatch())
        _assert_same(remote.observations, local_result.observations)
        assert remote.sim_time_s == local_result.sim_time_s
        assert remote.dt_s == local_result.dt_s
        assert dict(remote.terminations) == dict(local_result.terminations)


def _assert_same(a: dict[str, Observation], b: dict[str, Observation]) -> None:
    assert set(a) == set(b)
    for agent_id in a:
        assert a[agent_id].self_state.pose == b[agent_id].self_state.pose
        assert [r.values for r in a[agent_id].sensors] == [r.values for r in b[agent_id].sensors]


def test_actions_reach_the_served_environment(
    served: tuple[RemoteEnvironment, list[tuple[int, bytes]]],
) -> None:
    # The Core ActionBatch crosses the wire and actuates the remote engine — a served environment is
    # driveable, not just observable.
    env, _frames = served
    env.reset(seed=1)
    stop = ActionBatch(
        actions=[
            Action(
                agent_id="rover0",
                kind=ActionKind.ACTUATOR,
                actuator=ActuatorCommand(
                    target="base", control_mode=ControlMode.VELOCITY, setpoint=[0.0, 0.0, 0.0]
                ),
            )
        ]
    )
    for _ in range(4):
        result = env.step(stop)
    velocity = result.observations["rover0"].self_state.linear_velocity_mps
    # The self-state projection carries no velocity (it is proprioceptive), so assert on the pose:
    # rover0 was commanded to stop and rover1 was not, so they must have diverged.
    assert velocity is None
    x0 = result.observations["rover0"].self_state.pose.translation_m.x
    x1 = result.observations["rover1"].self_state.pose.translation_m.x
    assert x1 - 1.0 > x0  # rover1 started 1 m ahead and kept driving; rover0 stopped


def test_step_is_server_streaming(
    served: tuple[RemoteEnvironment, list[tuple[int, bytes]]],
) -> None:
    # sim.md §3, §6: one action batch in, one response per tick out — so a horizon costs one round
    # trip. The client buffers the stream and still hands the caller one tick per step(), so the
    # Environment contract is unchanged whatever the stream depth.
    scenario = _scenario()
    server, address = serve(lambda: Simulator(scenario), scenario=scenario)
    channel, env = connect(address, stream_steps=4)
    try:
        env.reset(seed=2)
        ticks = [env.step(ActionBatch()).sim_time_s for _ in range(4)]
        assert ticks == [60.0, 120.0, 180.0, 240.0]  # four ticks from one RPC
    finally:
        channel.close()
        server.stop(grace=None)


def test_a_stream_depth_below_one_is_rejected() -> None:
    scenario = _scenario()
    server, address = serve(lambda: Simulator(scenario), scenario=scenario)
    channel = None
    try:
        with pytest.raises(ValueError, match="stream_steps must be >= 1"):
            channel, _env = connect(address, stream_steps=0)
    finally:
        if channel is not None:  # pragma: no cover  (connect raises before returning a channel)
            channel.close()
        server.stop(grace=None)


def test_live_frames_stream_through_the_on_frame_seam(
    served: tuple[RemoteEnvironment, list[tuple[int, bytes]]],
) -> None:
    # sim.md §6 "live frames stream to View": the on_frame seam spans the network boundary, so a
    # served environment pushes each frame to a consumer as it is produced — the same idea that
    # keeps
    # headless and interactive one runtime in-process (sim.md §2.6).
    env, frames = served
    env.reset(seed=4)
    for _ in range(3):
        env.step(ActionBatch())

    assert [tick for tick, _payload in frames] == [0, 1, 2, 3]  # the reset frame plus one per step
    # Each pushed frame is a real, decodable observation frame — a live consumer can render it.
    _tick, payload = frames[-1]
    decoded = decode_frame(payload)
    assert set(decoded) == {"rover0", "rover1"}


def test_a_malformed_action_batch_is_rejected_by_the_service() -> None:
    import grpc

    scenario = _scenario()
    server, address = serve(lambda: Simulator(scenario), scenario=scenario)
    channel, env = connect(address)
    try:
        env.reset()
        from astro_mine.sim.service._proto import environment_pb2 as pb
        from astro_mine.sim.service._proto import environment_pb2_grpc as pb_grpc

        stub = pb_grpc.EnvironmentServiceStub(channel)
        with pytest.raises(grpc.RpcError) as excinfo:
            list(stub.Step(pb.StepRequest(action_batch_json="{not json", steps=1)))
        assert excinfo.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    finally:
        channel.close()
        server.stop(grace=None)


# --- the generic Environment-as-Ray-actor wrapper (sim.md §6, §7) -----------------


def test_the_environment_actor_wraps_any_environment_not_just_the_brax_tier() -> None:
    # The acceptance criterion: a *generic* Ray-actor wrapper, distinct from the Brax tier's
    # internal
    # fan-out (which parallelizes inside one vectorizable engine). This one exposes any Core
    # Environment — here a reduced-order mobility scenario, which the Brax fan-out cannot touch.
    scenario = _scenario()
    actor = EnvironmentActor(scenario.model_dump_json(), seed=9)
    seed, frames = actor.run()
    assert seed == 9
    assert len(frames) == scenario.horizon_steps + 1  # the reset frame plus one per step
    assert set(frames[0]) == {"rover0", "rover1"}


def test_the_actor_ships_only_core_payloads_across_the_boundary() -> None:
    # A Ray boundary is a serialization boundary: the actor is built from a JSON scenario and
    # returns
    # a JSON string, so no engine object, JAX array, or gRPC channel is ever pickled.
    import json

    scenario = _scenario()
    payload = EnvironmentActor(scenario.model_dump_json(), seed=9).run_json()
    decoded = json.loads(payload)
    assert decoded["seed"] == 9
    assert isinstance(decoded["frames"], list)


def test_a_fanned_out_sweep_reassembles_to_the_in_process_oracle() -> None:
    # The equivalence gate the live-cluster fan-out is checked against: each episode is a pure
    # function of (scenario, seed), so a sweep dispatched across actors must return exactly what the
    # single process does — in seed order. The dispatch itself needs live Ray workers (`-m ray`);
    # the
    # actor and the seed-order reassembly it composes are exercised here.
    scenario = _scenario(horizon=3)
    seeds = [1, 2, 3]
    # Pinned to the kinematic engine on purpose. The "seeds really diverge" assertion below needs
    # a scenario whose *dynamics* are stochastic, and the kinematic reference engine's seeded
    # +/-0.01 m jitter is the only such source here: the neutron sensor would supply noise, but the
    # actor threads no resource field, so it renders `valid=False` and contributes nothing.
    #
    # Before #65 this test passed by accident — the default was kinematic no matter what a scenario
    # declared, so these `MobilityDynamics` rovers were silently integrated by the jitter engine.
    # With the coupler as the default they run real mobility, which is deterministic given the same
    # initial conditions, so every seed produces an identical rollout. That is correct physics and
    # a useless divergence test, so the engine is now named rather than inherited.
    engine_factory = kinematic_engine_factory
    results = run_episode_in_process(scenario, seeds, engine_factory=engine_factory)

    assert [seed for seed, _frames in results] == seeds  # seed order preserved
    scenario_json = scenario.model_dump_json()
    for seed, frames in results:
        again = EnvironmentActor(scenario_json, seed, engine_factory=engine_factory).run()[1]
        assert frames == again  # a pure function of (scenario, seed)
    # Different seeds really do diverge, so the sweep is not trivially uniform.
    assert results[0][1] != results[1][1]


def test_a_declared_mobility_scenario_is_stepped_by_the_mobility_engine() -> None:
    """The #65 regression: the served path must run the dynamics the scenario declares.

    Its sibling above pins the kinematic engine deliberately; this one asserts the *default* is no
    longer kinematic-for-everything.
    """
    scenario = _scenario(horizon=2)
    _seed, frames = EnvironmentActor(scenario.model_dump_json(), 11).run()

    assert frames  # the run produced something
    engine = Simulator(scenario)
    engine.reset(seed=11)
    assert [d.name for d in engines_that_ran(engine.engine)] == ["astro-mine.sim.mobility"]


def test_the_actor_honours_an_injected_engine_factory() -> None:
    # A sweep can fan out *any* tier, not just the vectorizable ones — the engine factory is
    # injected.
    from astro_mine.sim.engines import kinematic_engine_factory

    scenario = _scenario(horizon=2)
    actor = EnvironmentActor(
        scenario.model_dump_json(), seed=1, engine_factory=kinematic_engine_factory
    )
    _seed, frames = actor.run()
    assert len(frames) == 3


def test_the_actor_stops_early_when_every_agent_terminates() -> None:
    # Core's attrition contract: the active set only shrinks, and the actor must not keep stepping
    # an
    # empty environment.
    scenario = Scenario(
        name="dying",
        horizon_steps=8,
        agents=(AgentSpec(agent_id="a", battery_soc_j=0.0, battery_floor_j=0.0),),
    )
    _seed, frames = EnvironmentActor(scenario.model_dump_json(), seed=0).run()
    assert len(frames) < scenario.horizon_steps + 1


# --- the in-process library path is unaffected -----------------------------------


def test_the_library_path_never_imports_grpc() -> None:
    # `service/` is additive (the acceptance criterion): importing and driving Sim in-process must
    # not
    # pull in gRPC. Checked in a clean subprocess, because the test process has already imported it.
    code = (
        "import sys; import astro_mine.sim.runtime as r;"
        "s = r.Scenario(name='x', agents=(r.AgentSpec(agent_id='a'),));"
        "r.run_episode(s);"
        "assert 'grpc' not in sys.modules, sorted(m for m in sys.modules if 'grpc' in m);"
        "assert 'astro_mine.sim.service' not in sys.modules;"
        "print('CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_an_unknown_service_export_raises_attribute_error() -> None:
    import astro_mine.sim.service as service

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = service.nonexistent  # type: ignore[attr-defined]


def test_the_generated_protobuf_stubs_match_the_proto() -> None:
    # The generated modules are committed (so the wheel installs with no protoc in the toolchain);
    # this gates them against drift from environment.proto — regenerate with scripts/gen_proto.py.
    from astro_mine.sim.service._proto import environment_pb2 as pb

    descriptor = pb.DESCRIPTOR
    service = descriptor.services_by_name["EnvironmentService"]
    assert {m.name for m in service.methods} == {"Describe", "Reset", "Step"}
    step = service.methods_by_name["Step"]
    assert step.server_streaming is True  # sim.md §3, §6: server-streaming step
    assert step.client_streaming is False


def test_the_rng_streams_import_is_available_for_actor_seeding() -> None:
    # A guard on the seeding contract the actor relies on: same root seed ⇒ same streams.
    assert RngStreams(3).stream("a").random() == RngStreams(3).stream("a").random()
