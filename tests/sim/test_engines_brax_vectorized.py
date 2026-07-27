"""RM-P1-SIM-04 — the ``jax.vmap`` GPU-batched swarm-scale rollout path.

Thousands of low-fidelity parallel envs stepped as one XLA-compiled batch (sim.md §8), driven by a
Core ``ActionBatch`` and emitting Core ``Observation``\\ s per env — nothing engine-typed crosses
the waist. Exercised on the CPU here (a small 16-64 env batch); a true GPU-device / large-batch
assertion is marked ``gpu`` and deselected in CI. Skips without ``astro-mine-sim[brax]``.
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    ModeCommand,
    Observation,
)
from astro_mine.sim.engines.brax import build_vectorized_rollout
from astro_mine.sim.runtime import AgentSpec, BraxContactDynamics, RngStreams, Scenario


@pytest.fixture(autouse=True)
def _require_brax():
    """Skip every test here unless the JAX stack (``[brax]`` extra) is importable."""
    pytest.importorskip("jax")
    pytest.importorskip("brax")
    pytest.importorskip("mujoco")


def _scenario(*, n_agents: int = 2, jitter: float = 0.0, batch_size: int = 32) -> Scenario:
    agents = tuple(
        AgentSpec(
            agent_id=f"rover-{i}",
            battery_soc_j=1.0e7,
            velocity_mps=(0.1, 0.0, 0.0),
            dynamics=BraxContactDynamics(
                mass_kg=800.0,
                max_speed_mps=1.0,
                max_traction_n=400.0,
                drive_power_w_per_mps=50.0,
                batch_size=batch_size,
                init_speed_jitter_mps=jitter,
            ),
        )
        for i in range(n_agents)
    )
    return Scenario(name="brax-batch", horizon_steps=1, dt_s=0.5, agents=agents)


def _velocity_cmd(agent_id: str, vec: tuple[float, float, float]) -> Action:
    return Action(
        agent_id=agent_id,
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="base", control_mode=ControlMode.VELOCITY, setpoint=list(vec)
        ),
    )


def test_default_env_count_comes_from_batch_size() -> None:
    rollout = build_vectorized_rollout(_scenario(batch_size=48), RngStreams(0))
    assert rollout.n_envs == 48
    assert rollout.n_agents == 2
    assert tuple(rollout.env_indices) == tuple(range(48))


def test_reset_returns_per_env_per_agent_observations() -> None:
    rollout = build_vectorized_rollout(_scenario(n_agents=3), RngStreams(0), n_envs=16)
    observations = rollout.reset()
    assert len(observations) == 16  # one tuple per env
    assert all(len(env_obs) == 3 for env_obs in observations)  # one Observation per agent
    first = observations[0][0]
    assert isinstance(first, Observation)
    assert first.agent_id == "rover-0"


def test_batched_step_has_the_batch_shape_and_advances() -> None:
    rollout = build_vectorized_rollout(_scenario(n_agents=2), RngStreams(0), n_envs=24)
    rollout.reset()
    actions = ActionBatch(
        actions=[
            _velocity_cmd("rover-0", (0.8, 0.0, 0.0)),
            _velocity_cmd("rover-1", (0.0, 0.8, 0.0)),
        ]
    )
    observations = rollout.step(actions)
    assert rollout.positions.shape == (24, 2, 3)  # (envs, agents, xyz)
    assert len(observations) == 24
    # Every env advanced the same commanded rovers (identical envs, jitter off → identical rows).
    assert observations[0][0].self_state.pose.translation_m.x > 0.0


def test_per_env_reproducibility_under_a_fixed_seed() -> None:
    def final_positions():
        rollout = build_vectorized_rollout(_scenario(jitter=0.5), RngStreams(5), n_envs=16)
        rollout.reset()
        actions = ActionBatch(
            actions=[
                _velocity_cmd("rover-0", (0.7, 0.2, 0.0)),
                _velocity_cmd("rover-1", (0.3, 0.6, 0.0)),
            ]
        )
        for _ in range(4):
            rollout.step(actions)
        pos = rollout.positions
        return [[[float(pos[e, a, k]) for k in range(3)] for a in range(2)] for e in range(16)]

    assert final_positions() == final_positions()  # same seed ⇒ identical batch


def test_domain_randomization_varies_envs_within_a_batch() -> None:
    # With seeded per-env jitter the envs are genuinely different rollouts (the point of batching
    # for domain randomization), so at least two envs' first-agent velocities differ.
    rollout = build_vectorized_rollout(_scenario(jitter=1.0), RngStreams(3), n_envs=16)
    observations = rollout.reset()
    velocities = {
        (
            round(env_obs[0].self_state.linear_velocity_mps.x, 6),
            round(env_obs[0].self_state.linear_velocity_mps.y, 6),
        )
        for env_obs in observations
        if env_obs[0].self_state.linear_velocity_mps is not None
    }
    assert len(velocities) > 1


def test_a_shard_of_env_indices_reproduces_the_whole_batch_rows() -> None:
    # Seeding is by GLOBAL env index, so a rollout over indices [8, 16) matches rows [8, 16) of the
    # full run — the property the Ray fan-out relies on.
    full = build_vectorized_rollout(_scenario(jitter=0.5), RngStreams(9), n_envs=16)
    shard = build_vectorized_rollout(_scenario(jitter=0.5), RngStreams(9), env_indices=range(8, 16))
    full_obs = full.reset()
    shard_obs = shard.reset()
    assert shard.n_envs == 8

    def vel(obs):
        v = obs.self_state.linear_velocity_mps
        assert v is not None
        return (round(v.x, 9), round(v.y, 9), round(v.z, 9))

    assert [vel(shard_obs[i][0]) for i in range(8)] == [vel(full_obs[8 + i][0]) for i in range(8)]


def test_mode_command_and_over_cap_velocity_across_the_batch() -> None:
    rollout = build_vectorized_rollout(_scenario(n_agents=2), RngStreams(0), n_envs=8)
    rollout.reset()
    actions = ActionBatch(
        actions=[
            Action(agent_id="rover-0", kind=ActionKind.MODE, mode=ModeCommand(mode="driving")),
            _velocity_cmd("rover-1", (5.0, 0.0, 0.0)),  # above the 1.0 m/s cap → clamp path
            _velocity_cmd("ghost", (1.0, 0.0, 0.0)),  # unowned agent → ignored
        ]
    )
    observations = rollout.step(actions)
    assert observations[0][0].self_state.mode == "driving"
    v = observations[0][1].self_state.linear_velocity_mps
    assert v is not None and (v.x**2 + v.y**2 + v.z**2) ** 0.5 <= 1.0 + 1e-9  # speed-capped


def test_empty_or_missing_brax_agents_raise() -> None:
    kin = Scenario(name="kin", agents=(AgentSpec(agent_id="a"),))
    with pytest.raises(ValueError, match="brax_contact"):
        build_vectorized_rollout(kin, RngStreams(0))
    with pytest.raises(ValueError, match="at least one env"):
        build_vectorized_rollout(_scenario(), RngStreams(0), env_indices=[])


def test_mismatched_brax_params_raise() -> None:
    scenario = Scenario(
        name="mismatch",
        horizon_steps=1,
        dt_s=0.5,
        agents=(
            AgentSpec(
                agent_id="r1",
                battery_soc_j=1.0e7,
                dynamics=BraxContactDynamics(
                    mass_kg=800.0, max_speed_mps=1.0, max_traction_n=400.0
                ),
            ),
            AgentSpec(
                agent_id="r2",
                battery_soc_j=1.0e7,
                dynamics=BraxContactDynamics(
                    mass_kg=900.0, max_speed_mps=1.0, max_traction_n=400.0
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="share one parameter set"):
        build_vectorized_rollout(scenario, RngStreams(0), n_envs=4)


def _roll_positions(n_envs: int, steps: int, device: object | None = None) -> list:
    import jax

    def build_and_roll() -> list:
        rollout = build_vectorized_rollout(_scenario(n_agents=4), RngStreams(0), n_envs=n_envs)
        rollout.reset()
        actions = ActionBatch(
            actions=[_velocity_cmd(f"rover-{i}", (0.5, 0.1, 0.0)) for i in range(4)]
        )
        for _ in range(steps):
            rollout.step(actions)
        pos = rollout.positions
        return [[[float(pos[e, a, k]) for k in range(3)] for a in range(4)] for e in range(n_envs)]

    if device is None:
        return build_and_roll()
    with jax.default_device(device):
        return build_and_roll()


@pytest.mark.gpu
def test_swarm_scale_batch_on_gpu() -> None:
    # The swarm-scale target (sim.md §8): a modest-thousands parallel-env batch stepped on the GPU
    # device, cross-checked against the CPU result within the documented CPU↔GPU tolerance (the
    # descriptor's TOLERANCE rationale — XLA reductions are non-associative / not bit-portable).
    # Deselected in CI (no GPU on runners); on a GPU host `pytest -m gpu` runs it on device.
    import jax

    devices = [d for d in jax.devices() if d.platform != "cpu"]
    if not devices:
        pytest.skip("no GPU device visible to JAX (install jax[cuda12] to verify on device)")

    n_envs = 512  # modest for 8 GB VRAM; the kernel is tiny, so this is trivially on-device
    on_gpu = _roll_positions(n_envs, steps=4, device=devices[0])
    on_cpu = _roll_positions(n_envs, steps=4, device=jax.devices("cpu")[0])
    assert len(on_gpu) == n_envs and len(on_gpu[0]) == 4

    worst = max(
        abs(g - c)
        for env_g, env_c in zip(on_gpu, on_cpu, strict=True)
        for row_g, row_c in zip(env_g, env_c, strict=True)
        for g, c in zip(row_g, row_c, strict=True)
    )
    assert worst <= 1e-6, f"CPU↔GPU rollout disagreed beyond tolerance: {worst}"
