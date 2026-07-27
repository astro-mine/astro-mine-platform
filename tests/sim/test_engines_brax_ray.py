"""RM-P1-SIM-04 — the Ray fan-out of the vectorized rollout (sim.md §6, §7).

Shards the N-env batch across :class:`ray.remote` actors and aggregates back to one batch. The
sharder, the aggregation, and the actor's in-process rollout are ordinary Python — unit-tested
**without a cluster** (they run in CI), including the global-index equivalence property the fan-out
relies on. The **live Ray-cluster** dispatch is marked ``ray`` and **deselected in CI** (GitHub
runners can't spawn Ray worker processes — analogous to the ``gpu`` test), and is verified locally;
on Cloud the actors land on KubeRay GPU workers (RM-P1-CLOUD-01). Skips without the JAX stack.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Action, ActionBatch, ActuatorCommand
from astro_mine.sim.runtime import AgentSpec, BraxContactDynamics, RngStreams, Scenario

# _ray imports the JAX batch kernel at module load, so gate the whole file on the [brax] stack.
pytest.importorskip("jax")
pytest.importorskip("brax")
pytest.importorskip("mujoco")

from astro_mine.sim.engines.brax import _ray
from tests.sim._equivalence import assert_shards_match_oracle


def _scenario(*, jitter: float = 0.5, batch_size: int = 16, n_agents: int = 2) -> Scenario:
    agents = tuple(
        AgentSpec(
            agent_id=f"rover-{i}",
            battery_soc_j=1.0e7,
            velocity_mps=(0.1, 0.0, 0.0),
            dynamics=BraxContactDynamics(
                mass_kg=800.0,
                max_speed_mps=1.0,
                max_traction_n=400.0,
                batch_size=batch_size,
                init_speed_jitter_mps=jitter,
            ),
        )
        for i in range(n_agents)
    )
    return Scenario(name="brax-ray", horizon_steps=1, dt_s=0.5, agents=agents)


def _actions() -> ActionBatch:
    return ActionBatch(
        actions=[
            Action(
                agent_id=f"rover-{i}",
                kind=ActionKind.ACTUATOR,
                actuator=ActuatorCommand(
                    target="base", control_mode=ControlMode.VELOCITY, setpoint=[0.7, 0.2, 0.0]
                ),
            )
            for i in range(2)
        ]
    )


# -- pure driver-side sharding / aggregation (no Ray cluster) --------------------------------


def test_shard_ranges_partition_the_index_range_completely() -> None:
    assert _ray._shard_ranges(16, 4) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    assert _ray._shard_ranges(10, 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]  # remainder up front
    assert _ray._shard_ranges(3, 5) == [[0], [1], [2]]  # more shards than envs → clamp


def test_aggregate_orders_rows_by_global_env_index() -> None:
    shard_b: _ray.ShardResult = ([[[1.0, 0.0, 0.0]], [[1.1, 0.0, 0.0]]], [2, 3])
    shard_a: _ray.ShardResult = ([[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]], [0, 1])
    assert _ray._aggregate([shard_b, shard_a], total=4) == [
        [[0.0, 0.0, 0.0]],
        [[0.5, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[1.1, 0.0, 0.0]],
    ]


def test_fan_out_needs_a_brax_agent() -> None:
    kin = Scenario(name="kin", agents=(AgentSpec(agent_id="a"),))
    with pytest.raises(ValueError, match="brax_contact"):
        _ray.fan_out(kin, RngStreams(0), actions=ActionBatch(), steps=1)


# -- the actor's in-process rollout: the equivalence gate WITHOUT a cluster (JAX, no Ray) ------


def test_actor_shard_matches_the_in_process_batch_rows() -> None:
    # Global-index seeding means a _RolloutActor over env indices [8, 16) yields rows [8, 16) of
    # the whole-batch in-process run — the property the Ray fan-out relies on, verified in-process
    # so CI covers the actor / rollout / aggregation without spawning Ray workers.
    #
    # Matched numerically, not bitwise: the reference vmaps over 16 envs and the shard over 8, and
    # XLA's reduction order follows the batch shape (astro-mine-sim#46; tests/_equivalence.py).
    # The *indices* are exact — those are integers, and a wrong one is a sharding bug.
    scenario = _scenario(batch_size=16)
    actions = _actions()
    reference = _ray.run_in_process(scenario, RngStreams(4), actions=actions, steps=3, n_envs=16)

    actor = _ray._RolloutActor(scenario.model_dump_json(), 4, list(range(8, 16)))
    positions, indices = actor.rollout(actions.model_dump_json(), 3)
    assert indices == list(range(8, 16))
    assert_shards_match_oracle(positions, reference[8:16], what="actor shard [8,16) vs batch rows")

    # Aggregating two in-process shards reproduces the whole batch (the fan-out's pure core).
    lo = _ray._RolloutActor(scenario.model_dump_json(), 4, list(range(0, 8))).rollout(
        actions.model_dump_json(), 3
    )
    aggregated = _ray._aggregate([lo, (positions, indices)], total=16)
    assert_shards_match_oracle(aggregated, reference, what="2x8 shards vs 16-env oracle")


# -- live Ray cluster: dispatch equivalence (deselected in CI; verified locally) --------------


@pytest.fixture(scope="module")
def ray_local() -> Iterator[object]:
    """A tiny CPU-only local Ray cluster (2 workers, no dashboard); shut down on teardown.

    Ray removed ``local_mode`` (2.40+), so the actors run in real worker processes — which GitHub
    runners can't spawn (hence the ``ray`` marker is deselected in CI). The uv auto-runtime-env is
    disabled so workers just inherit this venv (fast, offline)."""
    ray = pytest.importorskip("ray")
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        configure_logging=False,
    )
    try:
        yield ray
    finally:
        ray.shutdown()


@pytest.mark.ray
def test_ray_fan_out_matches_the_in_process_runner(ray_local: object) -> None:
    scenario = _scenario(batch_size=16)
    actions = _actions()
    reference = _ray.run_in_process(scenario, RngStreams(4), actions=actions, steps=5)
    sharded = _ray.fan_out(scenario, RngStreams(4), actions=actions, steps=5, num_shards=4)
    assert len(sharded) == 16
    # Global-index seeding ⇒ shards reassemble to the batch. Numerically, not bitwise: 4 shards
    # of 4 envs vmap differently than one batch of 16 (astro-mine-sim#46).
    assert_shards_match_oracle(sharded, reference, what="Ray fan-out (4 shards vs 16-env oracle)")


@pytest.mark.ray
def test_ray_fan_out_public_alias_agrees(ray_local: object) -> None:
    from astro_mine.sim.engines.brax import fan_out as public_fan_out

    scenario = _scenario(batch_size=12)
    actions = _actions()
    reference = _ray.run_in_process(scenario, RngStreams(1), actions=actions, steps=3)
    sharded = public_fan_out(scenario, RngStreams(1), actions=actions, steps=3, num_shards=3)
    assert_shards_match_oracle(sharded, reference, what="public fan_out (3 shards vs 12 envs)")
