"""The GPU-vectorized batched rollout kernel (RM-P1-LEARN-04; learn.md §8) — needs [rllib].

The gap this closes: ``VectorExecutor`` used to "batch" by running :class:`LocalExecutor` in a
sequential CPU loop. It now runs a **genuinely batched kernel** behind the same seam — one world
step and one policy forward per agent per tick, for the *whole* batch of env copies — with the
sequential loop kept as the graceful fallback.

Three things must hold, and each is a test below:

1. **It is actually batched.** Per tick the policy is called *once* per agent for all
   ``num_envs`` copies (not ``num_envs`` times), and the world steps once. Counting the calls is
   the only way to prove "not a sequential loop".
2. **It is still a Rollout.** The trainer's ``_update`` is unchanged code, so the batched kernel
   must return the identical shape — and, critically, keep each env copy's trajectory
   **contiguous**, because GAE bootstraps across adjacent steps and interleaving the copies
   would silently corrupt every advantage.
3. **It degrades gracefully.** No batched world, no JAX, or a non-batchable step ⇒ the
   sequential CPU loop, because learn.md §7 says tier 1 MUST always work.

The batched-kernel tests run against a **fake** BatchedWorld (no JAX needed — it is the
*protocol* Sim's Brax/MJX GPU tier will satisfy). The JAX reference world is exercised on its own
below, and the ``gpu``-marked test asserts it is resident on a real accelerator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from astro_mine.core.env.model import AgentId
from astro_mine.learn import TrainConfig, default_registry, make_swarm_env
from astro_mine.learn.algos.policy import flat_obs_dim
from astro_mine.learn.envs.vector import (
    BatchedObservation,
    BatchedTransition,
    BatchedWorld,
    JaxBatchedWorld,
    VectorExecutor,
    accelerator_devices,
    batched_rollout,
    jax_available,
    jax_batched_world_factory,
    peers_from_mask,
)
from astro_mine.learn.train import LocalExecutor, StepDecision
from tests.learn.fakes import AGENTS, FakeSwarmWorld, build_assets


def _env_factory():
    return make_swarm_env(FakeSwarmWorld(horizon=64), build_assets())


#: Derived from the real SwarmEnv, never hardcoded: a BatchedWorld's whole contract is that it
#: mirrors the CPU tier's static shape exactly, so the fake must too (or the nets would not fit).
_ENV = _env_factory()
_OBS_DIM = {a: flat_obs_dim(spec.observation_space) for a, spec in _ENV.agent_specs.items()}
_STATE_DIM = int(np.asarray(_ENV.state()).shape[0])


class FakeBatchedWorld:
    """A conformant :class:`BatchedWorld` with no JAX — the protocol Sim's GPU tier satisfies.

    Deliberately trivial dynamics (a tick counter): the point is the *seam*, not the physics.
    It counts its ``step_batch`` calls so a test can prove the kernel steps the whole batch once
    per tick rather than once per copy."""

    def __init__(self, *, num_envs: int = 4, horizon: int = 6) -> None:
        self._num_envs = num_envs
        self._horizon = horizon
        self._tick = 0
        self.step_calls = 0
        self.seeds: list[int] = []

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def possible_agents(self) -> tuple[AgentId, ...]:
        return AGENTS

    @property
    def agent_obs_dim(self) -> Mapping[AgentId, int]:
        return dict(_OBS_DIM)

    @property
    def state_dim(self) -> int:
        return _STATE_DIM

    def _render(self) -> dict[str, Any]:
        live = np.full(self._num_envs, self._tick < self._horizon, dtype=np.bool_)
        # Every agent reaches every other (a fully connected channel) — so peers_from_mask has
        # something non-trivial to decode.
        reach = np.tile(
            (1.0 - np.eye(len(AGENTS), dtype=np.float32))[np.newaxis, ...],
            (self._num_envs, 1, 1),
        )
        return {
            "obs": {
                a: np.full((self._num_envs, _OBS_DIM[a]), float(self._tick), dtype=np.float32)
                + np.arange(self._num_envs, dtype=np.float32)[:, None]
                for a in AGENTS
            },
            "state": np.full((self._num_envs, _STATE_DIM), float(self._tick), dtype=np.float32),
            "live": {a: live for a in AGENTS},
            "reach": reach,
        }

    def reset_batch(self, seeds: Sequence[int]) -> BatchedObservation:
        self.seeds = list(seeds)
        self._tick = 0
        return BatchedObservation(**self._render())

    def step_batch(
        self, actions: Mapping[AgentId, Sequence[Mapping[str, Any]]]
    ) -> BatchedTransition:
        self.step_calls += 1
        # Every agent must have been handed one action per env copy.
        for agent in AGENTS:
            assert len(actions[agent]) == self._num_envs
        self._tick += 1
        rendered = self._render()
        done = np.full(self._num_envs, self._tick >= self._horizon, dtype=np.bool_)
        return BatchedTransition(
            **rendered,
            reward={
                a: np.full(self._num_envs, -0.01 * self._tick, dtype=np.float32) for a in AGENTS
            },
            done={a: done for a in AGENTS},
        )


class _CountingStep:
    """A BatchedStep that records how many times it was called and with what batch width."""

    def __init__(self) -> None:
        self.batch_calls = 0
        self.widths: list[int] = []
        self.saw_reach: list[bool] = []

    def __call__(self, flat_obs):  # the sequential (AgentStepFn) path
        return {a: StepDecision(action_sample={"kind": 0, "mode": 0}) for a in flat_obs}

    def batch(self, flat_obs, reach=None):
        self.batch_calls += 1
        self.widths.append(int(next(iter(flat_obs.values())).shape[0]))
        self.saw_reach.append(reach is not None)
        rows = self.widths[-1]
        return {
            a: [StepDecision(action_sample={"kind": 0, "mode": 0}) for _ in range(rows)]
            for a in flat_obs
        }


def test_fake_world_satisfies_the_batched_world_protocol() -> None:
    assert isinstance(FakeBatchedWorld(), BatchedWorld)


# --- 1. it is actually batched -------------------------------------------------------


def test_the_kernel_decides_once_per_tick_for_the_whole_batch() -> None:
    world = FakeBatchedWorld(num_envs=8, horizon=5)
    step = _CountingStep()
    rollout = batched_rollout(world, step, steps=5, seed=0)

    # THE property: 5 ticks ⇒ 5 policy calls and 5 world steps, NOT 5 x 8 (a sequential loop).
    assert step.batch_calls == 5
    assert world.step_calls == 5
    assert step.widths == [8] * 5  # each call saw the whole batch of env copies
    assert all(step.saw_reach)  # the batched world's reach mask reaches the policy
    # ... and it collected 8 copies' worth of experience: 5 ticks x 8 copies x 3 agents.
    assert rollout.env_steps() == 5 * 8 * 3


def test_batched_rollout_returns_the_ordinary_rollout_shape() -> None:
    world = FakeBatchedWorld(num_envs=4, horizon=4)
    rollout = batched_rollout(world, _CountingStep(), steps=4, seed=1)
    assert rollout.possible_agents == AGENTS
    assert rollout.agent_obs_dim == _OBS_DIM
    assert rollout.state_dim == _STATE_DIM
    first = rollout.steps[0]
    assert set(first.obs) == set(AGENTS)
    assert first.obs["rover"].shape == (_OBS_DIM["rover"],)  # unstacked to ONE env copy
    assert first.state.shape == (_STATE_DIM,)
    # The batched world's (n, n) mask decodes back to the tuple form the CPU tier records.
    assert set(first.reach["rover"]) == {"digger", "relay"}


def test_each_env_copy_trajectory_stays_contiguous() -> None:
    # GAE walks a trajectory in order and bootstraps across adjacent steps. If the copies were
    # interleaved, every advantage estimate would be silently wrong. The obs encode the copy
    # index (+arange), so a copy's steps must appear as one unbroken run.
    world = FakeBatchedWorld(num_envs=3, horizon=4)
    rollout = batched_rollout(world, _CountingStep(), steps=4, seed=0)
    assert len(rollout.steps) == 3 * 4
    # rover's obs = tick + copy_index; ticks 0..3 within a copy, copies concatenated.
    copies = [step.obs["rover"][0] - step.state[0] for step in rollout.steps]
    assert copies == [0.0] * 4 + [1.0] * 4 + [2.0] * 4


def test_copy_zero_uses_the_run_seed_verbatim() -> None:
    world = FakeBatchedWorld(num_envs=4)
    batched_rollout(world, _CountingStep(), steps=2, seed=17)
    assert world.seeds[0] == 17  # topology-fixed reproducibility (derive_seeds)
    assert len(world.seeds) == 4


def test_peers_from_mask_decodes_a_reach_row() -> None:
    mask = np.zeros((3, 3), dtype=np.float32)
    mask[AGENTS.index("rover"), AGENTS.index("relay")] = 1.0
    assert peers_from_mask(mask, "rover", AGENTS) == ("relay",)
    assert peers_from_mask(mask, "digger", AGENTS) == ()


# --- 2. the trainers drive it --------------------------------------------------------


@pytest.mark.parametrize("tag", ["ippo", "mappo", "qmix", "comms_ppo"])
def test_every_baseline_step_is_batchable_and_trains_on_the_batched_tier(tag: str) -> None:
    from astro_mine.learn.train.executor import BatchedStep

    config = TrainConfig(seed=4, iterations=1, rollout_steps=4, hidden_sizes=(16, 16))
    executor = VectorExecutor(
        _env_factory,
        num_envs=4,
        batched_world=lambda: FakeBatchedWorld(num_envs=4, horizon=8),
        backend="jax",
    )
    trainer = default_registry().get(tag).make_trainer(_env_factory(), config, executor=executor)
    assert isinstance(trainer.rollout_step, BatchedStep)
    metrics = trainer.train_iteration()
    # The trainer's _update is UNCHANGED code and consumes the batched Rollout as-is.
    assert metrics["env_steps"] == 4 * 4 * 3


def test_batched_and_sequential_agree_on_a_batch_of_one() -> None:
    # A batch of one must draw the same actions as the sequential path: act_batch consumes the
    # seeded generator in the same order as act. (Different worlds, so compare the DECISIONS.)
    from astro_mine.learn.algos import TrainConfig as TC

    config = TC(seed=9, hidden_sizes=(16, 16))
    step_a = default_registry().get("ippo").make_trainer(_env_factory(), config).rollout_step
    step_b = default_registry().get("ippo").make_trainer(_env_factory(), config).rollout_step
    obs = {a: np.full((1, _OBS_DIM[a]), 0.3, dtype=np.float32) for a in AGENTS}

    sequential = step_a({a: row[0] for a, row in obs.items()})
    batched = step_b.batch(obs)
    for agent in AGENTS:
        assert batched[agent][0].action_sample["kind"] == sequential[agent].action_sample["kind"]
        assert batched[agent][0].log_prob == pytest.approx(sequential[agent].log_prob)


# --- 3. it degrades gracefully -------------------------------------------------------


def test_no_batched_world_falls_back_to_the_sequential_loop() -> None:
    executor = VectorExecutor(_env_factory, num_envs=2)
    assert executor.backend == "cpu"
    assert executor.batched_world is None


def test_a_missing_jax_extra_falls_back_rather_than_exploding() -> None:
    # auto: an ImportError from the batched world (no [jax] installed) means "no accelerator
    # here" — the tier-1 workstation MUST still train (learn.md §7).
    def missing_jax():
        raise ImportError("No module named 'jax'")

    executor = VectorExecutor(_env_factory, num_envs=2, batched_world=missing_jax)
    assert executor.backend == "cpu"


def test_explicitly_requesting_jax_fails_loudly() -> None:
    # An explicit backend='jax' must NOT silently run the slow path.
    def missing_jax():
        raise ImportError("No module named 'jax'")

    with pytest.raises(ImportError, match=r"\[jax\]"):
        VectorExecutor(_env_factory, num_envs=2, batched_world=missing_jax, backend="jax")
    with pytest.raises(ValueError, match="batched_world"):
        VectorExecutor(_env_factory, num_envs=2, backend="jax")


def test_a_non_batchable_step_uses_the_sequential_loop() -> None:
    # A bare callable cannot be run as a batch; degrade rather than fail.
    def plain_step(flat_obs):
        return {a: StepDecision(action_sample={"kind": 0, "mode": 0}) for a in flat_obs}

    executor = VectorExecutor(
        _env_factory, num_envs=2, batched_world=lambda: FakeBatchedWorld(num_envs=2)
    )
    assert executor.backend == "jax"
    rollout = executor.rollout(_env_factory(), plain_step, steps=4, seed=0)
    # It ran the CPU fallback against the real SwarmEnv (whose state_dim is the env's, not the
    # fake batched world's), so the fallback really was taken.
    assert rollout.possible_agents == AGENTS


def test_sequential_fallback_is_byte_identical_to_local_for_one_copy() -> None:
    def plain_step(flat_obs):
        return {a: StepDecision(action_sample={"kind": 0, "mode": 0}) for a in flat_obs}

    def signature(rollout):
        return [(sorted(s.obs), s.state.tolist()) for s in rollout.steps]

    local = LocalExecutor().rollout(_env_factory(), plain_step, steps=8, seed=4)
    vector = VectorExecutor(_env_factory, num_envs=1, backend="cpu").rollout(
        _env_factory(), plain_step, steps=8, seed=4
    )
    assert signature(local) == signature(vector)


# --- the JAX reference world ---------------------------------------------------------

jax_only = pytest.mark.skipif(not jax_available(), reason="needs the optional [jax] extra")


@jax_only
def test_jax_world_is_a_conformant_batched_world() -> None:
    world = JaxBatchedWorld.from_swarm_env(_env_factory(), num_envs=4, horizon=8)
    assert isinstance(world, BatchedWorld)
    # It mirrors the CPU SwarmEnv's static shape exactly — that is what makes a policy
    # tensor-compatible across the two fidelity tiers.
    env = _env_factory()
    assert world.possible_agents == tuple(env.possible_agents)
    assert world.state_dim == int(np.asarray(env.state()).shape[0])

    batch = world.reset_batch([1, 2, 3, 4])
    assert batch.obs["rover"].shape == (4, world.agent_obs_dim["rover"])
    assert batch.reach.shape == (4, 3, 3)
    assert np.allclose(np.trace(batch.reach, axis1=1, axis2=2), 0.0)  # no self-links


@jax_only
def test_jax_world_drives_a_real_training_iteration() -> None:
    config = TrainConfig(seed=1, iterations=1, rollout_steps=6, hidden_sizes=(16, 16))
    executor = VectorExecutor(
        _env_factory,
        num_envs=8,
        batched_world=jax_batched_world_factory(_env_factory, num_envs=8, horizon=16),
        backend="jax",
    )
    assert executor.backend == "jax"
    trainer = (
        default_registry().get("mappo").make_trainer(_env_factory(), config, executor=executor)
    )
    metrics = trainer.train_iteration()
    assert metrics["env_steps"] == 6 * 8 * 3  # 6 ticks x 8 GPU-resident copies x 3 agents


@jax_only
def test_jax_world_rollout_is_reproducible_for_a_fixed_topology() -> None:
    def run():
        world = JaxBatchedWorld.from_swarm_env(_env_factory(), num_envs=3, horizon=8)
        return [
            (step.obs["rover"].tolist(), step.reward["rover"])
            for step in batched_rollout(world, _CountingStep(), steps=5, seed=7).steps
        ]

    assert run() == run()


@pytest.mark.gpu
def test_batched_kernel_runs_on_a_real_accelerator() -> None:
    # The `gpu`-marked test the gap called for (there were zero). Deselected in CI (-m 'not gpu');
    # on a CUDA box it proves the batch is GPU-RESIDENT and that the kernel — the same jit/vmap
    # XLA program CI runs on CPU — actually executes there.
    pytest.importorskip("jax")
    accelerators = accelerator_devices()
    if not accelerators:
        pytest.skip("no JAX accelerator visible (install jax[cuda12] on a GPU host)")

    world = JaxBatchedWorld.from_swarm_env(_env_factory(), num_envs=256, horizon=32)
    assert world.device_platform in {"gpu", "tpu"}

    step = _CountingStep()
    rollout = batched_rollout(world, step, steps=16, seed=0)
    assert step.batch_calls == 16  # one forward per tick for all 256 copies
    assert rollout.env_steps() == 16 * 256 * 3

    # And the throughput claim RM-P1-LEARN-04 is judged on, measured on the accelerator.
    from astro_mine.learn.envs.vector.benchmark import baseline_step, rollout_throughput

    result = rollout_throughput(_env_factory, baseline_step(_env_factory), num_envs=256, steps=32)
    assert result.batched is not None
    assert result.batched.device in {"gpu", "tpu"}
    assert result.batched.speedup_vs_sequential > 1.0
