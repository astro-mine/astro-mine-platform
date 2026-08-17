# SPDX-License-Identifier: Apache-2.0
"""Rollout collection behind a single executor seam (RM-P1-LEARN-03; LEARN-04 seam).

"Library first, cluster second" (learn.md §2.1): the distributed path is the *same* code
with a different **executor**, never a fork. Every baseline collects experience through a
:class:`RolloutExecutor`, so RM-P1-LEARN-04 swaps the in-process :class:`LocalExecutor`
(tier 1 — a single workstation, no cloud) for a KubeRay-backed executor **without touching
any algorithm code**. This module is deliberately Torch-free: it steps the
:class:`~astro_mine.learn.envs.SwarmEnv`, records the per-agent transition (flat obs,
decision, reward, done, reachable peers) and the global ``state()`` for CTDE, and hands the
numpy batch back to the (Torch-side) trainer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.learn.algos.policy import flat_obs_dim, flatten_obs

if TYPE_CHECKING:
    from astro_mine.learn.envs import SwarmEnv
    from astro_mine.learn.envs.adapter.encode import ObsSample

__all__ = [
    "AgentStepFn",
    "BatchedStep",
    "BroadcastableStep",
    "CommsAwareStep",
    "EnvFactory",
    "KubeRayExecutor",
    "LocalExecutor",
    "RewardFn",
    "Rollout",
    "RolloutExecutor",
    "RolloutStep",
    "StepDecision",
    "WorkerBatch",
    "WorkerResult",
    "default_reward_fn",
    "derive_seeds",
    "fan_out_rollout",
    "in_process_worker_batch",
]

#: Shape a deterministic reward for a reward-free env (FakeSwarmWorld is reward-free by
#: default; core.md §3). Maps (agent, its observation sample) → a scalar reward. Recorded in
#: provenance so the shaped run stays reproducible.
RewardFn = Callable[[AgentId, "ObsSample"], float]


def derive_seeds(seed: int, count: int) -> list[int]:
    """Per-slice seeds for a topology-fixed batched/distributed rollout.

    Slice 0 uses ``seed`` **verbatim** — so a single-slice run (one KubeRay worker, one
    vector env) is byte-identical to :class:`LocalExecutor` — while each extra slice gets a
    decorrelated draw mixed through :class:`numpy.random.SeedSequence` (the salting pattern
    ``CommsModel.reset`` uses), so a fixed ``(count, seed)`` reproduces run-to-run. Shared by
    :class:`KubeRayExecutor` (distributed) and the vector executor (in-process batch)."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    seeds = [int(seed)]
    for index in range(1, count):
        seeds.append(int(np.random.SeedSequence([int(seed), index]).generate_state(1)[0]))
    return seeds


def default_reward_fn(_agent: AgentId, obs: ObsSample) -> float:
    """A deterministic shaping reward for a reward-free env (FakeSwarmWorld; core.md §3).

    A small negative distance-to-origin read off the agent's ``self_state`` pose — a pure,
    reproducible function of the observation, so the smoke/determinism runs have a
    non-trivial-but-deterministic signal. Torch-free (numpy only) so the eval harness and the
    contract path can use it without the ``[rllib]`` extra.

    **Reward shaping is the consumer's job in every case, including against a real simulator.**
    A Sim-backed env leaves ``StepResult.rewards`` empty too — Sim renders physics, not training
    signal — so this function (or one a caller supplies) is what produces the return, not only on
    the fake. What differs is whether the shaped landscape is *policy-sensitive*: on
    ``FakeSwarmWorld`` the dynamics are a pure function of the tick, so actions cannot move the
    pose this reads and the landscape is flat with respect to the policy *by construction*;
    against an env whose actions move the agent it at least varies with what the policy does.
    Whether that variation is enough to *learn* from is a property of the reward and the training
    budget, not of this function — it is a smoke signal, not a training objective."""
    pose = np.asarray(obs.get("self_state", np.zeros(3, dtype=np.float32))[:3], dtype=np.float64)
    return float(-1.0e-4 * np.linalg.norm(pose))


@dataclass(frozen=True)
class StepDecision:
    """One agent's decision for a step: the action sample plus opaque learner bookkeeping
    (log-prob / value for PPO, chosen Q index for QMIX)."""

    action_sample: Mapping[str, Any]
    log_prob: float = 0.0
    value: float = 0.0
    extra: Mapping[str, Any] = field(default_factory=dict)


#: A learner's per-step decision function: live-agent flat observations → decisions.
AgentStepFn = Callable[[Mapping[AgentId, NDArray[np.float32]]], Mapping[AgentId, StepDecision]]

#: A picklable, argument-free factory that builds a fresh :class:`~astro_mine.learn.envs.SwarmEnv`
#: — the seam :class:`KubeRayExecutor` hands to each rollout worker so a worker builds its *own*
#: env (never ships one), mirroring the RLlib env-registry pattern (``train/rllib.py``).
EnvFactory = Callable[[], "SwarmEnv"]


@dataclass
class RolloutStep:
    """The recorded transition at one env step (only live agents present)."""

    obs: dict[AgentId, NDArray[np.float32]]
    action: dict[AgentId, Mapping[str, Any]]
    log_prob: dict[AgentId, float]
    value: dict[AgentId, float]
    reward: dict[AgentId, float]
    done: dict[AgentId, bool]
    reach: dict[AgentId, tuple[AgentId, ...]]
    extra: dict[AgentId, Mapping[str, Any]]
    state: NDArray[np.float32]


@dataclass
class Rollout:
    """A collected trajectory + the static shapes the trainer needs to build tensors."""

    steps: list[RolloutStep]
    possible_agents: tuple[AgentId, ...]
    agent_obs_dim: dict[AgentId, int]
    state_dim: int

    def env_steps(self) -> int:
        """Total env-steps summed over live agents (the throughput numerator)."""
        return sum(len(step.obs) for step in self.steps)

    def agent_trajectory(self, agent: AgentId) -> list[RolloutStep]:
        """The contiguous steps where ``agent`` was live (its per-agent sub-trajectory)."""
        return [step for step in self.steps if agent in step.obs]


@runtime_checkable
class BroadcastableStep(Protocol):
    """An :data:`AgentStepFn` whose learner state can be **shipped to a remote rollout
    worker and reconstructed there** — the seam that lets :class:`KubeRayExecutor` collect
    experience with the *same* policy the learner holds without pickling the (unpicklable)
    trainer/optimizer/``torch.Generator``.

    A trainer's step object satisfies this by exposing:

    - :meth:`broadcast` — ``(rebuild, state)`` where ``rebuild`` is a picklable, module-level
      callable and ``state`` a picklable payload (per-agent weights + net architecture + the
      generator state) such that ``rebuild(state)`` reconstructs a byte-equivalent step;
    - :meth:`rng_state` / :meth:`set_rng_state` — the sampling-RNG state, captured on the
      worker after its rollout and written back into the learner so a **single-worker**
      distributed run advances the learner RNG exactly as :class:`LocalExecutor` would (the
      byte-identity gate). ``LocalExecutor`` accepts *any* callable; only the distributed
      executor requires this richer contract.
    """

    def __call__(
        self, flat_obs: Mapping[AgentId, NDArray[np.float32]]
    ) -> Mapping[AgentId, StepDecision]: ...

    def broadcast(self) -> tuple[Callable[[Any], AgentStepFn], Any]: ...

    def rng_state(self) -> Any: ...

    def set_rng_state(self, state: Any) -> None: ...


@runtime_checkable
class CommsAwareStep(Protocol):
    """An :data:`AgentStepFn` that also consumes the per-tick **reachability verdict** — the
    seam the comms-learning baseline (``comms_ppo``) learns messages over.

    Learned messages must ride the *same* gated/dropped/delayed channel the
    :class:`~astro_mine.learn.envs.CommsModel` imposes (learn.md §3), so the reachable-peer set
    a message-passing policy aggregates over is the environment's own post-channel verdict —
    read from ``infos[agent]["comms"]``, never from a side channel. Before each decision the
    executor hands the step that verdict via :meth:`observe_reach`; a step that does not
    implement this protocol (every comms-blind baseline) is simply not called.

    The reach map is *the same object* the executor records on the :class:`RolloutStep`, so the
    messages the policy conditioned on at rollout time are exactly the ones the trainer can
    recompute — differentiably — during its update.
    """

    def observe_reach(self, reach: Mapping[AgentId, tuple[AgentId, ...]]) -> None: ...


@runtime_checkable
class BatchedStep(Protocol):
    """An :data:`AgentStepFn` that can decide for a whole **batch of env copies in one forward**.

    The property that makes the GPU-vectorized rollout
    (:class:`~astro_mine.learn.envs.vector.VectorExecutor`) a genuinely *batched kernel* rather
    than a sequential loop: given each agent's stacked ``(num_envs, obs_dim)`` observations, the
    policy crosses the net **once** for the entire batch and returns one decision per env copy,
    in env order. A step that does not implement this protocol forces the executor onto its
    sequential CPU fallback (learn.md §8 strategy 1).
    """

    def batch(
        self,
        flat_obs: Mapping[AgentId, NDArray[np.float32]],
        reach: NDArray[np.float32] | None = None,
    ) -> Mapping[AgentId, list[StepDecision]]: ...


class RolloutExecutor(Protocol):
    """The seam RM-P1-LEARN-04 swaps (local ⇄ KubeRay ⇄ GPU-vectorized) without forking
    algorithm code."""

    def rollout(
        self,
        env: SwarmEnv,
        agent_step: AgentStepFn,
        *,
        steps: int,
        seed: int,
        reward_fn: RewardFn | None = None,
    ) -> Rollout: ...


class LocalExecutor:
    """In-process rollout — the tier-1 default (learn.md §7 tier 1: single workstation)."""

    def rollout(
        self,
        env: SwarmEnv,
        agent_step: AgentStepFn,
        *,
        steps: int,
        seed: int,
        reward_fn: RewardFn | None = None,
    ) -> Rollout:
        obs, infos = env.reset(seed=seed)
        collected: list[RolloutStep] = []
        for _ in range(steps):
            live = list(env.agents)
            if not live:
                break
            flat = {
                a: flatten_obs(obs[a], cast(spaces.Dict, env.observation_space(a))) for a in live
            }
            # The channel's post-gate/drop/delay verdict for THIS tick, read off the infos the
            # env just produced — computed *before* the decision so a comms-learning step
            # (CommsAwareStep) conditions its messages on the same reachability the executor
            # records on the RolloutStep. Comms-blind steps are unaffected: `reach` depends only
            # on `infos` and `live`, neither of which the decision touches.
            reach = {
                a: tuple(link.peer for link in infos.get(a, {}).get("comms", [])) for a in live
            }
            if isinstance(agent_step, CommsAwareStep):
                agent_step.observe_reach(reach)
            decisions = agent_step(flat)
            state = env.state()
            shaped = (
                {a: float(reward_fn(a, obs[a])) for a in live} if reward_fn is not None else None
            )
            actions = {a: decisions[a].action_sample for a in live}
            next_obs, rewards, terms, truncs, next_infos = env.step(actions)
            reward = shaped or {a: float(rewards.get(a, 0.0)) for a in live}
            done = {a: bool(terms.get(a, False) or truncs.get(a, False)) for a in live}
            collected.append(
                RolloutStep(
                    obs=flat,
                    action=actions,
                    log_prob={a: decisions[a].log_prob for a in live},
                    value={a: decisions[a].value for a in live},
                    reward=reward,
                    done=done,
                    reach=reach,
                    extra={a: decisions[a].extra for a in live},
                    state=np.asarray(state, dtype=np.float32),
                )
            )
            obs, infos = next_obs, next_infos
        return Rollout(
            steps=collected,
            possible_agents=tuple(env.possible_agents),
            agent_obs_dim={
                a: flat_obs_dim(cast(spaces.Dict, env.observation_space(a)))
                for a in env.possible_agents
            },
            state_dim=int(np.asarray(env.state()).shape[0]),
        )


#: One worker's returned trajectory + its post-rollout sampling-RNG state.
WorkerResult = tuple["Rollout", Any]


@runtime_checkable
class WorkerBatch(Protocol):
    """Runs a broadcast step on ``seeds`` workers and returns results in worker order.

    The single seam that differs between the distributed (Ray) and in-process paths: the
    orchestration around it (broadcast, per-worker seeds, concatenation, RNG write-back) is
    shared in :func:`fan_out_rollout`. The Ray implementation lives in
    :mod:`astro_mine.learn.train.kuberay`; :func:`in_process_worker_batch` is the Ray-free
    realization used off-cluster and to test the executor's semantics deterministically."""

    def __call__(
        self,
        rebuild: Callable[[Any], AgentStepFn],
        state: Any,
        env_factory: EnvFactory,
        seeds: list[int],
        *,
        steps: int,
        reward_fn: RewardFn | None,
    ) -> list[WorkerResult]: ...


def in_process_worker_batch(
    rebuild: Callable[[Any], AgentStepFn],
    state: Any,
    env_factory: EnvFactory,
    seeds: list[int],
    *,
    steps: int,
    reward_fn: RewardFn | None,
) -> list[WorkerResult]:
    """Run each worker's rollout **sequentially in-process** (no Ray): rebuild the broadcast
    step, run the exact :class:`LocalExecutor` loop over a freshly built env, per seed.

    This is the off-cluster fan-out and the deterministic test double for
    :class:`KubeRayExecutor` — it exercises the *same* broadcast/rebuild round-trip and
    ordering the Ray path does, so it proves the executor's correctness (single-worker
    byte-identity to :class:`LocalExecutor`, topology-fixed reproduction) without a real
    cluster, whose autoscaling/preemption is unverifiable in CI anyway (cloud.md §7)."""
    executor = LocalExecutor()
    results: list[WorkerResult] = []
    for worker_seed in seeds:
        step = rebuild(state)
        rollout = executor.rollout(
            env_factory(), step, steps=steps, seed=worker_seed, reward_fn=reward_fn
        )
        results.append((rollout, step.rng_state()))  # type: ignore[attr-defined]
    return results


def _concat_rollouts(rollouts: list[Rollout]) -> Rollout:
    """Concatenate worker rollouts in fixed order into one batch.

    Every worker builds the same env, so the static shape (possible agents, per-agent obs
    dims, state dim) is identical and comes from worker 0; only the per-step trajectories
    accumulate."""
    if not rollouts:  # pragma: no cover - num_workers >= 1 always yields a rollout
        raise ValueError("no worker rollouts to concatenate")
    first = rollouts[0]
    return Rollout(
        steps=[step for rollout in rollouts for step in rollout.steps],
        possible_agents=first.possible_agents,
        agent_obs_dim=first.agent_obs_dim,
        state_dim=first.state_dim,
    )


def fan_out_rollout(
    agent_step: BroadcastableStep,
    env_factory: EnvFactory,
    *,
    steps: int,
    seed: int,
    reward_fn: RewardFn | None,
    num_workers: int,
    run_batch: WorkerBatch,
) -> Rollout:
    """Broadcast once, run ``num_workers`` workers via ``run_batch``, concatenate in order.

    Shared by the distributed and in-process paths: worker 0 uses ``seed`` verbatim
    (:func:`derive_seeds`), so a single worker reproduces :class:`LocalExecutor`; the worker-0
    post-rollout RNG state is written back into the learner step so a single-worker training
    loop advances the learner generator exactly as the in-process loop would."""
    rebuild, state = agent_step.broadcast()
    seeds = derive_seeds(seed, num_workers)
    results = run_batch(rebuild, state, env_factory, seeds, steps=steps, reward_fn=reward_fn)
    agent_step.set_rng_state(results[0][1])
    return _concat_rollouts([rollout for rollout, _ in results])


class KubeRayExecutor:
    """The distributed (KubeRay) executor — the RM-P1-LEARN-04 drop-in for
    :class:`LocalExecutor` behind the *same* trainer code (learn.md §2.1 "the same code with
    a different executor, never a fork").

    Fans rollout collection to a Ray actor pool: the learner :meth:`~BroadcastableStep.broadcast`
    its current per-agent weights once into the object store, each worker rebuilds the step
    and runs the **exact** :class:`LocalExecutor` loop against an env it builds from
    ``env_factory`` (workers never ship an env; mirrors ``train/rllib.py``'s env registry),
    and the learner concatenates the returned :class:`Rollout`\\ s in **fixed worker order**
    and runs its unchanged ``_update``.

    Determinism is **topology-fixed** (conventions.md §11): worker 0 always uses the run
    ``seed`` verbatim, so a *single-worker* KubeRay rollout is byte-identical to
    :class:`LocalExecutor` for that seed (the "cluster reproduces the workstation" gate);
    extra workers get decorrelated seeds, so any *fixed* ``(num_workers, seed)`` reproduces
    run-to-run but changing the worker count changes the batch composition. Ray-dependent
    worker code lives in :mod:`astro_mine.learn.train.kuberay` behind a lazy ``import ray``,
    so this module stays Torch/Ray-free-importable (only :meth:`rollout` pulls Ray in).
    """

    def __init__(
        self,
        env_factory: EnvFactory,
        *,
        num_workers: int = 1,
        ray_address: str | None = None,
        ray_init_kwargs: Mapping[str, Any] | None = None,
        runner: WorkerBatch | None = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self._env_factory = env_factory
        self._num_workers = num_workers
        self._ray_address = ray_address
        self._ray_init_kwargs = dict(ray_init_kwargs or {})
        #: The worker-execution seam. ``None`` uses the Ray actor pool (the real distributed
        #: path); inject :func:`in_process_worker_batch` to run off-cluster / in tests without
        #: a Ray cluster (the executor's orchestration is identical either way).
        self._runner = runner

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def rollout(
        self,
        env: SwarmEnv,
        agent_step: AgentStepFn,
        *,
        steps: int,
        seed: int,
        reward_fn: RewardFn | None = None,
    ) -> Rollout:
        if not isinstance(agent_step, BroadcastableStep):
            raise TypeError(
                "KubeRayExecutor requires a BroadcastableStep (the trainer step exposes "
                "broadcast()/rng_state()); a plain callable cannot be reconstructed on a "
                "remote worker — use LocalExecutor for a bare step function."
            )
        run_batch = self._runner if self._runner is not None else self._ray_batch()
        return fan_out_rollout(
            agent_step,
            self._env_factory,
            steps=steps,
            seed=seed,
            reward_fn=reward_fn,
            num_workers=self._num_workers,
            run_batch=run_batch,
        )

    def _ray_batch(self) -> WorkerBatch:
        """The Ray-backed worker batch, bound to this executor's cluster config. Imported
        lazily so :mod:`executor` stays Ray-free at import (mirrors train/rllib.py isolation)."""
        # Imported here (not at module top) so importing the executor never imports Ray.
        from astro_mine.learn.train import kuberay

        def run_batch(
            rebuild: Callable[[Any], AgentStepFn],
            state: Any,
            env_factory: EnvFactory,
            seeds: list[int],
            *,
            steps: int,
            reward_fn: RewardFn | None,
        ) -> list[WorkerResult]:
            return kuberay.run_worker_batch(
                rebuild,
                state,
                env_factory,
                seeds,
                steps=steps,
                reward_fn=reward_fn,
                ray_address=self._ray_address,
                ray_init_kwargs=self._ray_init_kwargs,
            )

        return run_batch
