"""The GPU-vectorized rollout executor (RM-P1-LEARN-04; learn.md §8 throughput strategies).

Simulation throughput is the dominant training cost (learn.md §8), so a policy trained on a
GPU-vectorized env — a batched Brax/MJX world or a surrogate — collects many parallel
trajectories per step. :class:`VectorExecutor` is that path as a plain
:class:`~astro_mine.learn.train.executor.RolloutExecutor`: the **same trainer code** swaps it in
exactly like the distributed executor, and it returns the identical
:class:`~astro_mine.learn.train.executor.Rollout` shape so ``_update`` is unchanged.

It has **two** realizations behind one seam, and picks between them automatically:

1. **Batched** (``backend="jax"``) — the real GPU-vectorized kernel. A
   :class:`~astro_mine.learn.envs.vector.batched.BatchedWorld` steps its whole batch in one call
   and a :class:`~astro_mine.learn.train.executor.BatchedStep` policy decides for the whole batch
   in one forward; :func:`~astro_mine.learn.envs.vector.batched.batched_rollout` runs them in
   lockstep. This is *not* a sequential loop: per tick there is one world step and one policy
   forward per agent, for all ``num_envs`` copies. Consumes Sim's GPU tier where available
   (same protocol); Learn ships :class:`JaxBatchedWorld` as the reference realization.
2. **Sequential CPU** (``backend="cpu"``) — the graceful fallback: ``num_envs`` in-process env
   copies rolled out one after another through :class:`LocalExecutor`. Torch-free, Ray-free,
   JAX-free; runs on tier-1 with no cloud and no GPU. Copy 0 uses the run seed verbatim, so a
   single copy is byte-identical to :class:`LocalExecutor`.

**Fallback is graceful and explicit.** ``backend="auto"`` (the default) takes the batched path
when a batched world is supplied *and* constructible (the ``[jax]`` extra is installed) *and* the
trainer's step implements :class:`BatchedStep`; otherwise it silently degrades to the sequential
CPU loop. :attr:`VectorExecutor.backend` reports which path was actually taken — the honest
signal the throughput benchmark records. ``backend="jax"`` demands the batched path and fails
loudly if it is unavailable (an explicit request must not silently run 100x slower).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from astro_mine.learn.envs.vector.batched import BatchedWorld, batched_rollout
from astro_mine.learn.train.executor import (
    AgentStepFn,
    BatchedStep,
    LocalExecutor,
    RewardFn,
    Rollout,
    derive_seeds,
)

if TYPE_CHECKING:
    from astro_mine.learn.envs import SwarmEnv
    from astro_mine.learn.train.executor import EnvFactory

__all__ = ["Backend", "BatchedWorldFactory", "VectorExecutor"]

#: The rollout backend. ``auto`` prefers the batched GPU kernel and degrades to the sequential
#: CPU loop; ``jax`` demands the batched kernel; ``cpu`` forces the sequential loop.
Backend = Literal["auto", "jax", "cpu"]

#: A zero-arg factory yielding a :class:`BatchedWorld` — the GPU tier's construction seam (Sim's
#: Brax/MJX world, a JAX surrogate, or :func:`jax_batched_world_factory`).
BatchedWorldFactory = Callable[[], BatchedWorld]


class VectorExecutor:
    """Batched rollout over ``num_envs`` env copies (the ``gpu_vectorized`` fidelity tier).

    Selectable per curriculum stage via ``TrainConfig.fidelity``; injected into a trainer as its
    ``executor`` exactly like :class:`LocalExecutor` / :class:`KubeRayExecutor`.

    With a ``batched_world`` (and a :class:`BatchedStep` trainer step) it runs the **genuinely
    batched** GPU kernel — one world step and one policy forward per tick for the whole batch.
    Without one it falls back to the sequential CPU loop: each copy built from ``env_factory``
    and rolled out with a topology-fixed derived seed
    (:func:`~astro_mine.learn.train.executor.derive_seeds`), the per-copy trajectories
    concatenated into one :class:`Rollout` — so ``num_envs=1`` reproduces the single-env
    :class:`LocalExecutor` run byte-for-byte and a fixed ``(num_envs, seed)`` reproduces
    run-to-run. Both paths return the identical :class:`Rollout` shape and keep each copy's
    trajectory contiguous (GAE bootstraps within a copy, never across two)."""

    def __init__(
        self,
        env_factory: EnvFactory,
        *,
        num_envs: int = 1,
        batched_world: BatchedWorldFactory | None = None,
        backend: Backend = "auto",
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self._env_factory = env_factory
        self._num_envs = num_envs
        self._requested = backend
        self._world = self._resolve_world(batched_world, backend)

    def _resolve_world(
        self, batched_world: BatchedWorldFactory | None, backend: Backend
    ) -> BatchedWorld | None:
        """Build the batched world, or degrade to the CPU loop — the graceful-fallback rule.

        ``cpu`` never builds one. ``jax`` demands one (loud failure otherwise). ``auto`` tries,
        and treats *both* "no world supplied" and "the [jax] extra is not installed"
        (``ImportError``) as "no accelerator here" — falling back to the sequential loop rather
        than exploding on a workstation without a GPU (learn.md §7 tier 1 MUST always work)."""
        if backend == "cpu":
            return None
        if batched_world is None:
            if backend == "jax":
                raise ValueError(
                    "backend='jax' needs a batched_world (a BatchedWorld factory — Sim's GPU "
                    "tier, or jax_batched_world_factory(env_factory)); none was supplied"
                )
            return None
        try:
            return batched_world()
        except ImportError as exc:
            if backend == "jax":
                raise ImportError(
                    "backend='jax' was requested but the GPU-vectorized world could not be "
                    "built — install the optional extra: "
                    "`pip install astro-mine-platform[learn-jax]`"
                ) from exc
            # auto: no JAX here → the sequential CPU loop, which always works.
            return None

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def backend(self) -> Literal["jax", "cpu"]:
        """The backend actually resolved — ``jax`` for the batched kernel, ``cpu`` for the
        sequential fallback. The honest signal the throughput benchmark records (a "GPU" number
        measured on the CPU fallback would be a lie)."""
        return "jax" if self._world is not None else "cpu"

    @property
    def batched_world(self) -> BatchedWorld | None:
        """The resolved batched world (``None`` on the sequential CPU path)."""
        return self._world

    def rollout(
        self,
        env: SwarmEnv,
        agent_step: AgentStepFn,
        *,
        steps: int,
        seed: int,
        reward_fn: RewardFn | None = None,
    ) -> Rollout:
        """Collect ``steps`` ticks over ``num_envs`` copies through whichever backend resolved.

        The batched path additionally requires the step to be a :class:`BatchedStep` (every Learn
        trainer's step is); a bare callable — a test double, a hand-written step — cannot be run
        as a batch, so it degrades to the sequential loop rather than failing."""
        if self._world is not None and isinstance(agent_step, BatchedStep):
            # The batched tier supplies its own rewards (it computes them in the same device
            # kernel as its dynamics), so the CPU tier's reward_fn shaping hook does not apply.
            return batched_rollout(self._world, agent_step, steps=steps, seed=seed)
        return self._sequential(agent_step, steps=steps, seed=seed, reward_fn=reward_fn)

    def _sequential(
        self,
        agent_step: AgentStepFn,
        *,
        steps: int,
        seed: int,
        reward_fn: RewardFn | None,
    ) -> Rollout:
        """``num_envs`` in-process env copies, rolled out one after another (the CPU fallback)."""
        local = LocalExecutor()
        rollouts = [
            local.rollout(
                self._env_factory(), agent_step, steps=steps, seed=env_seed, reward_fn=reward_fn
            )
            for env_seed in derive_seeds(seed, self._num_envs)
        ]
        first = rollouts[0]
        return Rollout(
            steps=[step for rollout in rollouts for step in rollout.steps],
            possible_agents=first.possible_agents,
            agent_obs_dim=first.agent_obs_dim,
            state_dim=first.state_dim,
        )
