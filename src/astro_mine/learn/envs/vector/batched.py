"""The batched-world seam + the batched rollout kernel (RM-P1-LEARN-04; learn.md §8).

"Simulation throughput is the dominant cost" (learn.md §8), and strategy 1 is **GPU-vectorized
envs**: thousands of env copies resident on one accelerator, stepped *together*. That only pays
if **both** halves of the rollout are batched:

1. the **world** steps its whole batch in one call — :class:`BatchedWorld`, the protocol a
   GPU-resident tier satisfies (Sim's Brax/MJX contact-physics tier; a JAX surrogate; the
   reference :class:`~astro_mine.learn.envs.vector.jax_world.JaxBatchedWorld` kernel here); and
2. the **policy** decides for the whole batch in one forward —
   :class:`~astro_mine.learn.train.executor.BatchedStep`.

:func:`batched_rollout` is that lockstep kernel. It is backend-agnostic (plain numpy at the
seam) — JAX/Brax/MJX/CUDA live *behind* :class:`BatchedWorld`, never in the rollout loop — and
it returns the ordinary :class:`~astro_mine.learn.train.executor.Rollout`, so the trainer's
``_update`` is byte-for-byte the same code it runs against the in-process CPU executor. That is
the RM-P1-LEARN-04 promise: "the same code with a different executor, never a fork".

**Trajectory ordering matters.** The batch is stepped in lockstep, but the returned
:class:`Rollout` is concatenated **copy-major**: each env copy's transitions stay *contiguous*,
because GAE (``compute_gae``) walks a trajectory in order and bootstraps across adjacent steps —
interleaving the copies would silently corrupt every advantage estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.learn.train.executor import (
    BatchedStep,
    Rollout,
    RolloutStep,
    derive_seeds,
)

if TYPE_CHECKING:
    from astro_mine.learn.train.executor import StepDecision

__all__ = [
    "BatchedObservation",
    "BatchedTransition",
    "BatchedWorld",
    "batched_rollout",
    "peers_from_mask",
]


@dataclass(frozen=True)
class BatchedObservation:
    """One lockstep tick of a batched world — every quantity stacked over the env copies.

    ``obs`` maps an agent to its ``(num_envs, obs_dim)`` flat observations; ``state`` is the
    ``(num_envs, state_dim)`` global CTDE state; ``live`` marks, per agent, which copies that
    agent is still active in; and ``reach`` is the ``(num_envs, n_agents, n_agents)``
    reachability mask — entry ``(e, i, j)`` is 1 iff, in copy ``e``, agent ``j``'s message
    reached agent ``i`` this tick (the comms channel's own verdict, the same semantics
    ``infos[agent]["comms"]`` carries on the CPU tier)."""

    obs: Mapping[AgentId, NDArray[np.float32]]
    state: NDArray[np.float32]
    live: Mapping[AgentId, NDArray[np.bool_]]
    reach: NDArray[np.float32]


@dataclass(frozen=True)
class BatchedTransition(BatchedObservation):
    """A batched observation plus the reward/done the world produced reaching it.

    The batched tier supplies its **own** rewards (a GPU-resident world computes them in the
    same kernel as its dynamics); the CPU tier's ``reward_fn`` shaping hook — a convenience for
    the deliberately reward-free ``FakeSwarmWorld`` — does not apply here."""

    reward: Mapping[AgentId, NDArray[np.float32]] = field(default_factory=dict)
    done: Mapping[AgentId, NDArray[np.bool_]] = field(default_factory=dict)


@runtime_checkable
class BatchedWorld(Protocol):
    """A world that steps ``num_envs`` copies in **one** call — the GPU-vectorized tier.

    This is the seam Learn consumes and *does not implement for real physics*: the
    high-fidelity realization is Sim's Brax/MJX GPU tier (the paired ``astro-mine-sim`` issue),
    reached — like every world — behind the Core Environment contract, never by importing Sim.
    Learn ships one reference realization (:class:`JaxBatchedWorld`) so the kernel, the tests,
    and the throughput benchmark are real without waiting on that work.

    A conformant world keeps the *same* static shape as the CPU ``SwarmEnv`` it stands in for
    (same ``possible_agents``, same per-agent flat obs widths, same global state width), so a
    policy is tensor-compatible across the two tiers and a fidelity switch stays a config change
    (learn.md §2.2).
    """

    @property
    def num_envs(self) -> int: ...

    @property
    def possible_agents(self) -> tuple[AgentId, ...]: ...

    @property
    def agent_obs_dim(self) -> Mapping[AgentId, int]: ...

    @property
    def state_dim(self) -> int: ...

    def reset_batch(self, seeds: Sequence[int]) -> BatchedObservation:
        """Reset every copy, copy ``e`` under ``seeds[e]`` (topology-fixed reproducibility)."""
        ...

    def step_batch(
        self, actions: Mapping[AgentId, Sequence[Mapping[str, Any]]]
    ) -> BatchedTransition:
        """Step **all** copies at once. ``actions[agent][e]`` is that agent's action in copy
        ``e`` (an action sample dict, exactly what the CPU tier's ``decode_action`` consumes)."""
        ...


def peers_from_mask(
    mask: NDArray[np.float32], agent: AgentId, agents: Sequence[AgentId]
) -> tuple[AgentId, ...]:
    """The peers whose message reached ``agent``, read off one copy's ``(n, n)`` reach mask.

    Converts the batched world's matrix form back into the tuple form the CPU executor records
    on :class:`~astro_mine.learn.train.executor.RolloutStep`, so a comms-learning trainer
    recomputes its message aggregate identically on either tier."""
    row = agents.index(agent)
    return tuple(peer for col, peer in enumerate(agents) if mask[row, col] > 0.0)


def batched_rollout(
    world: BatchedWorld,
    agent_step: BatchedStep,
    *,
    steps: int,
    seed: int,
) -> Rollout:
    """Roll ``world``'s whole batch out in lockstep — one world step and one policy forward per
    agent per tick, for **all** ``num_envs`` copies.

    This is the genuinely batched kernel (as opposed to running a single-env executor in a
    Python loop ``num_envs`` times): per tick there is exactly one
    :meth:`BatchedWorld.step_batch` and one :meth:`BatchedStep.batch` per live agent, so the
    accelerator sees ``num_envs``-wide tensors and the per-env Python work is reduced to
    unstacking the results.

    Copy ``0`` is seeded with ``seed`` verbatim (:func:`derive_seeds`), so a fixed
    ``(num_envs, seed)`` reproduces run-to-run. The returned :class:`Rollout` is concatenated
    copy-major — each copy's trajectory contiguous — so GAE bootstraps within a copy and never
    across two."""
    agents = tuple(world.possible_agents)
    batch = world.reset_batch(derive_seeds(seed, world.num_envs))
    per_copy: list[list[RolloutStep]] = [[] for _ in range(world.num_envs)]

    for _ in range(steps):
        live = [agent for agent in agents if bool(np.asarray(batch.live[agent]).any())]
        if not live:
            break
        stacked = {agent: np.asarray(batch.obs[agent], dtype=np.float32) for agent in live}
        # THE batched kernel: one forward per agent for the entire env batch.
        decisions = agent_step.batch(stacked, np.asarray(batch.reach, dtype=np.float32))
        actions = {
            agent: [decision.action_sample for decision in decisions[agent]] for agent in live
        }
        transition = world.step_batch(actions)
        _record(per_copy, batch, decisions, transition, live, agents)
        batch = transition

    return Rollout(
        steps=[step for copy_steps in per_copy for step in copy_steps],
        possible_agents=agents,
        agent_obs_dim=dict(world.agent_obs_dim),
        state_dim=world.state_dim,
    )


def _record(
    per_copy: list[list[RolloutStep]],
    batch: BatchedObservation,
    decisions: Mapping[AgentId, list[StepDecision]],
    transition: BatchedTransition,
    live: list[AgentId],
    agents: tuple[AgentId, ...],
) -> None:
    """Unstack one lockstep tick into per-copy :class:`RolloutStep`\\ s (the only per-env work)."""
    reach = np.asarray(batch.reach, dtype=np.float32)
    for copy in range(len(per_copy)):
        here = [agent for agent in live if bool(np.asarray(batch.live[agent])[copy])]
        if not here:
            continue
        per_copy[copy].append(
            RolloutStep(
                obs={a: np.asarray(batch.obs[a], dtype=np.float32)[copy] for a in here},
                action={a: decisions[a][copy].action_sample for a in here},
                log_prob={a: decisions[a][copy].log_prob for a in here},
                value={a: decisions[a][copy].value for a in here},
                reward={a: float(np.asarray(transition.reward[a])[copy]) for a in here},
                done={a: bool(np.asarray(transition.done[a])[copy]) for a in here},
                reach={a: peers_from_mask(reach[copy], a, agents) for a in here},
                extra={a: decisions[a][copy].extra for a in here},
                state=np.asarray(batch.state, dtype=np.float32)[copy],
            )
        )
