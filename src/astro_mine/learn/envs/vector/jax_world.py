"""The JAX/XLA reference :class:`BatchedWorld` — the GPU-vectorized rollout tier (LEARN-04).

learn.md §8 strategy 1 is **GPU-vectorized envs**: "thousands of envs resident on one GPU, best
sample-throughput when a differentiable/fast surrogate exists", and learn.md §4 names **JAX**
(Brax-style vectorized envs) as the path for "massively parallel, GPU-resident rollouts". This
module is that path's concrete kernel: a pure, ``jax.jit``-compiled, ``jax.vmap``-ed transition
over a **device-resident** batch of env states, exposed through the plain
:class:`~astro_mine.learn.envs.vector.batched.BatchedWorld` protocol so
:class:`~astro_mine.learn.envs.vector.VectorExecutor` drives it with no JAX in the rollout loop.

**What this is — and is not.** It is *not* a physics model, and it does not pretend to be: the
high-fidelity GPU tier is **Sim's** Brax/MJX contact physics, which lands behind this *same*
protocol (the paired ``astro-mine-sim`` issue) and which Learn will consume without importing
Sim — the env is the only physics boundary (learn.md §2.2). What this *is*: the reference
realization that makes the GPU-vectorized seam real today — it compiles one XLA program for the
whole batch, places it on an accelerator when one exists, and is what the ``gpu``-marked test
and the throughput benchmark (:mod:`~astro_mine.learn.envs.vector.benchmark`) actually exercise.
It mirrors a :class:`~astro_mine.learn.envs.SwarmEnv`'s **static shape exactly** (same agents,
same flat observation widths, same global-state width), so a policy is tensor-compatible across
the CPU and GPU tiers and switching between them stays a config change.

Dynamics (all vmapped over the batch, all on device):

- each agent integrates its ``goto`` action into a position — a real, differentiable transition;
- the comms channel is a **range gate**: agent ``j``'s message reaches ``i`` iff they are within
  ``comms_range_m`` — the same reachability semantics the CPU
  :class:`~astro_mine.learn.envs.CommsModel` produces, so the comms-learning baseline
  (``comms_ppo``) trains on this tier unchanged;
- the reward mirrors the CPU tier's ``default_reward_fn`` (a small negative distance-to-origin),
  so returns are comparable across tiers.

Needs the ``[jax]`` extra. Every JAX import is **lazy** (inside the constructor), so this module
imports fine without it and :class:`VectorExecutor` can fall back gracefully.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.learn.envs.vector.batched import BatchedObservation, BatchedTransition

if TYPE_CHECKING:
    from astro_mine.learn.envs import SwarmEnv

__all__ = [
    "JaxBatchedWorld",
    "accelerator_devices",
    "jax_available",
    "jax_batched_world_factory",
]


def jax_available() -> bool:
    """Whether the optional ``[jax]`` extra is installed (the GPU-vectorized tier's backend)."""
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def accelerator_devices() -> list[str]:
    """The non-CPU JAX device platforms visible here (``['gpu']``, ``['tpu']``, or ``[]``).

    Empty when JAX is absent **or** is the CPU-only wheel — the two cases
    :class:`VectorExecutor` must degrade gracefully through. The batched kernel is *correct*
    either way (XLA-CPU runs the identical jit/vmap program, which is exactly why CI can cover
    it); the accelerator is what makes it *fast*, so this is a throughput signal, not a
    correctness one."""
    if not jax_available():
        return []
    import jax

    return [device.platform for device in jax.devices() if device.platform != "cpu"]


class JaxBatchedWorld:
    """A JAX-backed :class:`~astro_mine.learn.envs.vector.batched.BatchedWorld` (see module doc).

    Build it against the :class:`~astro_mine.learn.envs.SwarmEnv` it stands in for
    (:meth:`from_swarm_env`) so it inherits that env's exact agents and tensor widths. The whole
    batch — positions, ticks, observations, reachability — lives in device arrays and advances
    through one compiled XLA program per :meth:`step_batch`."""

    def __init__(
        self,
        possible_agents: Sequence[AgentId],
        agent_obs_dim: Mapping[AgentId, int],
        state_dim: int,
        *,
        num_envs: int = 1,
        horizon: int = 32,
        dt_s: float = 1.0,
        comms_range_m: float = 150.0,
        device: str | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        # Lazy on purpose: jax is the optional [jax] extra. Raises loudly (ImportError) only if
        # the GPU-vectorized world is actually constructed without it — VectorExecutor catches
        # that and falls back to the sequential CPU loop.
        import jax
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        self._agents: tuple[AgentId, ...] = tuple(possible_agents)
        self._obs_dim = dict(agent_obs_dim)
        self._state_dim = state_dim
        self._num_envs = num_envs
        self._horizon = horizon
        self._dt = float(dt_s)
        self._range = float(comms_range_m)
        self._n = len(self._agents)
        #: The widest per-agent observation — the batch is carried as one dense
        #: ``(num_envs, n_agents, feat_dim)`` device array and sliced per agent on the way out,
        #: which keeps the kernel a single uniform vmap over heterogeneous agents.
        self._feat_dim = max(self._obs_dim.values()) if self._obs_dim else 1
        self._device = self._resolve_device(device)

        # One compiled XLA program each, vmapped over the leading (env-copy) axis.
        self._observe = jax.jit(jax.vmap(self._observe_kernel))
        self._advance = jax.jit(jax.vmap(self._advance_kernel))
        self._pos: Any = None
        self._tick: Any = None

    # --- construction ---------------------------------------------------------------

    @classmethod
    def from_swarm_env(cls, env: SwarmEnv, *, num_envs: int = 1, **kwargs: Any) -> JaxBatchedWorld:
        """Mirror a CPU :class:`SwarmEnv`'s static shape onto the GPU-vectorized tier.

        Reads the agents, the per-agent flat observation widths, and the global ``state()`` width
        straight off the env's declared spaces, so a policy built for that env drops onto this
        world with no reshaping — the tiers differ in *throughput*, not in tensor contract."""
        from astro_mine.learn.algos.policy import flat_obs_dim

        specs = env.agent_specs
        return cls(
            tuple(env.possible_agents),
            {agent: flat_obs_dim(spec.observation_space) for agent, spec in specs.items()},
            int(np.asarray(env.state()).shape[0]),
            num_envs=num_envs,
            **kwargs,
        )

    def _resolve_device(self, device: str | None) -> Any:
        """Pin the batch to an accelerator when one exists, else XLA-CPU (the same program)."""
        jax = self._jax
        if device is not None:
            return jax.devices(device)[0]
        accelerators = [d for d in jax.devices() if d.platform != "cpu"]
        return accelerators[0] if accelerators else jax.devices()[0]

    # --- BatchedWorld ---------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def possible_agents(self) -> tuple[AgentId, ...]:
        return self._agents

    @property
    def agent_obs_dim(self) -> Mapping[AgentId, int]:
        return dict(self._obs_dim)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def device_platform(self) -> str:
        """The platform the batch is resident on (``'gpu'``/``'tpu'``/``'cpu'``)."""
        platform: str = self._device.platform
        return platform

    def reset_batch(self, seeds: Sequence[int]) -> BatchedObservation:
        """Place a fresh, seeded batch on the device (copy ``e`` from ``seeds[e]``)."""
        if len(seeds) != self._num_envs:
            raise ValueError(f"expected {self._num_envs} seeds, got {len(seeds)}")
        jax = self._jax
        # Deterministic per-copy initial positions, derived from each copy's own seed.
        starts = np.stack(
            [
                np.random.default_rng(int(seed))
                .uniform(-100.0, 100.0, size=(self._n, 3))
                .astype(np.float32)
                for seed in seeds
            ]
        )
        self._pos = jax.device_put(self._jnp.asarray(starts), self._device)
        self._tick = jax.device_put(self._jnp.zeros((self._num_envs,), dtype=self._jnp.int32))
        return self._render()

    def step_batch(
        self, actions: Mapping[AgentId, Sequence[Mapping[str, Any]]]
    ) -> BatchedTransition:
        """Advance **every** copy through one compiled, vmapped XLA step."""
        goto = np.zeros((self._num_envs, self._n, 3), dtype=np.float32)
        for index, agent in enumerate(self._agents):
            samples = actions.get(agent)
            if samples is None:
                continue
            for copy, sample in enumerate(samples):
                block = sample.get("goto")
                if block is not None:
                    goto[copy, index] = np.asarray(block, dtype=np.float32)[:3]
        self._pos, self._tick = self._advance(
            self._pos, self._tick, self._jax.device_put(self._jnp.asarray(goto), self._device)
        )
        rendered = self._render()
        done = self._host(self._tick, np.int32) >= self._horizon
        reward = self._reward()
        return BatchedTransition(
            obs=rendered.obs,
            state=rendered.state,
            live={agent: ~done for agent in self._agents},
            reach=rendered.reach,
            reward={
                agent: np.array(reward[:, index], dtype=np.float32)
                for index, agent in enumerate(self._agents)
            },
            done={agent: done for agent in self._agents},
        )

    # --- device kernels (pure, vmapped over the env-copy axis) ----------------------

    def _advance_kernel(self, pos: Any, tick: Any, goto: Any) -> tuple[Any, Any]:
        """One env copy's transition: integrate the ``goto`` command into position."""
        return pos + self._dt * goto, tick + 1

    def _observe_kernel(self, pos: Any, tick: Any) -> tuple[Any, Any, Any]:
        """One env copy's observation, global state, and reachability — all on device.

        Returns the dense ``(n_agents, feat_dim)`` observation block (sliced to each agent's own
        declared width on the way out), the flat global CTDE state, and the range-gated
        ``(n_agents, n_agents)`` comms mask."""
        jnp = self._jnp
        # Per-agent features: pose, distance-to-origin, and the tick — tiled out to the dense
        # width. A real BatchedWorld (Sim's GPU tier) encodes the true SwarmEnv layout here.
        radius = jnp.linalg.norm(pos, axis=-1, keepdims=True)
        base = jnp.concatenate([pos, radius, jnp.full((self._n, 1), tick, dtype=jnp.float32)], -1)
        reps = -(-self._feat_dim // base.shape[-1])  # ceil-div
        obs = jnp.tile(base, (1, reps))[:, : self._feat_dim]

        state = jnp.zeros((self._state_dim,), dtype=jnp.float32)
        flat = pos.reshape(-1)[: self._state_dim]
        state = state.at[: flat.shape[0]].set(flat)

        # Range-gated comms: j's message reaches i iff they are within comms_range_m. Same
        # reachability semantics the CPU CommsModel's range gate produces; no self-links.
        delta = pos[:, None, :] - pos[None, :, :]
        distance = jnp.linalg.norm(delta, axis=-1)
        reach = (distance <= self._range).astype(jnp.float32)
        reach = reach * (1.0 - jnp.eye(self._n, dtype=jnp.float32))
        return obs, state, reach

    # --- host-side rendering --------------------------------------------------------

    def _host(self, array: Any, dtype: Any = np.float32) -> NDArray[Any]:
        """Copy a device array back to a **writable** host array.

        ``np.asarray`` on a JAX array yields a read-only view, and ``torch.from_numpy`` warns
        (and is undefined behaviour) on those — so the device→host boundary copies once, here,
        rather than leaving every downstream consumer to discover it."""
        return np.array(array, dtype=dtype)

    def _render(self) -> BatchedObservation:
        obs, state, reach = self._observe(self._pos, self._tick)
        dense = self._host(obs)
        live = self._host(self._tick, np.int32) < self._horizon
        return BatchedObservation(
            obs={
                # A copy per agent: slicing the dense block would alias one buffer across agents.
                agent: np.array(dense[:, index, : self._obs_dim[agent]], dtype=np.float32)
                for index, agent in enumerate(self._agents)
            },
            state=self._host(state),
            live={agent: live for agent in self._agents},
            reach=self._host(reach),
        )

    def _reward(self) -> NDArray[np.float32]:
        """A small negative distance-to-origin — mirrors the CPU tier's ``default_reward_fn`` so
        returns are comparable across the two fidelity tiers."""
        pos = self._host(self._pos)
        reward: NDArray[np.float32] = (-1.0e-4 * np.linalg.norm(pos, axis=-1)).astype(np.float32)
        return reward


def jax_batched_world_factory(env_factory: Any, *, num_envs: int = 1, **kwargs: Any) -> Any:
    """A zero-arg :class:`JaxBatchedWorld` factory mirroring ``env_factory``'s SwarmEnv.

    The shape :class:`VectorExecutor` (and ``astro-mine-train --batched-world``) consumes: it
    builds one CPU env only to *read its declared spaces*, then hands back the GPU-resident
    world. Raises ``ImportError`` without the ``[jax]`` extra — which the executor's ``auto``
    backend catches to fall back to the sequential CPU loop."""

    def factory() -> JaxBatchedWorld:
        return JaxBatchedWorld.from_swarm_env(env_factory(), num_envs=num_envs, **kwargs)

    return factory
