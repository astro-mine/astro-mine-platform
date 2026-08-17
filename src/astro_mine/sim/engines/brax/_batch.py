# SPDX-License-Identifier: Apache-2.0
"""GPU-batched swarm-scale rollout — the ``jax.vmap`` path over N parallel envs (RM-P1-SIM-04).

The throughput half of the Brax/MJX tier: :func:`agent_step` (the single-agent JAX kernel) mapped a
**second** time over an env axis, so one XLA-compiled call steps N low-fidelity envs at once —
"thousands of low-fidelity parallel rollouts per GPU-hour" for [Learn](learn.md) swarm-scale
training (sim.md §8). Learn consumes this through the Gymnasium/PettingZoo views over the **Core**
message types; nothing engine-typed crosses the seam here either: :class:`VectorizedRollout` is
driven by a Core :class:`~astro_mine.core.messages.model.ActionBatch` and emits Core
:class:`~astro_mine.core.messages.model.Observation`\\ s per env (sim.md §2 principle 1).

Per-env seeding is by **global** env index (``fold_in(root_key, env_index)``), so a shard of envs
computes the identical states it would in one in-process run — the property the Ray fan-out
(:mod:`._ray`) relies on to aggregate shard results back into one batch. JAX (and its x64 setting)
arrive with :mod:`._engine`; this module is JAX-heavy and loads only via the ``[brax]`` factory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Observation, Quat, StateSample, Transform, Vec3
from astro_mine.sim.engines._rover_mjcf import RoverModelSpec, rover_mjcf
from astro_mine.sim.engines.actuation import actions_by_agent
from astro_mine.sim.engines.brax._engine import (
    BraxParams,
    agent_step,
    brax_prng_key,
    sample_initial_velocity,
)
from astro_mine.sim.engines.brax._mjx import mjx_rollout, model_spec_from_mjx_dynamics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.core.units import ReferenceFrame
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = [
    "MjxVectorizedRollout",
    "RolloutBatch",
    "VectorizedRollout",
    "build_mjx_vectorized_rollout",
    "build_vectorized_rollout",
]


@runtime_checkable
class RolloutBatch(Protocol):
    """The batched-rollout surface the Ray fan-out drives — satisfied by **both** JAX tiers.

    :class:`VectorizedRollout` (the cheap reduced-order kernel) and :class:`MjxVectorizedRollout`
    (real MJX contact) expose the identical surface, so the fan-out, its sharding, and its
    aggregation oracle (:mod:`._ray`) work unchanged for either tier — the contact upgrade is a
    fidelity change *behind* the existing architecture, not a new integration surface
    (RM-P1-SIM-04)."""

    @property
    def n_envs(self) -> int: ...
    @property
    def n_agents(self) -> int: ...
    @property
    def env_indices(self) -> tuple[int, ...]: ...
    @property
    def positions(self) -> Any:
        """The live ``(N, A, 3)`` position array."""
        ...

    def reset(self) -> tuple[tuple[Observation, ...], ...]: ...
    def step(self, actions: ActionBatch) -> tuple[tuple[Observation, ...], ...]: ...


_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)

#: The agent kernel mapped over agents (inner) then envs (outer): ``(N, A, …)`` in, ``(N, A, …)``
#: out — the whole batch stepped by one XLA-compiled call.
_agents_step = jax.vmap(agent_step, in_axes=((0, 0, 0), 0, None, None))
_envs_step = jax.vmap(_agents_step, in_axes=((0, 0, 0), 0, None, None))


@dataclass(slots=True)
class _BatchState:
    """The batched kinematics — leading axis is the env, second the agent."""

    pos: Any  # (N, A, 3)
    vel: Any  # (N, A, 3)
    soc: Any  # (N, A)


class VectorizedRollout:
    """N parallel low-fidelity mobility envs stepped as one ``jax.vmap`` batch.

    Constructed at the shared initial state (per-env domain-randomization jitter aside), it renders
    Core observations per env and steps every env under one broadcast
    :class:`~astro_mine.core.messages.model.ActionBatch` — the swarm-scale training rollout. Rows
    are ordered by ``env_indices`` (the global env ids used for seeding), so a shard's rows line up
    with the same rows of a whole-batch run."""

    def __init__(
        self,
        *,
        state: _BatchState,
        params: BraxParams,
        agent_ids: tuple[str, ...],
        frames: tuple[ReferenceFrame, ...],
        commanded: list[tuple[float, float, float]],
        modes: list[str | None],
        floors: Any,
        env_indices: tuple[int, ...],
        dt_s: float,
    ) -> None:
        self._init = _BatchState(pos=state.pos, vel=state.vel, soc=state.soc)
        self._state = state
        self._params = params
        self._agent_ids = agent_ids
        self._frames = frames
        self._commanded0 = list(commanded)
        self._modes0 = list(modes)
        self._commanded = list(commanded)
        self._modes = list(modes)
        self._floors = floors
        self._env_indices = env_indices
        self._dt_s = dt_s
        self._tick = 0
        self._elapsed_s = 0.0

    @property
    def n_envs(self) -> int:
        """The number of parallel envs (the batch/leading axis)."""
        return len(self._env_indices)

    @property
    def n_agents(self) -> int:
        """The number of agents per env."""
        return len(self._agent_ids)

    @property
    def env_indices(self) -> tuple[int, ...]:
        """The global env ids this rollout covers (its rows), in row order."""
        return self._env_indices

    @property
    def positions(self) -> Any:
        """The live ``(N, A, 3)`` position array — for shape/aggregation checks."""
        return self._state.pos

    def reset(self) -> tuple[tuple[Observation, ...], ...]:
        """Rewind to the initial batched state and return the per-env initial observations."""
        self._state = _BatchState(pos=self._init.pos, vel=self._init.vel, soc=self._init.soc)
        self._commanded = list(self._commanded0)
        self._modes = list(self._modes0)
        self._tick = 0
        self._elapsed_s = 0.0
        return self._observations()

    def step(self, actions: ActionBatch) -> tuple[tuple[Observation, ...], ...]:
        """Advance every env one tick under ``actions`` (broadcast to all envs); return the batch.

        A ``VELOCITY`` actuator command updates an agent's commanded velocity across the whole batch
        and a ``mode`` command its mode; the desired velocity (top-speed clamped) is broadcast to
        all envs and the vmapped kernel steps them at once. The battery-floor clamp is applied to
        the returned SoC."""
        self._apply(actions)
        targets = self._target_velocities()
        pos, vel, soc = _envs_step(
            (self._state.pos, self._state.vel, self._state.soc),
            targets,
            self._params.as_tuple(),
            self._dt_s,
        )
        self._state = _BatchState(pos=pos, vel=vel, soc=jnp.maximum(self._floors, soc))
        self._tick += 1
        self._elapsed_s += self._dt_s
        return self._observations()

    def _apply(self, actions: ActionBatch) -> None:
        index = {aid: i for i, aid in enumerate(self._agent_ids)}
        for agent_id, action in actions_by_agent(actions).items():
            i = index.get(agent_id)
            if i is None:
                continue
            if action.kind is ActionKind.MODE and action.mode is not None:
                self._modes[i] = action.mode.mode
            elif (
                action.kind is ActionKind.ACTUATOR
                and action.actuator is not None
                and action.actuator.control_mode is ControlMode.VELOCITY
                and len(action.actuator.setpoint) == 3
            ):
                sx, sy, sz = action.actuator.setpoint
                self._commanded[i] = (sx, sy, sz)

    def _target_velocities(self) -> Any:
        """The per-agent desired velocity (top-speed clamped), broadcast to ``(N, A, 3)``."""
        max_speed = self._params.max_speed_mps
        per_agent = jnp.asarray(
            [_clamp_speed(v, max_speed) for v in self._commanded], dtype=jnp.float64
        )
        return jnp.broadcast_to(per_agent, (self.n_envs, self.n_agents, 3))

    def _observations(self) -> tuple[tuple[Observation, ...], ...]:
        return tuple(
            tuple(self._observe(e, a) for a in range(self.n_agents)) for e in range(self.n_envs)
        )

    def _observe(self, e: int, a: int) -> Observation:
        p = self._state.pos[e, a]
        v = self._state.vel[e, a]
        self_state = StateSample(
            agent_id=self._agent_ids[a],
            frame=self._frames[a],
            pose=Transform(
                translation_m=Vec3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                rotation_quat_xyzw=_IDENTITY_QUAT,
            ),
            linear_velocity_mps=Vec3(x=float(v[0]), y=float(v[1]), z=float(v[2])),
            battery_soc_j=float(self._state.soc[e, a]),
            mode=self._modes[a],
        )
        return Observation(
            tick=self._tick,
            sim_time_s=self._elapsed_s,
            agent_id=self._agent_ids[a],
            self_state=self_state,
        )


def _clamp_speed(v: tuple[float, float, float], max_speed: float) -> list[float]:
    speed = float(sum(c * c for c in v)) ** 0.5
    if speed <= max_speed or speed == 0.0:
        return list(v)
    return [c / speed * max_speed for c in v]


def build_vectorized_rollout(
    scenario: Scenario,
    rng: RngStreams,
    *,
    n_envs: int | None = None,
    env_indices: Sequence[int] | None = None,
) -> VectorizedRollout:
    """Build a :class:`VectorizedRollout` over the scenario's ``brax_contact`` agents.

    ``env_indices`` selects the **global** env ids this rollout covers (a Ray shard passes its
    subset); default is ``range(n_envs)``, and ``n_envs`` defaults to the ``batch_size`` the first
    ``brax_contact`` agent declares (sim.md §8). Each env/agent's initial velocity is seeded from
    ``fold_in(root_key, env_index)`` then per agent, so a shard reproduces the whole-batch rows
    exactly. Raises ``ValueError`` if the scenario has no ``brax_contact`` agents or they disagree
    on parameters (the batch steps under one shared :class:`BraxParams`)."""
    specs = [s for s in scenario.agents if s.dynamics.kind == "brax_contact"]
    if not specs:
        raise ValueError("a vectorized brax rollout needs at least one brax_contact agent")
    first = specs[0].dynamics
    assert first.kind == "brax_contact"  # narrows the union
    params = BraxParams(
        mass_kg=first.mass_kg,
        max_speed_mps=first.max_speed_mps,
        max_traction_n=first.max_traction_n,
        idle_power_w=first.idle_power_w,
        drive_power_w_per_mps=first.drive_power_w_per_mps,
    )
    for spec in specs[1:]:
        dyn = spec.dynamics
        assert dyn.kind == "brax_contact"
        other = BraxParams(
            mass_kg=dyn.mass_kg,
            max_speed_mps=dyn.max_speed_mps,
            max_traction_n=dyn.max_traction_n,
            idle_power_w=dyn.idle_power_w,
            drive_power_w_per_mps=dyn.drive_power_w_per_mps,
        )
        if other != params:
            raise ValueError(
                f"brax_contact agents must share one parameter set; {spec.agent_id!r} differs"
            )

    if env_indices is None:
        count = n_envs if n_envs is not None else first.batch_size
        env_indices = tuple(range(count))
    else:
        env_indices = tuple(env_indices)
    if not env_indices:
        raise ValueError("a vectorized brax rollout needs at least one env")

    key = brax_prng_key(rng)
    jitter = first.init_speed_jitter_mps
    agent_ids = tuple(s.agent_id for s in specs)
    frames = tuple(s.frame or scenario.frame for s in specs)
    base_pos = [list(s.initial_position_m) for s in specs]
    base_vel = [s.velocity_mps for s in specs]

    env_pos: list[list[list[float]]] = []
    env_vel: list[list[Any]] = []
    env_soc: list[list[float]] = []
    for e in env_indices:
        env_key = jax.random.fold_in(key, e)
        env_pos.append([list(p) for p in base_pos])
        env_vel.append(
            [
                sample_initial_velocity(jax.random.fold_in(env_key, a), base_vel[a], jitter)
                for a in range(len(specs))
            ]
        )
        env_soc.append([s.battery_soc_j for s in specs])

    state = _BatchState(
        pos=jnp.asarray(env_pos, dtype=jnp.float64),
        vel=jnp.asarray(env_vel, dtype=jnp.float64),
        soc=jnp.asarray(env_soc, dtype=jnp.float64),
    )
    floors = jnp.asarray([s.battery_floor_j for s in specs], dtype=jnp.float64)
    return VectorizedRollout(
        state=state,
        params=params,
        agent_ids=agent_ids,
        frames=frames,
        commanded=list(base_vel),
        modes=[s.mode for s in specs],
        floors=floors,
        env_indices=env_indices,
        dt_s=scenario.dt_s,
    )


# --- the MJX contact batch (RM-P1-SIM-04): the same surface, real contact physics -------------

#: The MJX kernel mapped over agents (inner) then envs (outer) and **JIT-compiled**: one XLA call
#: steps the whole ``(N_envs, N_agents)`` batch of **contact solves**. This is the vectorization
#: that makes real wheel-soil contact a training-speed tier (sim.md §8).
#:
#: ``jax.jit`` (with ``steps`` static) is load-bearing, not decoration: without it XLA re-traces and
#: re-compiles MJX's whole constraint-solver graph on *every* call — tens of seconds per step on
#: CPU.
#: Under ``jit`` the compile is cached on the batch's shapes/dtypes, so a rollout compiles once and
#: every subsequent step is a cache hit, which is the entire premise of a batched training tier.
_mjx_agents_rollout = jax.vmap(mjx_rollout, in_axes=(None, 0, 0, None))
_mjx_envs_rollout = jax.jit(
    jax.vmap(_mjx_agents_rollout, in_axes=(None, 0, 0, None)), static_argnums=(3,)
)


class MjxVectorizedRollout:
    """N parallel **MJX contact** envs stepped as one ``jax.vmap`` batch (RM-P1-SIM-04).

    The contact-physics counterpart of :class:`VectorizedRollout`, exposing the *identical* surface
    (:class:`RolloutBatch`) so the Ray fan-out and its aggregation oracle drive it unchanged. Each
    env/agent runs a real ``mjx.step`` constraint solve over the shared articulated wheel-soil rover
    (:mod:`astro_mine.sim.engines._rover_mjcf`) — the same machine the MuJoCo CPU tier steps.

    Rows are ordered by ``env_indices`` (the global env ids used for seeding), so a shard's rows
    line up with the same rows of a whole-batch run — the property the fan-out's aggregation relies
    on."""

    def __init__(
        self,
        *,
        mx: Any,
        data: Any,
        spec: RoverModelSpec,
        agent_ids: tuple[str, ...],
        frames: tuple[ReferenceFrame, ...],
        origins: tuple[tuple[float, float, float], ...],
        commanded: list[tuple[float, float, float]],
        modes: list[str | None],
        socs: Any,
        floors: Any,
        #: Per-agent ``(idle_power_w, drive_power_w_per_mps)`` battery-draw terms.
        power_terms: list[tuple[float, float]],
        env_indices: tuple[int, ...],
        dt_s: float,
    ) -> None:
        self._mx = mx
        self._init_data = data
        self._data = data
        self._spec = spec
        self._agent_ids = agent_ids
        self._frames = frames
        self._origins = origins
        self._commanded0 = list(commanded)
        self._modes0 = list(modes)
        self._commanded = list(commanded)
        self._modes = list(modes)
        self._init_soc = socs
        self._soc = socs
        self._floors = floors
        self._idle = jnp.asarray([p[0] for p in power_terms], dtype=jnp.float64)
        self._drive = jnp.asarray([p[1] for p in power_terms], dtype=jnp.float64)
        self._env_indices = env_indices
        self._dt_s = dt_s
        self._substeps = max(1, round(dt_s / spec.timestep_s))
        self._tick = 0
        self._elapsed_s = 0.0

    @property
    def n_envs(self) -> int:
        """The number of parallel envs (the leading/batch axis)."""
        return len(self._env_indices)

    @property
    def n_agents(self) -> int:
        """The number of agents per env."""
        return len(self._agent_ids)

    @property
    def env_indices(self) -> tuple[int, ...]:
        """The global env ids this rollout covers (its rows), in row order."""
        return self._env_indices

    @property
    def positions(self) -> Any:
        """The live ``(N, A, 3)`` position array — the MJX free-joint pose plus each agent's
        origin."""
        origins = jnp.asarray(self._origins, dtype=jnp.float64)  # (A, 3)
        return self._data.qpos[..., 0:3] + origins

    def reset(self) -> tuple[tuple[Observation, ...], ...]:
        """Rewind to the initial batched contact state and return the per-env initial
        observations."""
        self._data = self._init_data
        self._soc = self._init_soc
        self._commanded = list(self._commanded0)
        self._modes = list(self._modes0)
        self._tick = 0
        self._elapsed_s = 0.0
        return self._observations()

    def step(self, actions: ActionBatch) -> tuple[tuple[Observation, ...], ...]:
        """Advance every env one tick of **contact simulation** under ``actions`` (broadcast).

        The commands become per-wheel speed setpoints, sub-stepped through ``mjx.step`` to fill the
        macro ``dt`` — whether a rover actually reaches its setpoint is the friction cone's
        business,
        which is exactly what this tier exists to model."""
        self._apply(actions)
        ctrl = self._wheel_setpoints()
        self._data = _mjx_envs_rollout(self._mx, self._data, ctrl, self._substeps)
        speeds = jnp.linalg.norm(self._data.qvel[..., 0:3], axis=-1)  # (N, A)
        draw = (self._idle + self._drive * speeds) * self._dt_s
        self._soc = jnp.maximum(self._floors, self._soc - draw)
        self._tick += 1
        self._elapsed_s += self._dt_s
        return self._observations()

    def _apply(self, actions: ActionBatch) -> None:
        index = {aid: i for i, aid in enumerate(self._agent_ids)}
        for agent_id, action in actions_by_agent(actions).items():
            i = index.get(agent_id)
            if i is None:
                continue
            if action.kind is ActionKind.MODE and action.mode is not None:
                self._modes[i] = action.mode.mode
            elif (
                action.kind is ActionKind.ACTUATOR
                and action.actuator is not None
                and action.actuator.control_mode is ControlMode.VELOCITY
                and len(action.actuator.setpoint) == 3
            ):
                sx, sy, sz = action.actuator.setpoint
                self._commanded[i] = (sx, sy, sz)

    def _wheel_setpoints(self) -> Any:
        """The per-agent wheel angular-velocity setpoint, broadcast to ``(N, A, nu)``.

        The same reduced-order drive controller the CPU contact tier uses: commanded speed (capped
        at
        top speed) maps to a wheel speed ``v / r``, and the contact solve decides the rest."""
        per_agent = []
        for v in self._commanded:
            speed = min(self._spec.max_speed_mps, math.dist(v, (0.0, 0.0, 0.0)))
            direction = 1.0 if v[0] >= 0.0 else -1.0
            per_agent.append([direction * speed / self._spec.wheel_radius_m] * self._mx.nu)
        stacked = jnp.asarray(per_agent, dtype=jnp.float64)  # (A, nu)
        return jnp.broadcast_to(stacked, (self.n_envs, self.n_agents, self._mx.nu))

    def _observations(self) -> tuple[tuple[Observation, ...], ...]:
        return tuple(
            tuple(self._observe(e, a) for a in range(self.n_agents)) for e in range(self.n_envs)
        )

    def _observe(self, e: int, a: int) -> Observation:
        q = self._data.qpos[e, a]
        v = self._data.qvel[e, a]
        origin = self._origins[a]
        self_state = StateSample(
            agent_id=self._agent_ids[a],
            frame=self._frames[a],
            pose=Transform(
                translation_m=Vec3(
                    x=float(q[0]) + origin[0],
                    y=float(q[1]) + origin[1],
                    z=float(q[2]) + origin[2],
                ),
                # A contact-simulated chassis really does pitch and roll, so unlike the
                # reduced-order
                # kernel the attitude here is genuine (MuJoCo stores it scalar-first).
                rotation_quat_xyzw=Quat(x=float(q[4]), y=float(q[5]), z=float(q[6]), w=float(q[3])),
            ),
            linear_velocity_mps=Vec3(x=float(v[0]), y=float(v[1]), z=float(v[2])),
            battery_soc_j=float(self._soc[e, a]),
            mode=self._modes[a],
        )
        return Observation(
            tick=self._tick,
            sim_time_s=self._elapsed_s,
            agent_id=self._agent_ids[a],
            self_state=self_state,
        )


def build_mjx_vectorized_rollout(
    scenario: Scenario,
    rng: RngStreams,
    *,
    n_envs: int | None = None,
    env_indices: Sequence[int] | None = None,
) -> MjxVectorizedRollout:
    """Build an :class:`MjxVectorizedRollout` over the scenario's ``mjx_contact`` agents.

    ``env_indices`` selects the **global** env ids this rollout covers (a Ray shard passes its
    subset); default is ``range(n_envs)``, and ``n_envs`` defaults to the ``batch_size`` the first
    ``mjx_contact`` agent declares. Each env/agent's initial velocity is seeded from
    ``fold_in(root_key, env_index)`` then per agent, so a shard reproduces the whole-batch rows
    exactly — the same global-index seeding discipline the reduced-order tier uses, which is what
    lets the Ray fan-out's aggregation oracle hold for this tier too.

    All ``mjx_contact`` agents share one compiled MJX model (a single ``mjx.put_model``), so one
    vmapped call steps the whole batch; a mismatch raises ``ValueError``."""
    specs = [s for s in scenario.agents if s.dynamics.kind == "mjx_contact"]
    if not specs:
        raise ValueError("a vectorized MJX rollout needs at least one mjx_contact agent")
    first = specs[0].dynamics
    assert first.kind == "mjx_contact"  # narrows the union
    spec = model_spec_from_mjx_dynamics(first)
    for other in specs[1:]:
        dyn = other.dynamics
        assert dyn.kind == "mjx_contact"
        if model_spec_from_mjx_dynamics(dyn) != spec:
            raise ValueError(
                "mjx_contact agents must share one physical model for the batched MJX step; "
                f"agent {other.agent_id!r} differs"
            )

    if env_indices is None:
        count = n_envs if n_envs is not None else first.batch_size
        env_indices = tuple(range(count))
    else:
        env_indices = tuple(env_indices)
    if not env_indices:
        raise ValueError("a vectorized MJX rollout needs at least one env")

    model = mujoco.MjModel.from_xml_string(rover_mjcf(spec))
    mx = mjx.put_model(model)
    single = mjx.make_data(mx)
    n_envs_actual, n_agents = len(env_indices), len(specs)
    # One mjx.Data per (env, agent): stack the template out to the (N, A, ...) batch shape.
    data = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (n_envs_actual, n_agents, *leaf.shape)).copy(), single
    )

    key = brax_prng_key(rng)
    jitter = first.init_speed_jitter_mps
    velocities: list[list[Any]] = []
    for e in env_indices:
        env_key = jax.random.fold_in(key, e)
        velocities.append(
            [
                sample_initial_velocity(
                    jax.random.fold_in(env_key, a), specs[a].velocity_mps, jitter
                )
                for a in range(n_agents)
            ]
        )
    qvel = data.qvel.at[..., 0:3].set(jnp.asarray(velocities, dtype=jnp.float64))
    data = data.replace(qvel=qvel)

    socs = jnp.broadcast_to(
        jnp.asarray([s.battery_soc_j for s in specs], dtype=jnp.float64),
        (n_envs_actual, n_agents),
    )
    floors = jnp.asarray([s.battery_floor_j for s in specs], dtype=jnp.float64)
    return MjxVectorizedRollout(
        mx=mx,
        data=data,
        spec=spec,
        agent_ids=tuple(s.agent_id for s in specs),
        frames=tuple(s.frame or scenario.frame for s in specs),
        origins=tuple(s.initial_position_m for s in specs),
        commanded=[s.velocity_mps for s in specs],
        modes=[s.mode for s in specs],
        socs=socs,
        floors=floors,
        power_terms=[
            (s.dynamics.idle_power_w, s.dynamics.drive_power_w_per_mps)  # type: ignore[union-attr]
            for s in specs
        ],
        env_indices=env_indices,
        dt_s=scenario.dt_s,
    )
