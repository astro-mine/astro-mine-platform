"""The MJX contact kernel — real Brax/MJX wheel-soil contact, GPU-vectorized (RM-P1-SIM-04).

The physics-fidelity upgrade the sibling :mod:`._engine` deferred. That module's kernel is
algebraically the reduced-order kinematic mobility model re-expressed in ``jax.numpy``: fast,
``vmap``-able, and useful for very large sweeps, but it contains no contact. **This** module
compiles the *same articulated wheel-soil rover* the MuJoCo CPU tier steps
(:mod:`astro_mine.sim.engines._rover_mjcf`) through **MuJoCo MJX** — ``mjx.put_model`` /
``mjx.step``, the JAX reimplementation of MuJoCo's constraint solver — and ``jax.vmap``s it across
parallel envs. So the batched training tier now runs real frictional contact: wheels that roll,
slip, and sink, at swarm scale on a GPU (sim.md §8, §11).

The seams are unchanged by design (this is a fidelity upgrade *behind* the same architecture, not a
new integration surface):

- it is a :class:`~astro_mine.sim.engines.RegimeEngine` like every other engine, so it routes behind
  the Core Environment waist and is selected by ``dynamics.kind`` — configuration, not code;
- it exposes the same :class:`~astro_mine.sim.engines.brax._batch.VectorizedRollout` batching
surface
  the reduced-order JAX kernel does, so the in-process Ray fan-out and its aggregation oracle
  (:mod:`._ray`) drive it unchanged.

**Determinism (conventions.md §11).** ``TOLERANCE``, not ``BIT_EXACT``, and deliberately so — see
:data:`~astro_mine.sim.engines.brax._mjx_descriptor.MJX_CONTACT_ENGINE_DESCRIPTOR` for the full
statement. In-process, same-device, same-seed runs reproduce (all randomness flows through a seeded
``jax.random`` key folded from the ``RngStreams`` root, and ``mjx.step`` is a pure function of its
inputs); across devices/XLA versions the last bits differ, and a stiff contact solve *amplifies*
that, so the tier is gated by the analytic drawbar-pull error budget rather than a golden hash.

x64 is enabled process-wide (as in :mod:`._engine`) so battery-SoC joules and positions stay
physically meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.sim.engines._rover_mjcf import RoverModelSpec, rover_mjcf
from astro_mine.sim.engines.actuation import actions_by_agent
from astro_mine.sim.engines.adapter import CouplingState, EngineDescriptor
from astro_mine.sim.engines.brax._mjx_descriptor import MJX_CONTACT_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from collections.abc import Iterable

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.core.units import ReferenceFrame
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import MjxContactDynamics, Scenario

__all__ = [
    "MjxContactEngine",
    "build_mjx_contact_engine",
    "mjx_rollout",
    "model_spec_from_mjx_dynamics",
]

# Physically meaningful SoC (joules) and positions demand float64; JAX defaults to float32.
jax.config.update("jax_enable_x64", True)

#: The engine's named RNG sub-stream (folded from the RngStreams root) that seeds the JAX PRNGKey.
_RNG_STREAM = "mjx_contact"


def model_spec_from_mjx_dynamics(dyn: MjxContactDynamics) -> RoverModelSpec:
    """The shared rover contact model this agent's dynamics block describes.

    The *same* :class:`~astro_mine.sim.engines._rover_mjcf.RoverModelSpec` the MuJoCo CPU tier
    builds,
    so the two contact tiers cannot silently disagree about what a rover is."""
    return RoverModelSpec(
        mass_kg=dyn.mass_kg,
        body_half_extents_m=dyn.body_half_extents_m,
        wheel_radius_m=dyn.wheel_radius_m,
        wheel_width_m=dyn.wheel_width_m,
        wheel_mass_kg=dyn.wheel_mass_kg,
        wheel_torque_nm=dyn.wheel_torque_nm,
        max_speed_mps=dyn.max_speed_mps,
        gravity_m_s2=dyn.gravity_m_s2,
        friction_angle_deg=dyn.friction_angle_deg,
        bearing_capacity_pa=dyn.bearing_capacity_pa,
        timestep_s=dyn.timestep_s,
    )


def mjx_rollout(mx: Any, data: Any, ctrl: Any, steps: int) -> Any:
    """Step the MJX model ``steps`` times under a fixed wheel-speed command — the pure JAX kernel.

    A ``jax.lax.scan`` over ``mjx.step`` (MuJoCo's constraint solver, in JAX), so XLA compiles the
    whole rollout once and ``jax.vmap`` can map it across a batch axis. ``ctrl`` is the per-wheel
    angular-velocity setpoint ``(nu,)``; ``data`` is an ``mjx.Data`` (or a batch of them, under
    vmap).
    """

    def body(d: Any, _: Any) -> tuple[Any, None]:
        return mjx.step(mx, d.replace(ctrl=ctrl)), None

    stepped: Any
    stepped, _ = jax.lax.scan(body, data, None, length=steps)
    return stepped


#: The kernel mapped over a batch (agent/env) axis and **JIT-compiled**: each row carries its own
#: ``mjx.Data`` and its own wheel command. This is the vectorization that makes the tier a
#: *training* engine — thousands of parallel contact rollouts per GPU (sim.md §8).
#:
#: ``jax.jit`` is load-bearing, not decoration: without it XLA re-traces and re-compiles the whole
#: constraint-solver graph on **every** call (MJX's step is a large graph — tens of seconds on CPU).
#: Under ``jit`` the compile is cached on the argument shapes/dtypes plus the static ``steps``, so a
#: rollout compiles once and every subsequent step of the same batch shape is a cache hit — which is
#: the entire premise of the tier.
_batched_rollout = jax.jit(jax.vmap(mjx_rollout, in_axes=(None, 0, 0, None)), static_argnums=(3,))


@dataclass
class _MjxRoverState:
    """One rover's Python-side command + accounting state; the physics lives in the MJX data."""

    agent_id: str
    frame: ReferenceFrame
    commanded_velocity_mps: tuple[float, float, float]
    goto_point_m: tuple[float, float, float] | None
    origin_m: tuple[float, float, float]
    idle_power_w: float
    drive_power_w_per_mps: float
    battery_soc_j: float
    battery_floor_j: float
    mode: str | None


class MjxContactEngine:
    """The MJX contact :class:`~astro_mine.sim.engines.RegimeEngine` — batched wheel-soil contact.

    Owns one ``mjx.Data`` per rover, stacked along a leading agent axis, and steps them all through
    the ``jax.vmap``-ed :func:`mjx_rollout` kernel — the same kernel
    :class:`~astro_mine.sim.engines.brax._batch.VectorizedRollout` maps a second time across
    parallel envs. Behaviourally it is the MuJoCo CPU contact tier (it compiles the identical
    model), so the analytic drawbar-pull oracle cross-checks it; its distinction is that its step is
    an XLA-compiled
    batched function."""

    def __init__(
        self,
        states: dict[str, _MjxRoverState],
        mx: Any,
        data: Any,
        spec: RoverModelSpec,
    ) -> None:
        self._states = states
        self._order = tuple(states)
        self._index = {agent_id: i for i, agent_id in enumerate(self._order)}
        self._mx = mx
        self._data = data  # a batch of mjx.Data, stacked along the agent axis
        self._spec = spec
        self._elapsed_s = 0.0

    @property
    def descriptor(self) -> EngineDescriptor:
        return MJX_CONTACT_ENGINE_DESCRIPTOR

    def apply_actions(self, actions: ActionBatch) -> None:
        """Set a velocity setpoint, a goto target, or a mode for owned rovers — the same command
        surface every mobility tier exposes, so a policy cannot tell which tier it is driving."""
        for agent_id, action in actions_by_agent(actions).items():
            state = self._states.get(agent_id)
            if state is None:
                continue
            if action.kind is ActionKind.MODE and action.mode is not None:
                state.mode = action.mode.mode
            elif (
                action.kind is ActionKind.ACTUATOR
                and action.actuator is not None
                and action.actuator.control_mode is ControlMode.VELOCITY
                and len(action.actuator.setpoint) == 3
            ):
                sx, sy, sz = action.actuator.setpoint
                state.commanded_velocity_mps = (sx, sy, sz)
                state.goto_point_m = None
            elif (
                action.kind is ActionKind.TASK
                and action.task is not None
                and action.task.goto is not None
            ):
                t = action.task.goto.target_pose.translation_m
                state.goto_point_m = (t.x, t.y, t.z)

    def advance(self, dt_s: float) -> None:
        """Step every rover's MJX contact solve forward ``dt_s`` through the batched kernel."""
        if not self._order:  # owns no mjx_contact agents (heterogeneous co-step) — just tick
            self._elapsed_s += dt_s
            return
        substeps = max(1, round(dt_s / self._spec.timestep_s))
        ctrl = jnp.asarray(
            [[self._wheel_setpoint(state, dt_s)] * self._mx.nu for state in self._states.values()],
            dtype=jnp.float64,
        )
        self._data = _batched_rollout(self._mx, self._data, ctrl, substeps)
        for state in self._states.values():
            speed = math.dist(self._velocity_of(state.agent_id), (0.0, 0.0, 0.0))
            draw = (state.idle_power_w + state.drive_power_w_per_mps * speed) * dt_s
            state.battery_soc_j = max(state.battery_floor_j, state.battery_soc_j - draw)
        self._elapsed_s += dt_s

    def _wheel_setpoint(self, state: _MjxRoverState, dt_s: float) -> float:
        """The wheel angular-velocity setpoint (rad/s) tracking the rover's commanded motion —
        the same reduced-order drive controller the MuJoCo CPU tier uses, so the two tiers are
        commanded identically and only their execution substrate differs."""
        if state.goto_point_m is not None:
            position = self._position_of(state.agent_id)
            to_target = tuple(g - p for g, p in zip(state.goto_point_m, position, strict=True))
            distance = math.dist(to_target, (0.0, 0.0, 0.0))
            if distance == 0.0:
                return 0.0
            speed = min(self._spec.max_speed_mps, distance / dt_s)
            direction = 1.0 if to_target[0] >= 0.0 else -1.0
        else:
            commanded = state.commanded_velocity_mps
            speed = min(self._spec.max_speed_mps, math.dist(commanded, (0.0, 0.0, 0.0)))
            direction = 1.0 if commanded[0] >= 0.0 else -1.0
        return direction * speed / self._spec.wheel_radius_m

    def _position_of(self, agent_id: str) -> tuple[float, float, float]:
        i = self._index[agent_id]
        origin = self._states[agent_id].origin_m
        q = self._data.qpos[i]
        return (
            float(q[0]) + origin[0],
            float(q[1]) + origin[1],
            float(q[2]) + origin[2],
        )

    def _velocity_of(self, agent_id: str) -> tuple[float, float, float]:
        v = self._data.qvel[self._index[agent_id]]
        return (float(v[0]), float(v[1]), float(v[2]))

    def export_coupling_state(self) -> CouplingState:
        return CouplingState(
            sim_time_s=self._elapsed_s,
            samples=tuple(self._sample(agent_id) for agent_id in self._order),
        )

    def _sample(self, agent_id: str) -> StateSample:
        i = self._index[agent_id]
        state = self._states[agent_id]
        px, py, pz = self._position_of(agent_id)
        vx, vy, vz = self._velocity_of(agent_id)
        # MuJoCo's free joint stores the quaternion scalar-FIRST (w, x, y, z); Core is scalar-last.
        q = self._data.qpos[i]
        qw, qx, qy, qz = (float(q[3]), float(q[4]), float(q[5]), float(q[6]))
        return StateSample(
            agent_id=agent_id,
            frame=state.frame,
            pose=Transform(
                translation_m=Vec3(x=px, y=py, z=pz),
                rotation_quat_xyzw=Quat(x=qx, y=qy, z=qz, w=qw),
            ),
            linear_velocity_mps=Vec3(x=vx, y=vy, z=vz),
            battery_soc_j=state.battery_soc_j,
            mode=state.mode,
        )

    def import_coupling_state(self, state: CouplingState) -> None:
        """Overwrite live rovers from a boundary snapshot — writing the incoming pose/velocity into
        the batched MJX free-joint state (the coupler's cross-engine handoff)."""
        if not self._order:
            self._elapsed_s = state.sim_time_s
            return
        incoming = state.by_agent
        qpos = [list(map(float, self._data.qpos[i])) for i in range(len(self._order))]
        qvel = [list(map(float, self._data.qvel[i])) for i in range(len(self._order))]
        for agent_id, current in self._states.items():
            sample = incoming.get(agent_id)
            if sample is None:
                continue
            i = self._index[agent_id]
            t = sample.pose.translation_m
            qpos[i][0:3] = [
                t.x - current.origin_m[0],
                t.y - current.origin_m[1],
                t.z - current.origin_m[2],
            ]
            q = sample.pose.rotation_quat_xyzw
            qpos[i][3:7] = [q.w, q.x, q.y, q.z]  # Core scalar-last -> MuJoCo scalar-first
            velocity = sample.linear_velocity_mps
            if velocity is not None:
                qvel[i][0:3] = [velocity.x, velocity.y, velocity.z]
            if sample.battery_soc_j is not None:
                current.battery_soc_j = sample.battery_soc_j
            if sample.mode is not None:
                current.mode = sample.mode
        self._data = self._data.replace(
            qpos=jnp.asarray(qpos, dtype=jnp.float64),
            qvel=jnp.asarray(qvel, dtype=jnp.float64),
        )
        self._elapsed_s = state.sim_time_s

    def retire(self, agent_ids: Iterable[str]) -> None:
        drop = {aid for aid in agent_ids if aid in self._states}
        if not drop:
            return
        keep = [aid for aid in self._order if aid not in drop]
        rows = [self._index[aid] for aid in keep]
        self._states = {aid: self._states[aid] for aid in keep}
        self._order = tuple(keep)
        self._index = {aid: i for i, aid in enumerate(keep)}
        index = jnp.asarray(rows, dtype=jnp.int32)
        self._data = jax.tree.map(lambda leaf: leaf[index], self._data)


def build_mjx_contact_engine(scenario: Scenario, rng: RngStreams) -> MjxContactEngine:
    """Build an :class:`MjxContactEngine` for the scenario's ``mjx_contact`` agents.

    All ``mjx_contact`` agents share one compiled MJX model (a single ``mjx.put_model``), so one
    vmapped ``mjx.step`` covers the whole set — that shared compilation is the whole point of the
    tier. A mismatch in the physical model between agents raises ``ValueError``. Each rover's
    initial velocity is seeded from a JAX key folded from the ``RngStreams`` root, so the
    domain-randomization
    jitter (off by default) reproduces. Non-``mjx_contact`` agents are skipped."""
    key = jax.random.PRNGKey(rng.stream(_RNG_STREAM).getrandbits(32))
    states: dict[str, _MjxRoverState] = {}
    spec: RoverModelSpec | None = None
    velocities: list[Any] = []
    for i, agent in enumerate(a for a in scenario.agents if a.dynamics.kind == "mjx_contact"):
        dyn = agent.dynamics
        assert dyn.kind == "mjx_contact"  # narrows the union for type-checkers
        agent_spec = model_spec_from_mjx_dynamics(dyn)
        if spec is None:
            spec = agent_spec
        elif agent_spec != spec:
            raise ValueError(
                "mjx_contact agents must share one physical model for the batched MJX step; "
                f"agent {agent.agent_id!r} differs"
            )
        agent_key = jax.random.fold_in(key, i)
        base = jnp.asarray(agent.velocity_mps, dtype=jnp.float64)
        velocities.append(
            base + jax.random.normal(agent_key, (3,), dtype=jnp.float64) * dyn.init_speed_jitter_mps
        )
        states[agent.agent_id] = _MjxRoverState(
            agent_id=agent.agent_id,
            frame=agent.frame or scenario.frame,
            commanded_velocity_mps=agent.velocity_mps,
            goto_point_m=None,
            origin_m=agent.initial_position_m,
            idle_power_w=dyn.idle_power_w,
            drive_power_w_per_mps=dyn.drive_power_w_per_mps,
            battery_soc_j=agent.battery_soc_j,
            battery_floor_j=agent.battery_floor_j,
            mode=agent.mode,
        )
    if spec is None:  # no mjx_contact agents — an empty engine (heterogeneous co-step)
        return MjxContactEngine({}, _null_model(), None, RoverModelSpec(mass_kg=1.0))

    model = mujoco.MjModel.from_xml_string(rover_mjcf(spec))
    mx = mjx.put_model(model)
    single = mjx.make_data(mx)
    # Stack one mjx.Data per agent along a leading axis, then write each agent's initial velocity
    # in.
    data = jax.tree.map(lambda leaf: jnp.stack([leaf] * len(states)), single)
    qvel = data.qvel.at[:, 0:3].set(jnp.stack(velocities))
    data = data.replace(qvel=qvel)
    return MjxContactEngine(states, mx, data, spec)


def _null_model() -> Any:
    """A placeholder MJX model for an engine that owns no ``mjx_contact`` agents (nothing to
    step)."""
    return mjx.put_model(mujoco.MjModel.from_xml_string(rover_mjcf(RoverModelSpec(mass_kg=1.0))))
