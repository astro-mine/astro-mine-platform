"""The Brax/MJX contact :class:`RegimeEngine` — the JAX GPU-vectorizable mobility tier (SIM-04).

A minimal, **JAX-native** low-fidelity surface-mobility/contact model behind the ``RegimeEngine``
waist: a traction-limited, speed-capped point-mass rover integrated by a pure ``jax.numpy`` kernel
that is ``jax.vmap``-batchable across agents *and* across thousands of parallel envs (the batched
swarm-scale rollout path lives in :mod:`._batch`). Kept deliberately MVP — enough to prove the
vectorization seam and the tolerance gate; richer Brax/MJX contact (soft/hydroelastic contact,
articulated wheels) is a follow-up behind this same descriptor.

The kernel is algebraically the reduced-order mobility model
(:class:`~astro_mine.sim.engines.mobility.MobilityEngine`), so the analytic drawbar-pull oracle
cross-checks this tier within a tolerance (sim.md §11); the value of *this* tier is that its step is
a JAX function XLA compiles once and runs batched on a GPU (sim.md §8 "thousands of low-fidelity
parallel rollouts per GPU-hour").

**JAX lives here, never at package import.** :mod:`astro_mine.sim.engines.brax` imports this only
inside its factory, so the base wheel (and ``builtins.py``) stay JAX-free (the ``[brax]`` extra).
x64 is enabled process-wide on import so battery-SoC accounting (joules, ~1e6-1e9) and positions
stay physically meaningful — float32 would swamp per-step draws — and so the mobility oracle
cross-check stays tight. Determinism is ``TOLERANCE`` (XLA reductions are non-associative / not
bit-portable across builds); same-seed runs reproduce in-process because all randomness flows
through a seeded :class:`jax.random` key folded from the
:class:`~astro_mine.sim.runtime.rng.RngStreams` root.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.sim.engines.actuation import actions_by_agent
from astro_mine.sim.engines.adapter import CouplingState, EngineDescriptor
from astro_mine.sim.engines.brax._descriptor import BRAX_CONTACT_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.core.units import ReferenceFrame
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import BraxContactDynamics, Scenario

__all__ = [
    "BraxContactEngine",
    "BraxParams",
    "agent_step",
    "brax_prng_key",
    "build_brax_contact_engine",
    "sample_initial_velocity",
]

# Physically meaningful SoC (joules, ~1e6-1e9) and positions demand float64; JAX defaults to
# float32, which would round per-step battery draws to nothing. Enabling x64 here (the module is
# only imported by the [brax] factory) keeps the tier honest and the mobility oracle cross-check
# tight, while determinism stays TOLERANCE — x64 does not make XLA reductions bit-portable.
jax.config.update("jax_enable_x64", True)

_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)

#: The engine's named RNG sub-stream (folded from the RngStreams root seed via SHA-256, like the
#: DEM engine's per-agent packing seed) that seeds the JAX ``PRNGKey``.
_RNG_STREAM = "brax_contact"


@dataclass(frozen=True, slots=True)
class BraxParams:
    """Immutable per-rover physical parameters of the JAX mobility kernel (SI).

    Mirrors the reduced-order mobility model: acceleration is capped by the terramechanics
    drawbar-pull limit ``max_traction_n / mass_kg`` and speed by ``max_speed_mps``; battery draw
    is ``idle_power_w`` plus ``drive_power_w_per_mps`` * speed. Passed to the kernel as a flat
    tuple of floats (a JAX pytree leaf set) so ``jax.vmap`` broadcasts it with ``in_axes=None``."""

    mass_kg: float
    max_speed_mps: float
    max_traction_n: float
    idle_power_w: float
    drive_power_w_per_mps: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        """The kernel's ``prm`` argument — a flat, vmap-broadcastable float tuple."""
        return (
            self.mass_kg,
            self.max_speed_mps,
            self.max_traction_n,
            self.idle_power_w,
            self.drive_power_w_per_mps,
        )


def brax_prng_key(rng: RngStreams) -> Any:
    """The engine's root :class:`jax.random` key, folded deterministically from ``rng``.

    Draws a 32-bit seed from the engine's named RngStreams sub-stream (the same SHA-256 fan-out
    every other engine seeds through), so same-root-seed runs get byte-identical keys — the
    in-process reproducibility half of the ``TOLERANCE`` contract."""
    seed = rng.stream(_RNG_STREAM).getrandbits(32)
    return jax.random.PRNGKey(seed)


def sample_initial_velocity(
    key: Any, base_velocity_mps: tuple[float, float, float], jitter_mps: float
) -> Any:
    """A seeded initial velocity ``(3,)`` — the base velocity plus optional domain-randomization.

    ``jitter_mps`` (0 by default) is the std-dev of a per-env/per-agent Gaussian perturbation for
    swarm-scale domain randomization; at ``0.0`` the draw contributes nothing and the tier reduces
    exactly to the deterministic reduced-order model (so the mobility oracle cross-check is tight).
    The draw itself always runs, so the seeded key is genuinely load-bearing."""
    base = jnp.asarray(base_velocity_mps, dtype=jnp.float64)
    return base + jax.random.normal(key, (3,), dtype=jnp.float64) * jitter_mps


def agent_step(carry: Any, target_vel: Any, prm: Any, dt: float) -> Any:
    """One integration step for a single agent — the pure, ``vmap``-able JAX kernel.

    ``carry`` is ``(position, velocity, soc)`` (each a JAX array); ``target_vel`` is the desired
    velocity this step (already clamped to top speed on the host); ``prm`` is
    :meth:`BraxParams.as_tuple`. Ramps ``velocity`` toward ``target_vel`` under the traction-limited
    acceleration, caps the speed, advances the position, and draws the battery — the mobility model,
    in ``jax.numpy`` so XLA batches it over agents and envs. Returns the updated carry."""
    pos, vel, soc = carry
    mass, max_speed, max_traction, idle_power, drive_power = prm

    delta = target_vel - vel
    delta_norm = jnp.linalg.norm(delta)
    max_delta = (max_traction / mass) * dt
    safe_delta = jnp.where(delta_norm > 0.0, delta_norm, 1.0)
    factor = jnp.where(delta_norm > max_delta, max_delta / safe_delta, 1.0)
    vel = vel + delta * factor

    speed = jnp.linalg.norm(vel)
    safe_speed = jnp.where(speed > 0.0, speed, 1.0)
    vel = jnp.where(speed > max_speed, vel * (max_speed / safe_speed), vel)

    pos = pos + vel * dt
    draw = (idle_power + drive_power * jnp.linalg.norm(vel)) * dt
    soc = soc - draw
    return pos, vel, soc


#: The kernel mapped over the agent axis (axis 0 of the stacked ``(A, …)`` arrays); ``prm``/``dt``
#: are broadcast (``in_axes=None``). :mod:`._batch` maps a second time over the env axis.
_agents_step = jax.vmap(agent_step, in_axes=((0, 0, 0), 0, None, None))


@dataclass(slots=True)
class _BraxRoverState:
    """Per-rover command + accounting state (Python side); kinematics live in the JAX arrays."""

    agent_id: str
    frame: ReferenceFrame
    commanded_velocity_mps: tuple[float, float, float]
    goto_point_m: tuple[float, float, float] | None
    battery_floor_j: float
    mode: str | None


class BraxContactEngine:
    """The JAX GPU-vectorizable mobility :class:`~astro_mine.sim.engines.RegimeEngine` (single env).

    Owns its rovers' kinematics as stacked JAX arrays and steps them through the ``jax.vmap``-ed
    :func:`agent_step` kernel — the same kernel :mod:`._batch` maps a second time to fan out
    thousands of parallel envs on a GPU. Behaviourally it is the reduced-order mobility model, so
    it plugs in behind the identical waist and is oracle-cross-checked; its distinction is that its
    step is an XLA-compiled batched function (sim.md §8, §11)."""

    def __init__(
        self,
        states: dict[str, _BraxRoverState],
        pos: Any,
        vel: Any,
        soc: Any,
        params: BraxParams,
    ) -> None:
        self._states = states
        self._order = tuple(states)  # stable agent order aligned with the array rows
        self._index = {agent_id: i for i, agent_id in enumerate(self._order)}
        self._pos = pos
        self._vel = vel
        self._soc = soc
        self._params = params
        self._elapsed_s = 0.0

    @property
    def descriptor(self) -> EngineDescriptor:
        return BRAX_CONTACT_ENGINE_DESCRIPTOR

    def apply_actions(self, actions: ActionBatch) -> None:
        """Set a velocity setpoint, a goto target, or a mode for owned rovers (mirrors mobility).

        A ``VELOCITY`` actuator command sets the commanded velocity (and clears any goto); a
        ``goto`` task sets the target point; a mode command sets the mode. The subsequent
        :meth:`advance` ramps toward the resolved desired velocity under the traction cap."""
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
        """Step every owned rover forward ``dt_s`` through the batched JAX kernel.

        The per-agent desired velocity is resolved on the host (goto-vs-command, top-speed clamp —
        the reduced-order policy), stacked into a ``(A, 3)`` array, and handed to the vmapped
        kernel; the battery-floor clamp is applied to the returned SoC so a rover never charges past
        empty. The clock advances by ``dt_s``."""
        if not self._order:  # owns no brax_contact agents (heterogeneous co-step) — just tick
            self._elapsed_s += dt_s
            return
        max_speed = self._params.max_speed_mps
        targets = jnp.asarray(
            [
                _desired_velocity(state, self._row_pos(state.agent_id), max_speed, dt_s)
                for state in self._states.values()
            ],
            dtype=jnp.float64,
        )
        pos, vel, soc = _agents_step(
            (self._pos, self._vel, self._soc), targets, self._params.as_tuple(), dt_s
        )
        floors = jnp.asarray(
            [state.battery_floor_j for state in self._states.values()], dtype=jnp.float64
        )
        self._pos, self._vel = pos, vel
        self._soc = jnp.maximum(floors, soc)
        self._elapsed_s += dt_s

    def _row_pos(self, agent_id: str) -> tuple[float, float, float]:
        row = self._pos[self._index[agent_id]]
        return (float(row[0]), float(row[1]), float(row[2]))

    def export_coupling_state(self) -> CouplingState:
        return CouplingState(
            sim_time_s=self._elapsed_s,
            samples=tuple(self._sample(agent_id) for agent_id in self._order),
        )

    def _sample(self, agent_id: str) -> StateSample:
        i = self._index[agent_id]
        state = self._states[agent_id]
        p = self._pos[i]
        v = self._vel[i]
        return StateSample(
            agent_id=agent_id,
            frame=state.frame,
            pose=Transform(
                translation_m=Vec3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                rotation_quat_xyzw=_IDENTITY_QUAT,
            ),
            linear_velocity_mps=Vec3(x=float(v[0]), y=float(v[1]), z=float(v[2])),
            battery_soc_j=float(self._soc[i]),
            mode=state.mode,
        )

    def import_coupling_state(self, state: CouplingState) -> None:
        if not self._order:  # nothing to overwrite; keep the empty arrays' rank intact
            self._elapsed_s = state.sim_time_s
            return
        incoming = state.by_agent
        pos_rows = [list(map(float, self._pos[i])) for i in range(len(self._order))]
        vel_rows = [list(map(float, self._vel[i])) for i in range(len(self._order))]
        soc_vals = [float(self._soc[i]) for i in range(len(self._order))]
        for agent_id, current in self._states.items():
            sample = incoming.get(agent_id)
            if sample is None:
                continue
            i = self._index[agent_id]
            t = sample.pose.translation_m
            pos_rows[i] = [t.x, t.y, t.z]
            if sample.linear_velocity_mps is not None:
                lv = sample.linear_velocity_mps
                vel_rows[i] = [lv.x, lv.y, lv.z]
            if sample.battery_soc_j is not None:
                soc_vals[i] = sample.battery_soc_j
            if sample.mode is not None:
                current.mode = sample.mode
        self._pos = jnp.asarray(pos_rows, dtype=jnp.float64)
        self._vel = jnp.asarray(vel_rows, dtype=jnp.float64)
        self._soc = jnp.asarray(soc_vals, dtype=jnp.float64)
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
        self._pos = self._pos[jnp.asarray(rows, dtype=jnp.int32)] if rows else _empty(3)
        self._vel = self._vel[jnp.asarray(rows, dtype=jnp.int32)] if rows else _empty(3)
        self._soc = self._soc[jnp.asarray(rows, dtype=jnp.int32)] if rows else _empty()


def _empty(width: int | None = None) -> Any:
    """An empty JAX array of the right rank — the state after every agent retires."""
    return jnp.zeros((0, width), dtype=jnp.float64) if width else jnp.zeros((0,), dtype=jnp.float64)


def _desired_velocity(
    state: _BraxRoverState,
    position_m: tuple[float, float, float],
    max_speed: float,
    dt_s: float,
) -> tuple[float, float, float]:
    """The rover's desired velocity this tick (host-side, mirroring mobility): toward a goto target
    without overshooting in one step, else the commanded velocity, each capped at top speed."""
    if state.goto_point_m is not None:
        to_target = tuple(g - p for g, p in zip(state.goto_point_m, position_m, strict=True))
        distance = _norm(to_target)
        if distance == 0.0:
            return (0.0, 0.0, 0.0)
        speed = min(max_speed, distance / dt_s)
        return tuple(c / distance * speed for c in to_target)  # type: ignore[return-value]
    return _clamp_speed(state.commanded_velocity_mps, max_speed)


def _clamp_speed(v: tuple[float, float, float], max_speed: float) -> tuple[float, float, float]:
    speed = _norm(v)
    if speed <= max_speed or speed == 0.0:
        return v
    return tuple(c / speed * max_speed for c in v)  # type: ignore[return-value]


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(c * c for c in v))


def _params_from_dynamics(dyn: BraxContactDynamics) -> BraxParams:
    return BraxParams(
        mass_kg=dyn.mass_kg,
        max_speed_mps=dyn.max_speed_mps,
        max_traction_n=dyn.max_traction_n,
        idle_power_w=dyn.idle_power_w,
        drive_power_w_per_mps=dyn.drive_power_w_per_mps,
    )


def build_brax_contact_engine(scenario: Scenario, rng: RngStreams) -> BraxContactEngine:
    """Build a :class:`BraxContactEngine` for the scenario's ``brax_contact`` agents.

    Each rover's initial velocity is seeded from the folded JAX key (domain-randomization jitter is
    off by default, so the tier reduces to the deterministic mobility model); the kinematics are
    stacked into JAX arrays for the batched kernel. Non-``brax_contact`` agents are skipped (the
    heterogeneous co-step is RM-P0-SIM-04). All ``brax_contact`` agents must share one
    :class:`BraxParams` set so a single vmapped step covers them; a mismatch raises ``ValueError``.
    """
    key = brax_prng_key(rng)
    states: dict[str, _BraxRoverState] = {}
    positions: list[list[float]] = []
    velocities: list[Any] = []
    socs: list[float] = []
    params: BraxParams | None = None
    for i, spec in enumerate(
        spec for spec in scenario.agents if spec.dynamics.kind == "brax_contact"
    ):
        dyn = spec.dynamics
        assert dyn.kind == "brax_contact"  # narrows the union for type-checkers
        spec_params = _params_from_dynamics(dyn)
        if params is None:
            params = spec_params
        elif spec_params != params:
            raise ValueError(
                "brax_contact agents must share one parameter set for the batched kernel; "
                f"agent {spec.agent_id!r} differs"
            )
        agent_key = jax.random.fold_in(key, i)
        states[spec.agent_id] = _BraxRoverState(
            agent_id=spec.agent_id,
            frame=spec.frame or scenario.frame,
            commanded_velocity_mps=spec.velocity_mps,
            goto_point_m=None,
            battery_floor_j=spec.battery_floor_j,
            mode=spec.mode,
        )
        positions.append(list(spec.initial_position_m))
        velocities.append(
            sample_initial_velocity(agent_key, spec.velocity_mps, dyn.init_speed_jitter_mps)
        )
        socs.append(spec.battery_soc_j)
    if params is None:  # no brax_contact agents — an empty engine (heterogeneous co-step)
        return BraxContactEngine({}, _empty(3), _empty(3), _empty(), _NULL_PARAMS)
    pos = jnp.asarray(positions, dtype=jnp.float64)
    vel = jnp.stack(velocities) if velocities else _empty(3)
    soc = jnp.asarray(socs, dtype=jnp.float64)
    return BraxContactEngine(states, pos, vel, soc, params)


#: A placeholder parameter set for an engine that owns no ``brax_contact`` agents (nothing to step).
_NULL_PARAMS = BraxParams(
    mass_kg=1.0, max_speed_mps=0.0, max_traction_n=0.0, idle_power_w=0.0, drive_power_w_per_mps=0.0
)
