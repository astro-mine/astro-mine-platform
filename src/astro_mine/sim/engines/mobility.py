"""Mobility engine — mid-fidelity wheeled-rover surface mobility (RM-P0-SIM-03).

The Phase-0 surface-mobility tier: a deterministic rover model that tracks a commanded
velocity or a goto target, with acceleration capped by a **terramechanics drawbar-pull
limit** (``max_traction_n`` / ``mass_kg``) and speed capped by ``max_speed_mps`` — richer
than the constant-velocity reference engine, so *"rovers traverse the terrain at
mid-fidelity"*. Battery draw scales with speed. Pure-Python, the always-works local tier
(CX-LOCAL), behind the same :class:`~astro_mine.sim.engines.RegimeEngine` waist.

It is the ``KINEMATIC`` rung of the surface fidelity ladder; the contact-rich MuJoCo/Brax
backends and full Worlds-terrain/regolith coupling (slope-limited traction, sinkage) are
higher tiers behind this same contract later — here the terrain is parameterized on the
agent's ``dynamics`` (the Worlds data plumbing is deferred per the issue scope). Determinism
is ``TOLERANCE`` (libm ``sqrt`` for speed/heading).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines._vecmath import Vec, add, norm, normalize, scale, sub
from astro_mine.sim.engines.actuation import actions_by_agent
from astro_mine.sim.engines.adapter import (
    CouplingState,
    EngineDescriptor,
    FidelityDescriptor,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.core.units import ReferenceFrame
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = [
    "MOBILITY_ENGINE_DESCRIPTOR",
    "MobilityEngine",
    "mobility_engine_factory",
]

#: Identity orientation — the point-mass rover model carries no attitude (heading is
#: implicit in the velocity vector).
_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)

#: The mobility engine's static self-declaration: a surface kinematic tier in the lunar
#: body-fixed frame, ``TOLERANCE`` determinism.
MOBILITY_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.mobility",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED,),
    determinism_class=DeterminismClass.TOLERANCE,
    fidelity=FidelityDescriptor(tier=FidelityTier.KINEMATIC),
)


def _clamp_speed(v: Vec, max_speed: float) -> Vec:
    """``v`` with its magnitude capped at ``max_speed``."""
    speed = norm(v)
    if speed <= max_speed:
        return v
    return scale(normalize(v), max_speed)


@dataclass
class _RoverState:
    """Mutable per-rover integration state advanced in place each tick."""

    agent_id: str
    frame: ReferenceFrame
    position_m: Vec
    velocity_mps: Vec
    commanded_velocity_mps: Vec
    goto_point_m: Vec | None
    mass_kg: float
    max_speed_mps: float
    max_traction_n: float
    idle_power_w: float
    drive_power_w_per_mps: float
    battery_soc_j: float
    battery_floor_j: float
    mode: str | None

    def sample(self) -> StateSample:
        """The current state as a frame-explicit Core coupling sample (pose + velocity)."""
        return StateSample(
            agent_id=self.agent_id,
            frame=self.frame,
            pose=Transform(
                translation_m=Vec3(
                    x=self.position_m[0], y=self.position_m[1], z=self.position_m[2]
                ),
                rotation_quat_xyzw=_IDENTITY_QUAT,
            ),
            linear_velocity_mps=Vec3(
                x=self.velocity_mps[0], y=self.velocity_mps[1], z=self.velocity_mps[2]
            ),
            battery_soc_j=self.battery_soc_j,
            mode=self.mode,
        )


class MobilityEngine:
    """The mid-fidelity wheeled-rover :class:`~astro_mine.sim.engines.RegimeEngine`.

    Tracks a commanded velocity or a goto target under a traction-limited acceleration and a
    top-speed cap; battery draw scales with speed."""

    def __init__(self, states: dict[str, _RoverState]) -> None:
        self._states = states
        self._elapsed_s = 0.0

    @property
    def descriptor(self) -> EngineDescriptor:
        return MOBILITY_ENGINE_DESCRIPTOR

    def apply_actions(self, actions: ActionBatch) -> None:
        """Set a velocity setpoint, a goto target, or a mode for owned rovers.

        A ``VELOCITY`` actuator command sets the commanded velocity (and clears any goto); a
        ``goto`` task sets the target point; a mode command sets the mode. The subsequent
        :meth:`advance` ramps toward the command under the traction cap."""
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
        self._elapsed_s += dt_s
        for state in self._states.values():
            desired = self._desired_velocity(state, dt_s)
            state.velocity_mps = _clamp_speed(
                self._ramp(state.velocity_mps, desired, state, dt_s), state.max_speed_mps
            )
            state.position_m = add(state.position_m, scale(state.velocity_mps, dt_s))
            speed = norm(state.velocity_mps)
            draw = (state.idle_power_w + state.drive_power_w_per_mps * speed) * dt_s
            state.battery_soc_j = max(state.battery_floor_j, state.battery_soc_j - draw)

    @staticmethod
    def _desired_velocity(state: _RoverState, dt_s: float) -> Vec:
        """The velocity the rover wants this tick: toward a goto target (without
        overshooting in one step), else the commanded velocity, each capped at top speed."""
        if state.goto_point_m is not None:
            to_target = sub(state.goto_point_m, state.position_m)
            distance = norm(to_target)
            if distance == 0.0:
                return (0.0, 0.0, 0.0)
            speed = min(state.max_speed_mps, distance / dt_s)
            return scale(normalize(to_target), speed)
        return _clamp_speed(state.commanded_velocity_mps, state.max_speed_mps)

    @staticmethod
    def _ramp(velocity: Vec, desired: Vec, state: _RoverState, dt_s: float) -> Vec:
        """Move ``velocity`` toward ``desired``, bounded by the traction-limited
        acceleration ``max_traction_n / mass_kg`` over ``dt_s``."""
        delta = sub(desired, velocity)
        max_delta = (state.max_traction_n / state.mass_kg) * dt_s
        if norm(delta) > max_delta:
            delta = scale(normalize(delta), max_delta)
        return add(velocity, delta)

    def export_coupling_state(self) -> CouplingState:
        return CouplingState(
            sim_time_s=self._elapsed_s,
            samples=tuple(state.sample() for state in self._states.values()),
        )

    def import_coupling_state(self, state: CouplingState) -> None:
        incoming = state.by_agent
        for agent_id, current in self._states.items():
            sample = incoming.get(agent_id)
            if sample is None:
                continue
            t = sample.pose.translation_m
            current.position_m = (t.x, t.y, t.z)
            if sample.linear_velocity_mps is not None:
                lv = sample.linear_velocity_mps
                current.velocity_mps = (lv.x, lv.y, lv.z)
            if sample.battery_soc_j is not None:
                current.battery_soc_j = sample.battery_soc_j
            if sample.mode is not None:
                current.mode = sample.mode
        self._elapsed_s = state.sim_time_s

    def retire(self, agent_ids: Iterable[str]) -> None:
        for agent_id in agent_ids:
            self._states.pop(agent_id, None)


def mobility_engine_factory(scenario: Scenario, rng: RngStreams) -> MobilityEngine:
    """Build a :class:`MobilityEngine` for the scenario's ``mobility`` agents.

    Non-mobility agents are skipped (the heterogeneous co-step is RM-P0-SIM-04). The
    commanded velocity is seeded from the agent's initial ``velocity_mps`` so a rover coasts
    until commanded; the reduced-order model is deterministic, so ``rng`` is unused."""
    states: dict[str, _RoverState] = {}
    for spec in scenario.agents:
        dyn = spec.dynamics
        if dyn.kind != "mobility":
            continue
        states[spec.agent_id] = _RoverState(
            agent_id=spec.agent_id,
            frame=spec.frame or scenario.frame,
            position_m=spec.initial_position_m,
            velocity_mps=spec.velocity_mps,
            commanded_velocity_mps=spec.velocity_mps,
            goto_point_m=None,
            mass_kg=dyn.mass_kg,
            max_speed_mps=dyn.max_speed_mps,
            max_traction_n=dyn.max_traction_n,
            idle_power_w=dyn.idle_power_w,
            drive_power_w_per_mps=dyn.drive_power_w_per_mps,
            battery_soc_j=spec.battery_soc_j,
            battery_floor_j=spec.battery_floor_j,
            mode=spec.mode,
        )
    return MobilityEngine(states)
