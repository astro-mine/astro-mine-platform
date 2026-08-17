# SPDX-License-Identifier: Apache-2.0
"""The MuJoCo articulated mobility :class:`RegimeEngine` — real wheel-soil contact (RM-P0-SIM-03).

The contact-rich surface-mobility backend RM-P0-SIM-03 names ("mobility/contact (MuJoCo/Brax,
CPU-capable) for rovers"), behind the *same* ``RegimeEngine`` waist as the reduced-order kinematic
:class:`~astro_mine.sim.engines.mobility.MobilityEngine`. Routing it is **configuration** (a
scenario's ``dynamics.kind``), not a Sim code change, and no MuJoCo type leaks past the adapter.

What is actually different from the reduced-order tier: this engine steps a **physical machine** — a
free-jointed chassis and four torque-driven hinge wheels in frictional contact with a compliant
regolith plane (:mod:`astro_mine.sim.engines._rover_mjcf`) — instead of evaluating a closed-form
velocity ramp. Traction is limited by the **friction cone** (``mu = tan(phi)`` from the regolith's
internal friction angle), not by a hand-written ``a = F/m`` cap, so the rover can slip, sink, pitch,
and fail to make headway exactly as a real one does. Commanded velocity enters as a **wheel-speed
setpoint** the wheel motors track, which is what a rover's drive controller really does.

Each agent owns its own MuJoCo model/data pair (rovers do not interact, and a per-agent model keeps
the heterogeneous co-step trivial). The contact solver runs at its own stiff sub-step and the engine
sub-steps it to fill the macro ``dt`` the coupler hands down (sim.md §4: multi-rate by design).

**MuJoCo lives here, never at package import** — :mod:`astro_mine.sim.engines.mujoco` imports this
only inside its factory, so the base wheel and ``builtins.py`` stay MuJoCo-free (the ``[mujoco]``
extra). Determinism is ``TOLERANCE`` (contact iterations are not bit-portable across builds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.sim.engines._rover_mjcf import RoverModelSpec, rover_mjcf
from astro_mine.sim.engines.actuation import actions_by_agent
from astro_mine.sim.engines.adapter import CouplingState, EngineDescriptor
from astro_mine.sim.engines.mujoco._descriptor import MUJOCO_MOBILITY_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from collections.abc import Iterable

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.core.units import ReferenceFrame
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import MujocoMobilityDynamics, Scenario

__all__ = [
    "MujocoMobilityEngine",
    "build_mujoco_mobility_engine",
    "model_spec_from_dynamics",
]


def model_spec_from_dynamics(dyn: MujocoMobilityDynamics) -> RoverModelSpec:
    """The shared rover contact model this agent's dynamics block describes."""
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


@dataclass
class _MujocoRoverState:
    """One rover: its MuJoCo model/data pair plus the Sim-side command + accounting state."""

    agent_id: str
    frame: ReferenceFrame
    model: Any  # mujoco.MjModel
    data: Any  # mujoco.MjData
    spec: RoverModelSpec
    #: The scenario origin the MuJoCo world is anchored at. MuJoCo simulates in its own local frame
    #: (the ground plane at z=0); the agent's scenario position is that local pose plus this origin,
    #: so an agent can start anywhere in the body-fixed frame without moving the terrain.
    origin_m: tuple[float, float, float]
    commanded_velocity_mps: tuple[float, float, float]
    goto_point_m: tuple[float, float, float] | None
    idle_power_w: float
    drive_power_w_per_mps: float
    battery_soc_j: float
    battery_floor_j: float
    mode: str | None

    def position_m(self) -> tuple[float, float, float]:
        """The chassis position in the scenario frame (MuJoCo-local pose + the world origin)."""
        p = self.data.qpos[0:3]
        return (
            float(p[0]) + self.origin_m[0],
            float(p[1]) + self.origin_m[1],
            float(p[2]) + self.origin_m[2],
        )

    def velocity_mps(self) -> tuple[float, float, float]:
        """The chassis linear velocity (the free joint's linear DOFs) — as simulated, with slip."""
        v = self.data.qvel[0:3]
        return (float(v[0]), float(v[1]), float(v[2]))

    def sample(self) -> StateSample:
        """The current state as a frame-explicit Core coupling sample.

        The attitude is *real* here — a contact-simulated chassis pitches and rolls — so unlike the
        point-mass tiers this sample carries a genuine orientation rather than the identity."""
        px, py, pz = self.position_m()
        vx, vy, vz = self.velocity_mps()
        # MuJoCo's free joint stores the quaternion scalar-FIRST (w, x, y, z); Core is scalar-last.
        qw, qx, qy, qz = (float(c) for c in self.data.qpos[3:7])
        return StateSample(
            agent_id=self.agent_id,
            frame=self.frame,
            pose=Transform(
                translation_m=Vec3(x=px, y=py, z=pz),
                rotation_quat_xyzw=Quat(x=qx, y=qy, z=qz, w=qw),
            ),
            linear_velocity_mps=Vec3(x=vx, y=vy, z=vz),
            battery_soc_j=self.battery_soc_j,
            mode=self.mode,
        )


class MujocoMobilityEngine:
    """The MuJoCo articulated wheel-soil-contact :class:`~astro_mine.sim.engines.RegimeEngine`."""

    def __init__(self, states: dict[str, _MujocoRoverState]) -> None:
        self._states = states
        self._elapsed_s = 0.0

    @property
    def descriptor(self) -> EngineDescriptor:
        return MUJOCO_MOBILITY_ENGINE_DESCRIPTOR

    def apply_actions(self, actions: ActionBatch) -> None:
        """Set a velocity setpoint, a goto target, or a mode for owned rovers.

        Identical command surface to the reduced-order mobility engine — the point of the waist is
        that a policy cannot tell which tier it is driving. The commands become **wheel-speed
        setpoints** at the next :meth:`advance`, and whether the rover actually reaches them is the
        contact solver's business, not the command's."""
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
        """Sub-step every rover's contact solver to fill the macro step ``dt_s``.

        Contact is stiff, so the solver runs at its own (much smaller) timestep and we take as many
        sub-steps as fit — the multi-rate discipline sim.md §4 asks for. Each sub-step drives the
        wheel motors toward the resolved speed setpoint and lets the friction cone decide what the
        rover actually does."""
        self._elapsed_s += dt_s
        for state in self._states.values():
            omega = self._wheel_setpoint(state, dt_s)
            state.data.ctrl[:] = omega
            substeps = max(1, round(dt_s / state.spec.timestep_s))
            for _ in range(substeps):
                mujoco.mj_step(state.model, state.data)
            speed = math.dist(state.velocity_mps(), (0.0, 0.0, 0.0))
            draw = (state.idle_power_w + state.drive_power_w_per_mps * speed) * dt_s
            state.battery_soc_j = max(state.battery_floor_j, state.battery_soc_j - draw)

    @staticmethod
    def _wheel_setpoint(state: _MujocoRoverState, dt_s: float) -> float:
        """The wheel angular-velocity setpoint (rad/s) tracking the rover's commanded motion.

        A reduced-order differential-drive controller: the rover is driven along its own +x axis, so
        the commanded *speed* (toward a goto target, else the commanded velocity, capped at top
        speed) maps to a wheel speed ``v / r``. Steering is out of scope for this tier — the point
        under test is wheel-soil contact, not path tracking — so the sign of the setpoint follows
        the
        commanded direction along the body axis."""
        spec = state.spec
        if state.goto_point_m is not None:
            to_target = tuple(
                g - p for g, p in zip(state.goto_point_m, state.position_m(), strict=True)
            )
            distance = math.dist(to_target, (0.0, 0.0, 0.0))
            if distance == 0.0:
                return 0.0
            speed = min(spec.max_speed_mps, distance / dt_s)
            direction = 1.0 if to_target[0] >= 0.0 else -1.0
        else:
            commanded = state.commanded_velocity_mps
            speed = min(spec.max_speed_mps, math.dist(commanded, (0.0, 0.0, 0.0)))
            direction = 1.0 if commanded[0] >= 0.0 else -1.0
        return direction * speed / spec.wheel_radius_m

    def export_coupling_state(self) -> CouplingState:
        return CouplingState(
            sim_time_s=self._elapsed_s,
            samples=tuple(state.sample() for state in self._states.values()),
        )

    def import_coupling_state(self, state: CouplingState) -> None:
        """Overwrite live rovers from a boundary snapshot — writing the incoming pose/velocity
        straight into the MuJoCo free joint (and re-deriving the local pose from the world origin),
        then re-running forward kinematics so the contact state is consistent with the new pose."""
        incoming = state.by_agent
        for agent_id, current in self._states.items():
            sample = incoming.get(agent_id)
            if sample is None:
                continue
            t = sample.pose.translation_m
            current.data.qpos[0:3] = [
                t.x - current.origin_m[0],
                t.y - current.origin_m[1],
                t.z - current.origin_m[2],
            ]
            q = sample.pose.rotation_quat_xyzw
            current.data.qpos[3:7] = [q.w, q.x, q.y, q.z]  # Core scalar-last -> MuJoCo scalar-first
            velocity = sample.linear_velocity_mps
            if velocity is not None:
                current.data.qvel[0:3] = [velocity.x, velocity.y, velocity.z]
            mujoco.mj_forward(current.model, current.data)
            if sample.battery_soc_j is not None:
                current.battery_soc_j = sample.battery_soc_j
            if sample.mode is not None:
                current.mode = sample.mode
        self._elapsed_s = state.sim_time_s

    def retire(self, agent_ids: Iterable[str]) -> None:
        for agent_id in agent_ids:
            self._states.pop(agent_id, None)


def build_mujoco_mobility_engine(scenario: Scenario, rng: RngStreams) -> MujocoMobilityEngine:
    """Build a :class:`MujocoMobilityEngine` for the scenario's ``mujoco_mobility`` agents.

    Each rover gets its own compiled MuJoCo model, anchored so the scenario's initial position is
    the world origin of its local contact world (the ground plane stays at the rover's feet wherever
    it starts). Non-``mujoco_mobility`` agents are skipped (the heterogeneous co-step is
    RM-P0-SIM-04).
    The contact solve is deterministic for fixed inputs, so the seeded ``rng`` is unused."""
    states: dict[str, _MujocoRoverState] = {}
    for spec_agent in scenario.agents:
        dyn = spec_agent.dynamics
        if dyn.kind != "mujoco_mobility":
            continue
        spec = model_spec_from_dynamics(dyn)
        model = mujoco.MjModel.from_xml_string(rover_mjcf(spec))
        data = mujoco.MjData(model)
        # Seed the chassis' initial linear velocity from the scenario, then settle the model so the
        # wheels rest in contact before the first step (an un-settled model would drop on tick 1).
        data.qvel[0:3] = np.asarray(spec_agent.velocity_mps, dtype=float)
        mujoco.mj_forward(model, data)
        states[spec_agent.agent_id] = _MujocoRoverState(
            agent_id=spec_agent.agent_id,
            frame=spec_agent.frame or scenario.frame,
            model=model,
            data=data,
            spec=spec,
            origin_m=spec_agent.initial_position_m,
            commanded_velocity_mps=spec_agent.velocity_mps,
            goto_point_m=None,
            idle_power_w=dyn.idle_power_w,
            drive_power_w_per_mps=dyn.drive_power_w_per_mps,
            battery_soc_j=spec_agent.battery_soc_j,
            battery_floor_j=spec_agent.battery_floor_j,
            mode=spec_agent.mode,
        )
    return MujocoMobilityEngine(states)
