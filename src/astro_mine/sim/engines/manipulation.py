# SPDX-License-Identifier: Apache-2.0
"""Manipulation engine — reduced-order articulated excavator linkage (RM-P0-SIM-03).

The Phase-0 manipulation tier: a deterministic articulated-chain model for an excavator
arm. Joint setpoints (a ``POSITION`` actuator command) are tracked under a per-joint rate
limit, and **forward kinematics** over the ordered revolute/prismatic chain maps the joint
configuration to the **end-effector (tool-tip) pose** — the boundary quantity an excavation
or contact engine consumes. Pure-Python, the always-works local tier (CX-LOCAL), behind the
same :class:`~astro_mine.sim.engines.RegimeEngine` waist.

It is the ``ARTICULATED`` rung of the surface fidelity ladder; the contact-rich Drake
multibody backend (hydroelastic grasp/dig contact) is the higher tier behind this same
contract later. Determinism is ``TOLERANCE`` (libm ``sin``/``cos`` in the kinematics). The
reduced-order pose reports the tool-tip *position*; tool-tip orientation is left identity at
this tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, JointType, Regime
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines._vecmath import Vec, add, axis_angle_rotate, normalize, scale
from astro_mine.sim.engines.actuation import actions_by_agent
from astro_mine.sim.engines.adapter import (
    CouplingState,
    EngineDescriptor,
    FidelityDescriptor,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.core.units import ReferenceFrame
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import JointSpec, Scenario

__all__ = [
    "MANIPULATION_ENGINE_DESCRIPTOR",
    "ManipulationEngine",
    "manipulation_engine_factory",
]

#: Identity orientation — the reduced-order tip pose reports position only.
_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)

#: The manipulation engine's static self-declaration: a surface articulated tier in the
#: lunar body-fixed frame, ``TOLERANCE`` determinism (trig in the kinematics).
MANIPULATION_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.manipulation",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED,),
    determinism_class=DeterminismClass.TOLERANCE,
    fidelity=FidelityDescriptor(tier=FidelityTier.ARTICULATED),
)


def _forward_kinematics(base_offset_m: Vec, joints: Sequence[JointSpec], q: Sequence[float]) -> Vec:
    """The tool-tip position for joint configuration ``q``.

    Walks the chain from the base: a revolute joint rotates the running link frame about its
    (frame-local) axis by ``q``; a prismatic joint slides along its axis by ``q``; each link
    then extends by ``link_length_m`` along the running local +x. Non-revolute/prismatic
    joints are treated as fixed."""
    position = base_offset_m
    basis_x: Vec = (1.0, 0.0, 0.0)
    basis_y: Vec = (0.0, 1.0, 0.0)
    basis_z: Vec = (0.0, 0.0, 1.0)
    for joint, value in zip(joints, q, strict=True):
        ax = joint.axis
        axis_world = add(add(scale(basis_x, ax[0]), scale(basis_y, ax[1])), scale(basis_z, ax[2]))
        if joint.joint_type is JointType.REVOLUTE:
            basis_x = axis_angle_rotate(basis_x, axis_world, value)
            basis_y = axis_angle_rotate(basis_y, axis_world, value)
            basis_z = axis_angle_rotate(basis_z, axis_world, value)
        elif joint.joint_type is JointType.PRISMATIC:
            position = add(position, scale(normalize(axis_world), value))
        position = add(position, scale(basis_x, joint.link_length_m))
    return position


@dataclass
class _ArmState:
    """Mutable per-arm joint state advanced in place each tick."""

    agent_id: str
    frame: ReferenceFrame
    joints: tuple[JointSpec, ...]
    base_offset_m: Vec
    q: list[float]
    setpoint: list[float]
    actuation_power_w: float
    battery_soc_j: float
    battery_floor_j: float
    mode: str | None
    tip_position_m: Vec = field(default=(0.0, 0.0, 0.0))

    def sample(self) -> StateSample:
        """The current state as a frame-explicit Core coupling sample (tool-tip pose)."""
        return StateSample(
            agent_id=self.agent_id,
            frame=self.frame,
            pose=Transform(
                translation_m=Vec3(
                    x=self.tip_position_m[0], y=self.tip_position_m[1], z=self.tip_position_m[2]
                ),
                rotation_quat_xyzw=_IDENTITY_QUAT,
            ),
            battery_soc_j=self.battery_soc_j,
            mode=self.mode,
        )


class ManipulationEngine:
    """The reduced-order articulated :class:`~astro_mine.sim.engines.RegimeEngine`.

    Owns the per-arm joint configuration, ramps it toward a setpoint under each joint's rate
    limit, and reports the tool-tip pose via forward kinematics."""

    def __init__(self, states: dict[str, _ArmState]) -> None:
        self._states = states
        self._elapsed_s = 0.0
        for state in self._states.values():  # seed the reported tip from the initial config
            state.tip_position_m = _forward_kinematics(state.base_offset_m, state.joints, state.q)

    @property
    def descriptor(self) -> EngineDescriptor:
        return MANIPULATION_ENGINE_DESCRIPTOR

    def apply_actions(self, actions: ActionBatch) -> None:
        """Set the per-joint position setpoint (a ``POSITION`` actuator command whose setpoint
        is one target per joint) or the mode for owned arms."""
        for agent_id, action in actions_by_agent(actions).items():
            state = self._states.get(agent_id)
            if state is None:
                continue
            if action.kind is ActionKind.MODE and action.mode is not None:
                state.mode = action.mode.mode
            elif (
                action.kind is ActionKind.ACTUATOR
                and action.actuator is not None
                and action.actuator.control_mode is ControlMode.POSITION
                and len(action.actuator.setpoint) == len(state.joints)
            ):
                state.setpoint = list(action.actuator.setpoint)

    def advance(self, dt_s: float) -> None:
        self._elapsed_s += dt_s
        for state in self._states.values():
            moving = self._track_setpoint(state, dt_s)
            state.tip_position_m = _forward_kinematics(state.base_offset_m, state.joints, state.q)
            if moving:
                drained = state.battery_soc_j - state.actuation_power_w * dt_s
                state.battery_soc_j = max(state.battery_floor_j, drained)

    @staticmethod
    def _track_setpoint(state: _ArmState, dt_s: float) -> bool:
        """Step each joint toward its setpoint under its rate limit and joint limits; report
        whether anything moved."""
        moving = False
        for i, joint in enumerate(state.joints):
            current = state.q[i]
            delta = state.setpoint[i] - current
            max_step = joint.rate_limit * dt_s
            if delta > max_step:
                delta = max_step
            elif delta < -max_step:
                delta = -max_step
            new_value = current + delta
            if joint.lower is not None:
                new_value = max(joint.lower, new_value)
            if joint.upper is not None:
                new_value = min(joint.upper, new_value)
            if new_value != current:
                moving = True
            state.q[i] = new_value
        return moving

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
            t = sample.pose.translation_m  # the boundary quantity is the tool-tip pose
            current.tip_position_m = (t.x, t.y, t.z)
            if sample.battery_soc_j is not None:
                current.battery_soc_j = sample.battery_soc_j
            if sample.mode is not None:
                current.mode = sample.mode
        self._elapsed_s = state.sim_time_s

    def retire(self, agent_ids: Iterable[str]) -> None:
        for agent_id in agent_ids:
            self._states.pop(agent_id, None)


def manipulation_engine_factory(scenario: Scenario, rng: RngStreams) -> ManipulationEngine:
    """Build a :class:`ManipulationEngine` for the scenario's ``manipulation`` agents.

    Non-manipulation agents are skipped (the heterogeneous co-step is RM-P0-SIM-04); the
    reduced-order kinematics is deterministic, so ``rng`` is unused."""
    states: dict[str, _ArmState] = {}
    for spec in scenario.agents:
        dyn = spec.dynamics
        if dyn.kind != "manipulation":
            continue
        initial = [joint.initial for joint in dyn.joints]
        states[spec.agent_id] = _ArmState(
            agent_id=spec.agent_id,
            frame=spec.frame or scenario.frame,
            joints=dyn.joints,
            base_offset_m=dyn.base_offset_m,
            q=list(initial),
            setpoint=list(initial),
            actuation_power_w=dyn.actuation_power_w,
            battery_soc_j=spec.battery_soc_j,
            battery_floor_j=spec.battery_floor_j,
            mode=spec.mode,
        )
    return ManipulationEngine(states)
