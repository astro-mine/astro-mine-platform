"""The reference kinematic engine — the trivial ``RegimeEngine`` (RM-P0-SIM-02).

A deterministic, dependency-light stand-in that makes the engine-adapter seam real before
the concrete backends land (Basilisk/Orekit, MuJoCo/Brax, Drake, DEM/MPM — RM-P0-SIM-03).
It reproduces *exactly* the dynamics the SIM-01 stepping core shipped inline — constant
velocity + a bounded seeded per-agent jitter, with a linear battery draw clamped at each
agent's floor — now behind the public contract, so swapping in a real engine is a drop-in
and same-seed traces stay byte-for-byte identical.

It is the stepping core's default engine and the always-works local tier (CX-LOCAL):
pure-Python, ``BIT_EXACT`` determinism, no native dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import INERTIAL_J2000, MOON_BODY_FIXED
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
    "KINEMATIC_ENGINE_DESCRIPTOR",
    "KinematicEngine",
    "kinematic_engine_factory",
]

#: Identity orientation — the kinematic engine translates but does not rotate.
_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)
#: Bounded per-step position jitter (m) so the trace is genuinely seed-sensitive without
#: the kinematics diverging.
_JITTER_M = 0.01
#: Linear battery draw (W); a stand-in until power/thermal evolution (RM-P0-SIM-07).
_BATTERY_DRAW_W = 1.0

#: The reference engine's static self-declaration: a surface kinematic tier operating in
#: the lunar anchor's body-fixed and inertial frames, ``BIT_EXACT`` (pure-Python
#: ``random`` over hashlib-seeded streams reproduces byte-for-byte).
KINEMATIC_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.kinematic",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED, INERTIAL_J2000),
    determinism_class=DeterminismClass.BIT_EXACT,
    fidelity=FidelityDescriptor(tier=FidelityTier.KINEMATIC),
)


@dataclass
class _RoverState:
    """Mutable per-agent integration state advanced in place each tick."""

    agent_id: str
    frame: ReferenceFrame
    position_m: list[float]
    velocity_mps: tuple[float, float, float]
    battery_soc_j: float
    battery_floor_j: float
    mode: str | None

    def sample(self) -> StateSample:
        """The current state as a frame-explicit Core coupling sample (velocity included —
        the full dynamical handoff, richer than the projected observation)."""
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


class KinematicEngine:
    """The reference :class:`~astro_mine.sim.engines.adapter.RegimeEngine`.

    Owns the per-agent integration state and advances it deterministically. ``advance``
    takes the step ``dt`` so multi-rate sub-stepping — the coupler's job (RM-P0-SIM-04) —
    is expressible without changing the contract."""

    def __init__(self, states: dict[str, _RoverState], rng: RngStreams) -> None:
        self._states = states
        self._rng = rng
        self._elapsed_s = 0.0

    @property
    def descriptor(self) -> EngineDescriptor:
        return KINEMATIC_ENGINE_DESCRIPTOR

    def apply_actions(self, actions: ActionBatch) -> None:
        """Honor a per-agent velocity setpoint or mode command; ignore everything else.

        A ``VELOCITY`` :class:`~astro_mine.core.messages.model.ActuatorCommand` (3-vector
        setpoint) retargets the agent's constant velocity; a
        :class:`~astro_mine.core.messages.model.ModeCommand` sets its mode. An empty batch
        — what :func:`~astro_mine.sim.runtime.run_episode` passes — is a no-op, so the
        reference trace stays byte-for-byte identical (CX-REPRO)."""
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
                state.velocity_mps = (sx, sy, sz)

    def advance(self, dt_s: float) -> None:
        self._elapsed_s += dt_s
        for aid, state in self._states.items():
            jitter = self._rng.stream(aid)
            for axis in range(3):
                state.position_m[axis] += state.velocity_mps[axis] * dt_s
                state.position_m[axis] += jitter.uniform(-_JITTER_M, _JITTER_M)
            drained = state.battery_soc_j - _BATTERY_DRAW_W * dt_s
            state.battery_soc_j = max(state.battery_floor_j, drained)

    def export_coupling_state(self) -> CouplingState:
        return CouplingState(
            sim_time_s=self._elapsed_s,
            samples=tuple(state.sample() for state in self._states.values()),
        )

    def import_coupling_state(self, state: CouplingState) -> None:
        incoming = state.by_agent
        for aid, current in self._states.items():
            sample = incoming.get(aid)
            if sample is None:
                continue
            current.position_m = [
                sample.pose.translation_m.x,
                sample.pose.translation_m.y,
                sample.pose.translation_m.z,
            ]
            if sample.linear_velocity_mps is not None:
                current.velocity_mps = (
                    sample.linear_velocity_mps.x,
                    sample.linear_velocity_mps.y,
                    sample.linear_velocity_mps.z,
                )
            if sample.battery_soc_j is not None:
                current.battery_soc_j = sample.battery_soc_j
            if sample.mode is not None:
                current.mode = sample.mode
        self._elapsed_s = state.sim_time_s

    def retire(self, agent_ids: Iterable[str]) -> None:
        for aid in agent_ids:
            self._states.pop(aid, None)


def kinematic_engine_factory(scenario: Scenario, rng: RngStreams) -> KinematicEngine:
    """Build a :class:`KinematicEngine` from a scenario's agents.

    Resolves each agent's frame (its own override or the scenario default) and seeds the
    engine with the named RNG streams the stepping core provides — the same seeding that
    keeps same-seed traces byte-identical (CX-REPRO)."""
    states = {
        spec.agent_id: _RoverState(
            agent_id=spec.agent_id,
            frame=spec.frame or scenario.frame,
            position_m=list(spec.initial_position_m),
            velocity_mps=spec.velocity_mps,
            battery_soc_j=spec.battery_soc_j,
            battery_floor_j=spec.battery_floor_j,
            mode=spec.mode,
        )
        for spec in scenario.agents
    }
    return KinematicEngine(states, rng)
