"""Orbital engine — reduced-order two-body propagation for the relay (RM-P0-SIM-03).

The Phase-0 orbital tier: a deterministic, dependency-light **two-body** propagator
(RK4, sub-stepped) for the relay orbiter, in the lunar inertial frame
(:data:`~astro_mine.core.units.INERTIAL_J2000`). It integrates ``r̈ = -μ r / ‖r‖³`` from
the agent's initial position/velocity — *"the relay propagates orbitally"* — and is the
always-works CPU tier (CX-LOCAL) behind the same :class:`~astro_mine.sim.engines.RegimeEngine`
waist as every other engine.

It is deliberately the ``MASSMODEL`` rung of the orbital fidelity ladder: the
flight-grade Basilisk/Orekit backends and the oracle-validated regression against
STK/GMAT (RM-P0-SIM-10) are the higher tiers that plug in behind this same contract later.
Determinism is ``TOLERANCE`` (RK4 over libm ``sqrt`` — bit-identical in-process, bounded
across builds; sim.md §11). No maneuvers: a Δv/targeting capability is gated out of the
open commons (``operational_targeting``), so ``apply_actions`` honors only a mode command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import INERTIAL_J2000
from astro_mine.sim.engines._vecmath import Vec, add, norm, scale
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
    "ORBITAL_ENGINE_DESCRIPTOR",
    "OrbitalEngine",
    "orbital_engine_factory",
]

#: Identity orientation — the point-mass orbital state carries no attitude.
_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)

#: The orbital engine's static self-declaration: a proximity-orbit mass-model tier in the
#: lunar inertial frame, ``TOLERANCE`` determinism (RK4 over ``sqrt``).
ORBITAL_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.orbital",
    version="0.1.0",
    regimes=(Regime.PROXIMITY_ORBIT,),
    frames=(INERTIAL_J2000,),
    determinism_class=DeterminismClass.TOLERANCE,
    fidelity=FidelityDescriptor(tier=FidelityTier.MASSMODEL),
)


def _two_body_accel(r: Vec, mu: float) -> Vec:
    """Central-gravity acceleration ``-μ r / ‖r‖³`` at position ``r``."""
    distance = norm(r)
    return scale(r, -mu / (distance * distance * distance))


def _rk4_substep(r: Vec, v: Vec, mu: float, h: float) -> tuple[Vec, Vec]:
    """One classical RK4 step of the two-body system over ``h`` seconds."""
    k1v = _two_body_accel(r, mu)
    k2r = add(v, scale(k1v, h / 2.0))
    k2v = _two_body_accel(add(r, scale(v, h / 2.0)), mu)
    k3r = add(v, scale(k2v, h / 2.0))
    k3v = _two_body_accel(add(r, scale(k2r, h / 2.0)), mu)
    k4r = add(v, scale(k3v, h))
    k4v = _two_body_accel(add(r, scale(k3r, h)), mu)
    r_next = add(r, scale(add(add(v, scale(add(k2r, k3r), 2.0)), k4r), h / 6.0))
    v_next = add(v, scale(add(add(k1v, scale(add(k2v, k3v), 2.0)), k4v), h / 6.0))
    return r_next, v_next


@dataclass
class _OrbitalState:
    """Mutable per-orbiter integration state advanced in place each tick."""

    agent_id: str
    frame: ReferenceFrame
    position_m: Vec
    velocity_mps: Vec
    mu_m3_s2: float
    substeps: int
    station_keeping_power_w: float
    battery_soc_j: float
    battery_floor_j: float
    mode: str | None

    def sample(self) -> StateSample:
        """The current state as a frame-explicit Core coupling sample (position + velocity)."""
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


class OrbitalEngine:
    """The reduced-order two-body :class:`~astro_mine.sim.engines.RegimeEngine`.

    Owns the per-orbiter inertial state and propagates it with sub-stepped RK4; the only
    actuation is a mode command (no maneuvers in the open commons)."""

    def __init__(self, states: dict[str, _OrbitalState]) -> None:
        self._states = states
        self._elapsed_s = 0.0

    @property
    def descriptor(self) -> EngineDescriptor:
        return ORBITAL_ENGINE_DESCRIPTOR

    def apply_actions(self, actions: ActionBatch) -> None:
        """Honor a mode command for an owned orbiter; ignore everything else (no maneuvers)."""
        for agent_id, action in actions_by_agent(actions).items():
            state = self._states.get(agent_id)
            if state is None:
                continue
            if action.kind is ActionKind.MODE and action.mode is not None:
                state.mode = action.mode.mode

    def advance(self, dt_s: float) -> None:
        self._elapsed_s += dt_s
        for state in self._states.values():
            h = dt_s / state.substeps
            r, v = state.position_m, state.velocity_mps
            for _ in range(state.substeps):
                r, v = _rk4_substep(r, v, state.mu_m3_s2, h)
            state.position_m, state.velocity_mps = r, v
            drained = state.battery_soc_j - state.station_keeping_power_w * dt_s
            state.battery_soc_j = max(state.battery_floor_j, drained)

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


def orbital_engine_factory(scenario: Scenario, rng: RngStreams) -> OrbitalEngine:
    """Build an :class:`OrbitalEngine` for the scenario's ``orbital`` agents.

    Agents whose ``dynamics`` is not orbital are skipped, so the engine owns only its
    regime's assets (the heterogeneous co-step across engines is RM-P0-SIM-04). The
    reduced-order propagator is deterministic, so the seeded ``rng`` is unused here."""
    states: dict[str, _OrbitalState] = {}
    for spec in scenario.agents:
        dyn = spec.dynamics
        if dyn.kind != "orbital":
            continue
        states[spec.agent_id] = _OrbitalState(
            agent_id=spec.agent_id,
            frame=spec.frame or scenario.frame,
            position_m=spec.initial_position_m,
            velocity_mps=spec.velocity_mps,
            mu_m3_s2=dyn.mu_m3_s2,
            substeps=dyn.substeps,
            station_keeping_power_w=dyn.station_keeping_power_w,
            battery_soc_j=spec.battery_soc_j,
            battery_floor_j=spec.battery_floor_j,
            mode=spec.mode,
        )
    return OrbitalEngine(states)
