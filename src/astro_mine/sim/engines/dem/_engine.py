"""The DEM granular :class:`RegimeEngine` — ground-truth excavation behind the waist (RM-P1-SIM-06).

Wraps the numpy soft-sphere kernel (:mod:`._solver`) as a Sim regime engine: an ``excavate``
task drives a blade through a settled particle bed, and the engine exposes the *aggregate*
result across the Core waist (the excavator's pose + battery, drawn by the draft work) while
keeping the per-particle DEM state internal. The rich ground truth a
[Surrogate](surrogate.md) learns — particle kinematics, the draft (tool-reaction) force, and
the excavated mass — is offered through engine-specific accessors (outside the ``RegimeEngine``
Protocol, exactly as ``GranularEngine`` exposes ``excavated_mass_kg``); the Core observation
schema carries no particle-array channel, and none is needed to keep the waist thin.

This module imports numpy; it is loaded only by the ``[dem]``-gated factory
(:func:`build_dem_engine`), never at package import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.sim.engines.actuation import actions_by_agent
from astro_mine.sim.engines.adapter import CouplingState, EngineDescriptor
from astro_mine.sim.engines.dem._descriptor import DEM_GRANULAR_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.dem._solver import (
    DemBed,
    DemParams,
    FloatArray,
    build_params,
    make_bed,
    substep,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.core.units import ReferenceFrame
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import DemGranularDynamics, Scenario

__all__ = ["DemGranularEngine", "build_dem_engine"]

_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)


@dataclass(slots=True)
class _DemDiggerState:
    """Per-excavator DEM state: the particle bed, the dig command, and the coupling scalars."""

    agent_id: str
    frame: ReferenceFrame
    position_m: tuple[float, float, float]
    params: DemParams
    bed: DemBed
    battery_soc_j: float
    battery_floor_j: float
    mode: str | None
    digging: bool = False
    target_mass_kg: float = field(default=math.inf)

    def sample(self) -> StateSample:
        """The excavator's stationary base pose + battery as a Core coupling sample."""
        return StateSample(
            agent_id=self.agent_id,
            frame=self.frame,
            pose=Transform(
                translation_m=Vec3(
                    x=self.position_m[0], y=self.position_m[1], z=self.position_m[2]
                ),
                rotation_quat_xyzw=_IDENTITY_QUAT,
            ),
            battery_soc_j=self.battery_soc_j,
            mode=self.mode,
        )


class DemGranularEngine:
    """The high-fidelity DEM excavation :class:`~astro_mine.sim.engines.RegimeEngine`.

    Each owned excavator carries a settled particle bed; ``advance`` sub-steps the DEM at its
    stable internal timestep, sweeping the blade while digging and drawing battery for the draft
    work. Particle state stays internal — read it (and the draft force / excavated mass) through
    the accessors for surrogate datagen (RM-P1-SURR-02/03) and validation.
    """

    def __init__(self, states: dict[str, _DemDiggerState]) -> None:
        self._states = states
        self._elapsed_s = 0.0

    @property
    def descriptor(self) -> EngineDescriptor:
        return DEM_GRANULAR_ENGINE_DESCRIPTOR

    # -- engine-specific accessors (outside the RegimeEngine Protocol) --------------------

    def bed(self, agent_id: str) -> DemBed:
        """The live particle bed of ``agent_id`` — the ground-truth DEM state a surrogate learns."""
        return self._states[agent_id].bed

    def particles(self, agent_id: str) -> tuple[FloatArray, FloatArray]:
        """``(positions, velocities)`` of ``agent_id``'s particles, each ``(N, 2)`` (x, z)."""
        state = self._states[agent_id].bed
        return state.pos, state.vel

    def tool_reaction_force_n(self, agent_id: str) -> float:
        """The last-substep horizontal draft (tool-reaction) force on the blade (N)."""
        return self._states[agent_id].bed.tool_reaction_n

    def excavated_mass_kg(self, agent_id: str) -> float:
        """Excavated (displaced) mass so far — particles moved off their settled rest spot (kg)."""
        state = self._states[agent_id]
        return state.bed.displaced_mass_kg(state.params)

    def excavated_volume_m3(self, agent_id: str) -> float:
        """Excavated volume so far — displaced mass over bulk density (m³)."""
        state = self._states[agent_id]
        return state.bed.displaced_mass_kg(state.params) / state.params.regolith_density_kg_m3

    def floor_reaction_n(self, agent_id: str) -> float:
        """The last-substep total upward floor reaction (N) — ≈ bed weight at rest."""
        return self._states[agent_id].bed.floor_reaction_n

    # -- RegimeEngine contract ------------------------------------------------------------

    def apply_actions(self, actions: ActionBatch) -> None:
        """Start/continue a dig from an ``excavate`` task (target volume → target displaced mass),
        or set the mode, for owned excavators."""
        for agent_id, action in actions_by_agent(actions).items():
            state = self._states.get(agent_id)
            if state is None:
                continue
            if action.kind is ActionKind.MODE and action.mode is not None:
                state.mode = action.mode.mode
            elif (
                action.kind is ActionKind.TASK and action.task is not None and action.task.excavate
            ):
                target = action.task.excavate.target_volume_m3
                state.digging = True
                state.target_mass_kg = (
                    math.inf if target is None else target * state.params.regolith_density_kg_m3
                )

    def advance(self, dt_s: float) -> None:
        """Sub-step every owned bed forward ``dt_s`` at the stable internal timestep.

        While digging, the blade advances and the battery is drawn for the draft work
        (∫ draft·speed dt); digging stops when the excavated mass reaches the target or the
        blade sweeps past the bed.
        """
        for state in self._states.values():
            params = state.params
            n_sub = max(1, round(dt_s / params.dt_internal_s))
            for _ in range(n_sub):
                active = state.digging and _blade_in_bed(state)
                substep(state.bed, params, params.dt_internal_s, tool_active=active)
                if active:
                    work = state.bed.tool_reaction_n * params.tool_speed_mps * params.dt_internal_s
                    state.battery_soc_j = max(state.battery_floor_j, state.battery_soc_j - work)
            if state.digging and (
                state.bed.displaced_mass_kg(params) >= state.target_mass_kg
                or not _blade_in_bed(state)
            ):
                state.digging = False
        self._elapsed_s += dt_s

    def export_coupling_state(self) -> CouplingState:
        return CouplingState(
            sim_time_s=self._elapsed_s,
            samples=tuple(state.sample() for state in self._states.values()),
            # Same channel the reduced-order granular engine publishes (#64), so a scenario that
            # escalates to the DEM tier feeds the value chain identically — the high-fidelity
            # displaced mass is the same physical quantity, measured better.
            excavated_kg={agent_id: self.excavated_mass_kg(agent_id) for agent_id in self._states},
        )

    def import_coupling_state(self, state: CouplingState) -> None:
        incoming = state.by_agent
        for agent_id, current in self._states.items():
            sample = incoming.get(agent_id)
            if sample is None:
                continue
            t = sample.pose.translation_m
            current.position_m = (t.x, t.y, t.z)
            if sample.battery_soc_j is not None:
                current.battery_soc_j = sample.battery_soc_j
            if sample.mode is not None:
                current.mode = sample.mode
        self._elapsed_s = state.sim_time_s

    def retire(self, agent_ids: Iterable[str]) -> None:
        for agent_id in agent_ids:
            self._states.pop(agent_id, None)


def _blade_in_bed(state: _DemDiggerState) -> bool:
    """Whether the blade is still within the bed (has not swept off the far wall)."""
    return state.bed.tool_x_m < state.params.bed_width_m - state.params.particle_radius_m


def _params_from_dynamics(dyn: DemGranularDynamics) -> DemParams:
    return build_params(
        n_particles=dyn.n_particles,
        particle_radius_m=dyn.particle_radius_m,
        regolith_density_kg_m3=dyn.regolith_density_kg_m3,
        contact_stiffness_n_m=dyn.contact_stiffness_n_m,
        restitution=dyn.restitution,
        friction_coeff=dyn.friction_coeff,
        gravity_m_s2=dyn.gravity_m_s2,
        bed_width_m=dyn.bed_width_m,
        tool_x0_m=dyn.tool_x0_m,
        tool_height_m=dyn.tool_height_m,
        tool_speed_mps=dyn.tool_speed_mps,
    )


def build_dem_engine(scenario: Scenario, rng: RngStreams) -> DemGranularEngine:
    """Build a :class:`DemGranularEngine` for the scenario's ``dem_granular`` agents.

    Each bed is seeded from the agent's own named RNG stream (so the packing is reproducible
    and independent), then **settled** under gravity for ``settle_substeps`` before its rest
    state is re-baselined — so a subsequent dig's excavated mass measures excavation, not the
    settling transient. Non-``dem_granular`` agents are skipped.
    """
    states: dict[str, _DemDiggerState] = {}
    for spec in scenario.agents:
        dyn = spec.dynamics
        if dyn.kind != "dem_granular":
            continue
        params = _params_from_dynamics(dyn)
        seed = rng.stream(spec.agent_id).getrandbits(63)
        bed = make_bed(params, seed)
        for _ in range(dyn.settle_substeps):
            substep(bed, params, params.dt_internal_s, tool_active=False)
        bed.pos0 = bed.pos.copy()  # re-baseline: excavation is measured from the settled state
        states[spec.agent_id] = _DemDiggerState(
            agent_id=spec.agent_id,
            frame=spec.frame or scenario.frame,
            position_m=spec.initial_position_m,
            params=params,
            bed=bed,
            battery_soc_j=spec.battery_soc_j,
            battery_floor_j=spec.battery_floor_j,
            mode=spec.mode,
        )
    return DemGranularEngine(states)
