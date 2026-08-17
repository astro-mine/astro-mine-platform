# SPDX-License-Identifier: Apache-2.0
"""Granular engine — reduced-order excavation dynamics (RM-P0-SIM-03).

The Phase-0 granular/excavation tier: a deterministic, reduced-order model of an excavator
removing regolith. An ``excavate`` task sets a target volume; each tick removes volume at a
capped rate, accrues the excavated **mass** (``regolith_density_kg_m3`` * volume) and draws
battery for the **work** (``specific_energy_j_per_m3`` * volume — a reduced
resistive-force/energy law). Pure arithmetic, the always-works local tier (CX-LOCAL), behind
the same :class:`~astro_mine.sim.engines.RegimeEngine` waist.

It is the ``MASSMODEL`` rung of the excavation ladder, with a deliberate seam for the
bounded-error **surrogate** tier (the ``GRANULAR_EXCAVATION``
:class:`~astro_mine.core.sadf.enums.SurrogatePhysicsDomain`) and the ground-truth DEM/MPM
backend that plug in behind this same contract later (sim.md §4, §11). Determinism is
``BIT_EXACT`` — the model is exact IEEE arithmetic (no transcendentals), so same-seed runs
reproduce byte-for-byte across builds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines._vecmath import Vec
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
    "GRANULAR_ENGINE_DESCRIPTOR",
    "GranularEngine",
    "granular_engine_factory",
]

#: Identity orientation — the stationary excavator carries no attitude at this tier.
_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)

#: The granular engine's static self-declaration: a surface mass-model excavation tier in
#: the lunar body-fixed frame, ``BIT_EXACT`` (exact arithmetic). The surrogate tier
#: (``GRANULAR_EXCAVATION``) and the DEM/MPM ground truth are higher rungs behind this same
#: contract.
GRANULAR_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.granular",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED,),
    determinism_class=DeterminismClass.BIT_EXACT,
    fidelity=FidelityDescriptor(tier=FidelityTier.MASSMODEL),
)


@dataclass
class _DiggerState:
    """Mutable per-excavator excavation state advanced in place each tick."""

    agent_id: str
    frame: ReferenceFrame
    position_m: Vec
    regolith_density_kg_m3: float
    specific_energy_j_per_m3: float
    max_dig_rate_m3_s: float
    remaining_target_m3: float
    excavated_volume_m3: float
    battery_soc_j: float
    battery_floor_j: float
    mode: str | None

    def sample(self) -> StateSample:
        """The current state as a frame-explicit Core coupling sample (the stationary pose)."""
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


class GranularEngine:
    """The reduced-order excavation :class:`~astro_mine.sim.engines.RegimeEngine`.

    Owns the per-excavator dig bookkeeping: an ``excavate`` task sets a target volume, and
    each tick removes volume at the capped rate, accruing excavated mass and the work it
    costs. The excavated mass/volume are internal accumulators (no Core observation field
    yet); wiring them into resource accounting is RM-P0-SIM-06 / Bench."""

    def __init__(self, states: dict[str, _DiggerState]) -> None:
        self._states = states
        self._elapsed_s = 0.0

    @property
    def descriptor(self) -> EngineDescriptor:
        return GRANULAR_ENGINE_DESCRIPTOR

    def excavated_volume_m3(self, agent_id: str) -> float:
        """Total volume removed so far by ``agent_id`` (m³)."""
        return self._states[agent_id].excavated_volume_m3

    def excavated_mass_kg(self, agent_id: str) -> float:
        """Total mass removed so far by ``agent_id`` (kg)."""
        state = self._states[agent_id]
        return state.excavated_volume_m3 * state.regolith_density_kg_m3

    def apply_actions(self, actions: ActionBatch) -> None:
        """Set the dig target from an ``excavate`` task (a ``None`` target volume digs until
        commanded otherwise), or set the mode, for owned excavators."""
        for agent_id, action in actions_by_agent(actions).items():
            state = self._states.get(agent_id)
            if state is None:
                continue
            if action.kind is ActionKind.MODE and action.mode is not None:
                state.mode = action.mode.mode
            elif (
                action.kind is ActionKind.TASK
                and action.task is not None
                and action.task.excavate is not None
            ):
                target = action.task.excavate.target_volume_m3
                state.remaining_target_m3 = math.inf if target is None else target

    def advance(self, dt_s: float) -> None:
        self._elapsed_s += dt_s
        for state in self._states.values():
            volume = min(state.max_dig_rate_m3_s * dt_s, state.remaining_target_m3)
            if volume <= 0.0:
                continue
            state.excavated_volume_m3 += volume
            state.remaining_target_m3 -= volume
            work = state.specific_energy_j_per_m3 * volume
            state.battery_soc_j = max(state.battery_floor_j, state.battery_soc_j - work)

    def export_coupling_state(self) -> CouplingState:
        return CouplingState(
            sim_time_s=self._elapsed_s,
            samples=tuple(state.sample() for state in self._states.values()),
            # The dig reaches the value chain here (#64). Until this channel existed the excavated
            # mass accrued on private accessors and was dropped at the coupling boundary every
            # tick, so an excavator could remove 10 m^3 of regolith and the plant see nothing.
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


def granular_engine_factory(scenario: Scenario, rng: RngStreams) -> GranularEngine:
    """Build a :class:`GranularEngine` for the scenario's ``granular`` agents.

    Non-granular agents are skipped (the heterogeneous co-step is RM-P0-SIM-04); the
    reduced-order model is deterministic, so ``rng`` is unused. An excavator idles (digs
    nothing) until an ``excavate`` task gives it a target."""
    states: dict[str, _DiggerState] = {}
    for spec in scenario.agents:
        dyn = spec.dynamics
        if dyn.kind != "granular":
            continue
        states[spec.agent_id] = _DiggerState(
            agent_id=spec.agent_id,
            frame=spec.frame or scenario.frame,
            position_m=spec.initial_position_m,
            regolith_density_kg_m3=dyn.regolith_density_kg_m3,
            specific_energy_j_per_m3=dyn.specific_energy_j_per_m3,
            max_dig_rate_m3_s=dyn.max_dig_rate_m3_s,
            remaining_target_m3=0.0,
            excavated_volume_m3=0.0,
            battery_soc_j=spec.battery_soc_j,
            battery_floor_j=spec.battery_floor_j,
            mode=spec.mode,
        )
    return GranularEngine(states)
