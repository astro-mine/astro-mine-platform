# SPDX-License-Identifier: Apache-2.0
"""Per-asset power/thermal evolution — the lunar-night survival constraint (RM-P0-SIM-07).

A reduced-order, coupled power-and-thermal model that turns a SADF ``PowerBudget`` /
``ThermalBudget`` (:mod:`astro_mine.core.sadf.model`) into an evolving battery state-of-charge
and temperature — the physics behind "survive the ~14-day lunar night" (scenario §10, a flagship
hard problem).

The two balances are coupled the way they are on a real asset:

- **Power.** Generation (solar, scaled by the world's insolation and gated off in shadow; plus
  steady RTG / fuel-cell / external supply) minus the load (the SADF ``loads_by_mode`` draw for
  the agent's current mode, never below the housekeeping ``floor_w``) integrates the battery SoC,
  bounded by capacity and the storage charge/discharge limits. A loads-only asset (no storage —
  the externally-powered ISRU plant) tracks no SoC.
- **Thermal.** A lumped-capacitance node: heat in (electronics dissipation + RHU heaters + the
  thermostat heater when below the operating floor + conductive coupling to the surface) minus
  radiator emission to deep space integrates the temperature. **The thermostat heater draws
  electrical power**, so keeping warm through the night drains the battery — that coupling is the
  whole survival story.

The environment is the **Core world contract**: the model reads insolation (``solar_flux_w_m2``),
shadow state, and surface temperature off a :class:`~astro_mine.core.world.SurfacePoint` produced
by a :class:`~astro_mine.core.world.WorldProvider` — the same contract Worlds implements (worlds.md
§6) and Link consumes, so power/thermal couples to the world through the narrow waist, not a
private side-channel (conventions.md §1.1). :class:`ReferenceWorldProvider` is the always-works
local stand-in (a diurnal day/night cycle) until Worlds plumbs in its real illumination +
surface-thermal fields (RM-P0-WORLDS-03/04; its P0 ``temperature_k`` is a coarse equilibrium value
that sharpens to the diurnal column at P1). The model is deterministic (no RNG); higher-fidelity
thermal is Phase 1.

Backlog: RM-P0-SIM-07 -- astro-mine-sim#7
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from astro_mine.core.sadf.enums import PowerSourceKind
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.core.world import Illumination, IlluminationState, RegolithParams, SurfacePoint

if TYPE_CHECKING:
    from collections.abc import Iterable

    from astro_mine.core.sadf.model import PowerBudget, ThermalBudget
    from astro_mine.core.units import Epoch, ReferenceFrame
    from astro_mine.core.world import Vector

__all__ = [
    "LUNAR_DIURNAL_PERIOD_S",
    "PowerThermalModel",
    "PowerThermalState",
    "ReferenceWorldProvider",
    "default_initial_temperature",
]

#: Stefan-Boltzmann constant (W/m^2/K^4) — radiator emission to the deep-space sink.
_SIGMA_W_M2_K4 = 5.670374419e-8
#: Solar constant at 1 au (W/m^2) — the reference insolation solar generation scales against.
_SOLAR_CONSTANT_W_M2 = 1361.0
#: One lunar synodic day (s) — the reference diurnal period (~29.53 d; matches Worlds').
LUNAR_DIURNAL_PERIOD_S = 2_551_442.8
#: Lunar surface gravity (m/s^2), down the local normal — the reference world's gravity vector.
_LUNAR_GRAVITY_M_S2 = 1.62


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class PowerThermalState:
    """An asset's evolving power/thermal state: battery state-of-charge (J) and temperature (K,
    ``None`` for an asset with no thermal budget)."""

    soc_j: float
    temperature_k: float | None


def default_initial_temperature(thermal: ThermalBudget | None) -> float | None:
    """The starting temperature when a scenario does not declare one — the midpoint of the
    operating range, or ``None`` if the asset has no thermal budget."""
    if thermal is None:
        return None
    return 0.5 * (thermal.operating_range_k.min + thermal.operating_range_k.max)


def _sum_or_inf(values: Iterable[float | None]) -> float:
    """Sum the limits, treating any unset (``None``) limit as unbounded."""
    collected = list(values)
    if any(v is None for v in collected):
        return math.inf
    return math.fsum(v for v in collected if v is not None)


class PowerThermalModel:
    """The per-asset power/thermal evolution built from its SADF budgets (RM-P0-SIM-07)."""

    def __init__(
        self,
        power: PowerBudget,
        thermal: ThermalBudget | None = None,
        *,
        heat_capacity_j_per_k: float = 5.0e4,
        ground_conductance_w_per_k: float = 2.0,
        emissivity: float = 0.9,
        radiator_sink_k: float = 3.0,
    ) -> None:
        if heat_capacity_j_per_k <= 0.0:
            raise ValueError(f"heat_capacity_j_per_k must be > 0, got {heat_capacity_j_per_k}")
        self._has_storage = bool(power.storage)
        self._capacity_j = math.fsum(s.capacity_j for s in power.storage)
        self._max_charge_w = _sum_or_inf(s.max_charge_w for s in power.storage)
        self._max_discharge_w = _sum_or_inf(s.max_discharge_w for s in power.storage)
        self._floor_w = power.floor_w or 0.0
        self._loads = {load.mode: load.power_w for load in power.loads_by_mode}
        self._solar_w = _source_power(power, PowerSourceKind.SOLAR)
        self._steady_elec_w = _source_power(
            power, PowerSourceKind.RTG, PowerSourceKind.FUEL_CELL, PowerSourceKind.EXTERNAL
        )
        self._rhu_thermal_w = _source_power(power, PowerSourceKind.RHU)
        self._thermal = thermal
        self._heat_capacity = heat_capacity_j_per_k
        self._ground_conductance = ground_conductance_w_per_k
        self._emissivity = emissivity
        self._radiator_sink_k = radiator_sink_k

    @property
    def has_storage(self) -> bool:
        """Whether the asset carries a battery whose SoC this model evolves (a loads-only,
        externally-powered asset does not)."""
        return self._has_storage

    def step(
        self, state: PowerThermalState, dt_s: float, surface: SurfacePoint, mode: str | None
    ) -> PowerThermalState:
        """Advance the asset's power/thermal state by ``dt_s`` under the current mode and the
        world's surface conditions (insolation, shadow, surface temperature)."""
        heater_w = self._heater_load_w(state.temperature_k)
        mode_load_w = self._loads.get(mode, self._floor_w) if mode else self._floor_w
        load_w = max(self._floor_w, mode_load_w) + heater_w
        generation_w = self._steady_elec_w + self._solar_generation(surface.illumination)

        soc_j = state.soc_j
        if self._has_storage:
            net_w = _clamp(generation_w - load_w, -self._max_discharge_w, self._max_charge_w)
            soc_j = _clamp(state.soc_j + net_w * dt_s, 0.0, self._capacity_j)

        return PowerThermalState(
            soc_j=soc_j,
            temperature_k=self._step_temperature(state.temperature_k, dt_s, surface, heater_w),
        )

    def survived(self, state: PowerThermalState) -> bool:
        """Whether the asset is still alive: its battery is not depleted and (if it has a thermal
        budget) its temperature is within the survival range — the M0.2 night-survival predicate."""
        if self._has_storage and state.soc_j <= 0.0:
            return False
        survival = None if self._thermal is None else self._thermal.survival_range_k
        return not (
            survival is not None
            and state.temperature_k is not None
            and not survival.min <= state.temperature_k <= survival.max
        )

    # --- internals ---------------------------------------------------------------

    def _solar_generation(self, illumination: Illumination) -> float:
        if illumination.state is IlluminationState.SHADOW:
            return 0.0  # in a PSR / eclipse: no insolation, whatever the flux field says
        return self._solar_w * _clamp(illumination.solar_flux_w_m2 / _SOLAR_CONSTANT_W_M2, 0.0, 1.0)

    def _heater_load_w(self, temperature_k: float | None) -> float:
        """The thermostat heater's electrical draw — the heater power when the asset is below
        its operating floor, else zero (and zero with no thermal budget or heater)."""
        thermal = self._thermal
        if thermal is None or temperature_k is None or thermal.heater_power_w is None:
            return 0.0
        return thermal.heater_power_w if temperature_k < thermal.operating_range_k.min else 0.0

    def _step_temperature(
        self, temperature_k: float | None, dt_s: float, surface: SurfacePoint, heater_w: float
    ) -> float | None:
        thermal = self._thermal
        if thermal is None or temperature_k is None:
            return temperature_k
        heat_in_w = (thermal.dissipation_w or 0.0) + self._rhu_thermal_w + heater_w
        if thermal.surface_coupling:
            heat_in_w += self._ground_conductance * (surface.temperature_k - temperature_k)
        area = thermal.radiator_area_m2 or 0.0
        heat_out_w = (
            self._emissivity * _SIGMA_W_M2_K4 * area * (temperature_k**4 - self._radiator_sink_k**4)
        )
        return temperature_k + (heat_in_w - heat_out_w) / self._heat_capacity * dt_s


def _source_power(power: PowerBudget, *kinds: PowerSourceKind) -> float:
    wanted = set(kinds)
    return math.fsum(s.nominal_power_w for s in power.sources if s.kind in wanted)


class ReferenceWorldProvider:
    """A deterministic diurnal :class:`~astro_mine.core.world.WorldProvider` for the always-works
    local tier.

    Insolation follows a half-cosine arc over the lit half of a (configurable) diurnal period and
    is in shadow through the night; the surface temperature swings between a night floor and a
    daytime peak with the insolation. Position-uniform with flat terrain — the per-site,
    horizon-mapped illumination and real surface-thermal field are Worlds' job (RM-P0-WORLDS-03/04);
    this is the stand-in that makes night survival real offline, satisfying the same Core contract
    Worlds will (so it is a drop-in)."""

    def __init__(
        self,
        *,
        period_s: float = LUNAR_DIURNAL_PERIOD_S,
        peak_flux_w_m2: float = _SOLAR_CONSTANT_W_M2,
        day_temp_k: float = 350.0,
        night_temp_k: float = 100.0,
        frame: ReferenceFrame = MOON_BODY_FIXED,
    ) -> None:
        if period_s <= 0.0:
            raise ValueError(f"period_s must be > 0, got {period_s}")
        self._period_s = period_s
        self._peak_flux = peak_flux_w_m2
        self._day_temp = day_temp_k
        self._night_temp = night_temp_k
        self._frame = frame
        self._gravity: Vector = (0.0, 0.0, -_LUNAR_GRAVITY_M_S2)

    @property
    def frame(self) -> ReferenceFrame:
        return self._frame

    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        seconds = 0.0 if epoch is None else epoch.tdb_seconds
        phase = (seconds % self._period_s) / self._period_s  # 0..1, noon at 0.5
        if 0.25 <= phase < 0.75:
            elevation = math.sin(math.pi * (phase - 0.25) / 0.5)  # 0 at sunrise/set, 1 at noon
            flux = self._peak_flux * elevation
            state = IlluminationState.LIT
        else:
            flux = 0.0
            state = IlluminationState.SHADOW
        ground = self._night_temp + (self._day_temp - self._night_temp) * (flux / self._peak_flux)
        return SurfacePoint(
            frame=self._frame,
            elevation_m=0.0,
            surface_normal=(0.0, 0.0, 1.0),
            gravity=self._gravity,
            illumination=Illumination(state=state, solar_flux_w_m2=flux),
            temperature_k=ground,
            regolith=RegolithParams(),
        )

    def ray_intersect(self, origin: Vector, direction: Vector) -> Vector | None:
        return None  # flat reference terrain casts no occluding geometry

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        return True  # no terrain to occlude line-of-sight in the reference world
