"""Illumination-driven per-cell surface thermal (RM-P1-WORLDS-13).

The Phase-1 fidelity upgrade to RM-P0-WORLDS-04: instead of forcing the 1-D thermophysical
column with a *representative per-class* insolation arc (a self-contained sun-elevation
sinusoid), drive it with the world's **actual per-cell insolation** — the
:class:`~astro_mine.worlds.illumination.IlluminationModel`'s horizon-shadowed Sun visibility
crossed with the SPICE Sun incidence over a diurnal window. A permanently-shadowed cell then
receives zero direct insolation and settles to the deep-shadow floor; a lit cell tracks the
real Sun track and its local horizon; slope/latitude/local-shadowing differences show up as
per-cell diurnal curves — none of which the idealized arc can express.

The representative named-class curves (RM-P0-WORLDS-04, ``diurnal_curve``) stay as the fast
low-fidelity tier (worlds.md §8 multi-fidelity); this is the high-fidelity tier, wired into
the Core Environment-API provider through its injectable ``ThermalSource`` hook — replacing
the coarse radiative-equilibrium placeholder RM-P0-WORLDS-06 falls back to.

Curves are cached per cell (the forcing sampling + column solve is the cost), keyed by grid
row/col, so a per-tick Sim query is an O(1) interpolation of a cached periodic curve.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import rasterio.transform

from astro_mine.core.units import Epoch, TimeScale
from astro_mine.worlds.illumination import IlluminationModel
from astro_mine.worlds.provider._geometry import solar_flux
from astro_mine.worlds.thermal import (
    BUNDLE_SCHEMA,
    DiurnalCurve,
    ThermalClass,
    _worlds_version,
)
from astro_mine.worlds.thermal._solver import (
    LUNAR_DIURNAL_PERIOD_S,
    diurnal_hash,
    solve_surface_curve,
)

__all__ = [
    "IlluminationThermalSource",
    "illumination_insolation",
    "solve_illumination_driven_curve",
]


def illumination_insolation(
    illumination: IlluminationModel,
    map_x: float,
    map_y: float,
    *,
    reference_epoch: Epoch,
    albedo: float,
    period_s: float = LUNAR_DIURNAL_PERIOD_S,
    n_samples: int = 48,
) -> Callable[[float], float]:
    """Absorbed-flux forcing ``phase -> W/m^2`` from the real per-cell insolation.

    Samples the cell's Sun visibility + elevation over one diurnal ``period_s`` (``n_samples``
    epochs anchored at ``reference_epoch``), converts each to absorbed flux
    ``(1-albedo) * SOLAR_CONSTANT * sin(elevation)`` while the Sun clears the local terrain
    horizon (else 0), and returns a periodic linear-interpolating callable — the drop-in
    forcing for :func:`~astro_mine.worlds.thermal._solver.solve_surface_curve`.
    """
    phases = np.arange(n_samples, dtype=np.float64) / n_samples
    absorbed = np.empty(n_samples, dtype=np.float64)
    for i, phase in enumerate(phases):
        epoch = Epoch(
            tdb_seconds=reference_epoch.tdb_seconds + float(phase) * period_s,
            scale=TimeScale.TDB,
        )
        lit, elevation_deg = illumination.illumination_at(map_x, map_y, epoch)
        absorbed[i] = (1.0 - albedo) * solar_flux(elevation_deg, lit=lit)
    # Close the loop for periodic wraparound interpolation.
    phase_grid = np.append(phases, 1.0)
    flux_grid = np.append(absorbed, absorbed[0])

    def flux_fn(phase: float) -> float:
        return float(np.interp(phase % 1.0, phase_grid, flux_grid))

    return flux_fn


def solve_illumination_driven_curve(
    illumination: IlluminationModel,
    map_x: float,
    map_y: float,
    thermal_class: ThermalClass,
    *,
    reference_epoch: Epoch,
    period_s: float = LUNAR_DIURNAL_PERIOD_S,
    n_samples: int = 48,
    **solver_kwargs: Any,
) -> DiurnalCurve:
    """Solve the 1-D column at a cell under its **real** per-cell insolation forcing.

    Uses ``thermal_class``'s thermophysics (inertia/density/albedo/emissivity/environment
    flux) but replaces the idealized arc with :func:`illumination_insolation` at
    ``(map_x, map_y)``. Returns a :class:`~astro_mine.worlds.thermal.DiurnalCurve` whose
    provenance records the cell + reference epoch so it reproduces deterministically.
    """
    flux_fn = illumination_insolation(
        illumination,
        map_x,
        map_y,
        reference_epoch=reference_epoch,
        albedo=thermal_class.albedo,
        period_s=period_s,
        n_samples=n_samples,
    )
    phases, temperatures, converged, periods_run = solve_surface_curve(
        thermal_inertia_tiu=thermal_class.thermal_inertia_tiu,
        density_kg_m3=thermal_class.density_kg_m3,
        specific_heat_j_kg_k=thermal_class.specific_heat_j_kg_k,
        albedo=thermal_class.albedo,
        emissivity=thermal_class.emissivity,
        peak_sun_elevation_deg=thermal_class.peak_sun_elevation_deg,
        environment_flux_w_m2=thermal_class.environment_flux_w_m2,
        period_s=period_s,
        absorbed_flux=flux_fn,
        **solver_kwargs,
    )
    meta: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "terrain_class": thermal_class.name,
        "forcing": "illumination_driven",
        "cell": [float(map_x), float(map_y)],
        "reference_epoch_tdb_s": reference_epoch.tdb_seconds,
        "illumination_hash": illumination.illumination_hash,
        "params": vars(thermal_class),
        "solver": {"period_s": period_s, "n_samples": n_samples, **solver_kwargs},
        "toolchain": {"astro_mine_worlds": _worlds_version(), "numpy": np.__version__},
    }
    return DiurnalCurve(
        terrain_class=thermal_class.name,
        phases=phases,
        temperatures_k=temperatures,
        period_s=period_s,
        converged=converged,
        periods_run=periods_run,
        thermal_hash=diurnal_hash(temperatures, meta),
        params={**dict(vars(thermal_class)), "forcing": "illumination_driven"},
    )


class IlluminationThermalSource:
    """A per-cell, illumination-driven :class:`~astro_mine.worlds.provider.ThermalSource`.

    Injected into :class:`~astro_mine.worlds.provider.DemWorldProvider`, it replaces the
    coarse radiative-equilibrium temperature placeholder with a real per-cell diurnal curve:
    ``temperature_k`` interpolates the cached, illumination-forced curve for the queried cell
    at the queried epoch's diurnal phase. Curves are solved lazily and cached per grid cell.
    """

    def __init__(
        self,
        illumination: IlluminationModel,
        thermal_class: ThermalClass,
        *,
        reference_epoch: Epoch,
        period_s: float = LUNAR_DIURNAL_PERIOD_S,
        n_samples: int = 48,
        **solver_kwargs: Any,
    ) -> None:
        self._illumination = illumination
        self._class = thermal_class
        self._reference = reference_epoch
        self._period_s = period_s
        self._n_samples = n_samples
        self._solver_kwargs = solver_kwargs
        self._cache: dict[tuple[int, int], DiurnalCurve] = {}

    def curve_at(self, map_x: float, map_y: float) -> DiurnalCurve:
        """The (cached) illumination-driven diurnal curve for the cell at ``(map_x, map_y)``."""
        row, col = rasterio.transform.rowcol(self._illumination.transform, map_x, map_y)
        key = (int(row), int(col))
        if key not in self._cache:
            self._cache[key] = solve_illumination_driven_curve(
                self._illumination,
                map_x,
                map_y,
                self._class,
                reference_epoch=self._reference,
                period_s=self._period_s,
                n_samples=self._n_samples,
                **self._solver_kwargs,
            )
        return self._cache[key]

    def temperature_k(
        self, *, map_x: float, map_y: float, epoch: Epoch, solar_flux_w_m2: float
    ) -> float:
        """Surface temperature (K) at ``(map_x, map_y)`` and ``epoch`` from the cell curve."""
        curve = self.curve_at(map_x, map_y)
        phase = ((epoch.tdb_seconds - self._reference.tdb_seconds) / self._period_s) % 1.0
        return curve.temperature_at(phase)
