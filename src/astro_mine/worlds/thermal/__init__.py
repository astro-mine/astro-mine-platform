"""Surface thermal (first-cut) - 1-D thermophysical diurnal curves (RM-P0-WORLDS-04).

Per-terrain-class precomputed diurnal surface-temperature curves, driving the ~14-Earth-day
lunar-night survival constraint (scenario §5; worlds.md §11/§12). Each named terrain class
(:data:`TERRAIN_CLASSES`) carries representative thermophysical parameters and an idealized
diurnal insolation profile; :func:`solve_diurnal_curve` integrates the 1-D transient
heat-diffusion column (``_solver``) to periodic steady state and returns the surface curve.
:func:`diurnal_curve` is the cached registry lookup the rest of the platform queries.

**First-cut scope.** The forcing is a *representative* per-class insolation arc (a
self-contained idealization), not the world's actual horizon-mapped illumination - the
per-cell, illumination-driven thermal that consumes the RM-P0-WORLDS-03 horizon maps is a
Phase-1 refinement. Higher-fidelity 3-D thermal and dust are also deferred (worlds.md §11).
The curves are *parameter data* Sim's power/thermal model (RM-P0-SIM-07) consumes; the
constitutive use lives in Sim.

**Error bounds.** The sunlit peak tracks the radiative-equilibrium subsolar value
``T = ((1-A)*S / (eps*sigma))^(1/4)`` (lunar subsolar ~390 K) to within the conduction draw
of a low-inertia regolith; the permanently-shadowed floor sits at the few-tens-of-K level set
by the lumped environmental flux (Diviner-class PSR floors ~30-50 K); the post-sunset night
floor of sunlit terrain settles near ~90-110 K, sustained by substrate conduction. These are
order-level first-cut bounds, not validated against on-world data (none in CI; charter §8).

Backlog: RM-P0-WORLDS-04 - https://github.com/astro-mine/astro-mine-worlds/issues/4
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.worlds.thermal._solver import (
    LUNAR_DIURNAL_PERIOD_S,
    diurnal_hash,
    solve_surface_curve,
)

__all__ = [
    "BUNDLE_SCHEMA",
    "TERRAIN_CLASSES",
    "DiurnalCurve",
    "IlluminationThermalSource",
    "ThermalClass",
    "ThermalError",
    "diurnal_curve",
    "illumination_insolation",
    "solve_diurnal_curve",
    "solve_illumination_driven_curve",
]

BUNDLE_SCHEMA = "astro-mine-worlds/thermal/v0.1"


class ThermalError(Exception):
    """Raised on an unknown terrain class."""


@dataclass(frozen=True)
class ThermalClass:
    """Representative thermophysical parameters + insolation profile for a terrain class.

    ``thermal_inertia_tiu`` is in SI thermal-inertia units (J m^-2 K^-1 s^-1/2);
    ``peak_sun_elevation_deg`` is the noon Sun elevation of the class's representative diurnal
    arc (0 for a permanently shadowed floor); ``environment_flux_w_m2`` is the lumped
    scattered-IR + geothermal background that sets the deep-shadow floor.
    """

    name: str
    thermal_inertia_tiu: float
    density_kg_m3: float
    specific_heat_j_kg_k: float
    albedo: float
    emissivity: float
    peak_sun_elevation_deg: float
    environment_flux_w_m2: float


#: The Phase-0 terrain classes (scenario §5): a hot equatorial reference, the Shackleton-class
#: low-grazing-sun polar ridge, and a permanently shadowed crater floor. ~0.14 W m^-2 of lumped
#: environmental flux fixes the shadow floor near 40 K.
TERRAIN_CLASSES: dict[str, ThermalClass] = {
    "equatorial": ThermalClass(
        name="equatorial",
        thermal_inertia_tiu=55.0,
        density_kg_m3=1500.0,
        specific_heat_j_kg_k=700.0,
        albedo=0.12,
        emissivity=0.95,
        peak_sun_elevation_deg=90.0,
        environment_flux_w_m2=0.14,
    ),
    "polar_lit": ThermalClass(
        name="polar_lit",
        thermal_inertia_tiu=70.0,
        density_kg_m3=1500.0,
        specific_heat_j_kg_k=700.0,
        albedo=0.12,
        emissivity=0.95,
        peak_sun_elevation_deg=12.0,
        environment_flux_w_m2=0.14,
    ),
    "crater_floor": ThermalClass(
        name="crater_floor",
        thermal_inertia_tiu=100.0,
        density_kg_m3=1500.0,
        specific_heat_j_kg_k=700.0,
        albedo=0.12,
        emissivity=0.95,
        peak_sun_elevation_deg=0.0,  # permanently shadowed: no direct insolation
        environment_flux_w_m2=0.14,
    ),
}


def _worlds_version() -> str:
    try:
        return importlib.metadata.version("astro-mine-platform")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed in dev/CI
        return "0+unknown"


@dataclass(frozen=True)
class DiurnalCurve:
    """A periodic diurnal surface-temperature curve for a terrain class, with provenance.

    ``phases`` are sample points in [0, 1) over ``period_s`` (noon at 0.5) and
    ``temperatures_k`` the periodic surface temperature at each. ``converged`` flags whether
    the solve reached periodic steady state within tolerance.
    """

    terrain_class: str
    phases: NDArray[np.float64]
    temperatures_k: NDArray[np.float64]
    period_s: float
    converged: bool
    periods_run: int
    thermal_hash: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def night_floor_k(self) -> float:
        """The coldest surface temperature over the diurnal cycle (K)."""
        return float(np.min(self.temperatures_k))

    @property
    def peak_k(self) -> float:
        """The warmest surface temperature over the diurnal cycle (K)."""
        return float(np.max(self.temperatures_k))

    def temperature_at(self, phase: float) -> float:
        """Surface temperature (K) at diurnal ``phase``, linearly interpolated with wraparound."""
        wrapped = phase % 1.0
        extended_phases = np.append(self.phases, 1.0)
        extended_temps = np.append(self.temperatures_k, self.temperatures_k[0])
        return float(np.interp(wrapped, extended_phases, extended_temps))


def solve_diurnal_curve(
    thermal_class: ThermalClass, *, period_s: float = LUNAR_DIURNAL_PERIOD_S, **solver_kwargs: Any
) -> DiurnalCurve:
    """Solve the 1-D thermophysical column for ``thermal_class`` into a :class:`DiurnalCurve`.

    ``solver_kwargs`` forward to :func:`~astro_mine.worlds.thermal._solver.solve_surface_curve`
    (e.g. ``n_phase``, ``substeps``, ``n_layers``, ``max_periods``, ``tol_k``).
    """
    phases, temperatures, converged, periods_run = solve_surface_curve(
        thermal_inertia_tiu=thermal_class.thermal_inertia_tiu,
        density_kg_m3=thermal_class.density_kg_m3,
        specific_heat_j_kg_k=thermal_class.specific_heat_j_kg_k,
        albedo=thermal_class.albedo,
        emissivity=thermal_class.emissivity,
        peak_sun_elevation_deg=thermal_class.peak_sun_elevation_deg,
        environment_flux_w_m2=thermal_class.environment_flux_w_m2,
        period_s=period_s,
        **solver_kwargs,
    )
    meta: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "terrain_class": thermal_class.name,
        "params": vars(thermal_class),
        "solver": {"period_s": period_s, "n_phase": int(phases.shape[0]), **solver_kwargs},
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
        params=dict(vars(thermal_class)),
    )


@cache
def diurnal_curve(terrain_class: str) -> DiurnalCurve:
    """The precomputed diurnal temperature curve for a named terrain class (cached).

    Raises :class:`ThermalError` for an unknown class.
    """
    try:
        thermal_class = TERRAIN_CLASSES[terrain_class]
    except KeyError:
        known = ", ".join(sorted(TERRAIN_CLASSES))
        raise ThermalError(f"unknown terrain class {terrain_class!r}; known: {known}") from None
    return solve_diurnal_curve(thermal_class)


# The illumination-driven per-cell tier (RM-P1-WORLDS-13) is imported last: it depends on
# the class curves + solver above and on the illumination/provider packages, so importing it
# here (rather than those importing it) keeps the dependency acyclic.
from astro_mine.worlds.thermal._illumination import (  # noqa: E402
    IlluminationThermalSource,
    illumination_insolation,
    solve_illumination_driven_curve,
)
