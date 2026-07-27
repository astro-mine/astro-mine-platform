"""1-D thermophysical solver kernels - pure NumPy, IO-free (RM-P0-WORLDS-04).

The transient heat-diffusion engine behind the per-terrain-class diurnal curves: a vertical
regolith column on a geometric grid scaled to the thermal skin depth, integrated to periodic
steady state under a representative diurnal insolation forcing. Backward-Euler conduction
(unconditionally stable, tridiagonal Thomas solve) with the surface radiative loss linearized
each step about the previous surface temperature.

Pure array kernels (no rasterio, SPICE, or IO; those are not needed here) so they are cheap
to unit-test and fully type-checked. All quantities are SI; angles are degrees.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.worlds._hashing import canonical_meta_bytes

__all__ = [
    "LUNAR_DIURNAL_PERIOD_S",
    "SOLAR_CONSTANT_W_M2",
    "STEFAN_BOLTZMANN_W_M2_K4",
    "build_layers",
    "diurnal_hash",
    "insolation_flux",
    "skin_depth",
    "solve_surface_curve",
    "thermal_conductivity",
]

#: Total solar irradiance at ~1 AU (W m^-2).
SOLAR_CONSTANT_W_M2 = 1361.0

#: Stefan-Boltzmann constant (W m^-2 K^-4).
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8

#: Lunar synodic day (s) - the diurnal forcing period that drives the ~14-day night.
LUNAR_DIURNAL_PERIOD_S = 2_551_442.8

F64 = np.float64


def thermal_conductivity(
    thermal_inertia_tiu: float, density_kg_m3: float, specific_heat_j_kg_k: float
) -> float:
    """Thermal conductivity k (W m^-1 K^-1) from thermal inertia I and rho, c.

    Thermal inertia ``I = sqrt(k * rho * c)`` is the governing parameter, so
    ``k = I^2 / (rho * c)``.
    """
    return thermal_inertia_tiu**2 / (density_kg_m3 * specific_heat_j_kg_k)


def skin_depth(
    thermal_inertia_tiu: float,
    density_kg_m3: float,
    specific_heat_j_kg_k: float,
    period_s: float,
) -> float:
    """Thermal skin depth (m): the e-folding depth of the diurnal temperature wave.

    ``d_s = sqrt(kappa * P / pi)`` with thermal diffusivity ``kappa = k / (rho * c)``. The
    column is sized to a few skin depths so the diurnal wave is fully captured.
    """
    k = thermal_conductivity(thermal_inertia_tiu, density_kg_m3, specific_heat_j_kg_k)
    kappa = k / (density_kg_m3 * specific_heat_j_kg_k)
    return math.sqrt(kappa * period_s / math.pi)


def build_layers(
    n_layers: int, total_depth_m: float, growth: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Geometric vertical grid: layer thicknesses and centre depths (m).

    Thin near the surface (to resolve the steep diurnal gradient) and thickening with depth,
    summing to ``total_depth_m``.
    """
    raw = growth ** np.arange(n_layers, dtype=F64)
    thickness = raw / raw.sum() * total_depth_m
    centers = np.cumsum(thickness) - 0.5 * thickness
    return thickness, centers


def insolation_flux(
    phase: float,
    peak_sun_elevation_deg: float,
    albedo: float,
    *,
    solar_constant_w_m2: float = SOLAR_CONSTANT_W_M2,
) -> float:
    """Absorbed solar flux (W m^-2) at diurnal ``phase`` in [0, 1), noon at 0.5.

    The Sun elevation follows ``peak_elev * cos(2*pi*(phase-0.5))`` (a representative diurnal
    arc); absorbed flux is ``(1-albedo) * S * sin(elevation)`` while the Sun is up, else 0. A
    class with ``peak_sun_elevation_deg = 0`` (a permanently shadowed crater floor) receives
    no direct insolation. ``solar_constant_w_m2`` defaults to the ~1 AU value; a non-lunar
    body pack passes its own (Mars ~586 W m^-2, RM-P1-WORLDS-11).
    """
    elevation_deg = peak_sun_elevation_deg * math.cos(2.0 * math.pi * (phase - 0.5))
    if elevation_deg <= 0.0:
        return 0.0
    return (1.0 - albedo) * solar_constant_w_m2 * math.sin(math.radians(elevation_deg))


def _thomas(
    sub: NDArray[np.float64],
    diag: NDArray[np.float64],
    sup: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a tridiagonal system (Thomas algorithm). ``sub[0]`` and ``sup[-1]`` are unused."""
    n = diag.shape[0]
    c_prime = np.empty(n, dtype=F64)
    d_prime = np.empty(n, dtype=F64)
    c_prime[0] = sup[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - sub[i] * c_prime[i - 1]
        c_prime[i] = sup[i] / denom
        d_prime[i] = (rhs[i] - sub[i] * d_prime[i - 1]) / denom
    x = np.empty(n, dtype=F64)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def _backward_euler_step(
    temps: NDArray[np.float64],
    absorbed_flux_w_m2: float,
    dt_s: float,
    heat_capacity: NDArray[np.float64],
    conductance: NDArray[np.float64],
    environment_flux_w_m2: float,
    emissivity: float,
) -> NDArray[np.float64]:
    """One implicit conduction step with a linearized surface radiative boundary.

    Surface node: absorbed solar + lumped environmental flux - emitted IR (linearized about
    the current surface temperature) - conduction down. Bottom node: insulated. Returns the
    updated temperature column.
    """
    n = temps.shape[0]
    sub = np.zeros(n, dtype=F64)
    diag = np.empty(n, dtype=F64)
    sup = np.zeros(n, dtype=F64)
    rhs = np.empty(n, dtype=F64)

    # Surface radiative loss eps*sigma*T^4 linearized as a + b*T about temps[0].
    t0 = float(temps[0])
    b = 4.0 * emissivity * STEFAN_BOLTZMANN_W_M2_K4 * t0**3
    a = emissivity * STEFAN_BOLTZMANN_W_M2_K4 * t0**4 - b * t0

    # Surface node 0.
    diag[0] = heat_capacity[0] + dt_s * (conductance[0] + b)
    sup[0] = -dt_s * conductance[0]
    rhs[0] = heat_capacity[0] * temps[0] + dt_s * (absorbed_flux_w_m2 + environment_flux_w_m2 - a)

    # Interior nodes.
    for i in range(1, n - 1):
        sub[i] = -dt_s * conductance[i - 1]
        diag[i] = heat_capacity[i] + dt_s * (conductance[i - 1] + conductance[i])
        sup[i] = -dt_s * conductance[i]
        rhs[i] = heat_capacity[i] * temps[i]

    # Bottom node (insulated).
    sub[n - 1] = -dt_s * conductance[n - 2]
    diag[n - 1] = heat_capacity[n - 1] + dt_s * conductance[n - 2]
    rhs[n - 1] = heat_capacity[n - 1] * temps[n - 1]

    return _thomas(sub, diag, sup, rhs)


def solve_surface_curve(
    *,
    thermal_inertia_tiu: float,
    density_kg_m3: float,
    specific_heat_j_kg_k: float,
    albedo: float,
    emissivity: float,
    peak_sun_elevation_deg: float,
    environment_flux_w_m2: float,
    period_s: float = LUNAR_DIURNAL_PERIOD_S,
    n_phase: int = 48,
    substeps: int = 8,
    n_layers: int = 30,
    depth_skin_depths: float = 8.0,
    layer_growth: float = 1.2,
    max_periods: int = 40,
    tol_k: float = 0.5,
    t_init_k: float | None = None,
    absorbed_flux: Callable[[float], float] | None = None,
    solar_constant_w_m2: float = SOLAR_CONSTANT_W_M2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool, int]:
    """Integrate the 1-D column to periodic steady state; return the diurnal surface curve.

    Returns ``(phases, temperatures_k, converged, periods_run)`` where ``phases`` are the
    ``n_phase`` sample phases in [0, 1) and ``temperatures_k`` the periodic surface
    temperature at each. ``converged`` is whether successive periods agreed within ``tol_k``.
    ``t_init_k`` defaults to the radiative-equilibrium temperature of the diurnal-mean flux,
    a warm start that converges in a handful of periods.

    ``absorbed_flux`` is the surface forcing: a callable ``phase -> absorbed W/m^2``. It
    defaults to the representative per-class insolation arc
    (:func:`insolation_flux` of ``peak_sun_elevation_deg``); passing a real per-cell
    illumination series (RM-P1-WORLDS-13) drives the column with the world's actual
    horizon-shadowed insolation instead of the idealization.
    """
    if absorbed_flux is None:

        def absorbed_flux(phase: float) -> float:
            return insolation_flux(
                phase, peak_sun_elevation_deg, albedo, solar_constant_w_m2=solar_constant_w_m2
            )

    k = thermal_conductivity(thermal_inertia_tiu, density_kg_m3, specific_heat_j_kg_k)
    d_s = skin_depth(thermal_inertia_tiu, density_kg_m3, specific_heat_j_kg_k, period_s)
    thickness, centers = build_layers(n_layers, depth_skin_depths * d_s, layer_growth)
    heat_capacity = density_kg_m3 * specific_heat_j_kg_k * thickness
    conductance = k / np.diff(centers)  # length n_layers - 1

    if t_init_k is None:
        mean_absorbed = float(np.mean([absorbed_flux(p / n_phase) for p in range(n_phase)]))
        mean_flux = environment_flux_w_m2 + mean_absorbed
        t_init_k = max((mean_flux / (emissivity * STEFAN_BOLTZMANN_W_M2_K4)) ** 0.25, 3.0)
    temps = np.full(n_layers, t_init_k, dtype=F64)
    phases = np.arange(n_phase, dtype=F64) / n_phase
    dt_s = period_s / (n_phase * substeps)

    curve = np.empty(n_phase, dtype=F64)
    previous = np.empty(n_phase, dtype=F64)
    converged = False
    periods_run = 0
    for period_idx in range(max_periods):
        periods_run = period_idx + 1
        for p in range(n_phase):
            curve[p] = temps[0]
            for s in range(substeps):
                phase = (p + s / substeps) / n_phase
                absorbed = absorbed_flux(phase)
                temps = _backward_euler_step(
                    temps,
                    absorbed,
                    dt_s,
                    heat_capacity,
                    conductance,
                    environment_flux_w_m2,
                    emissivity,
                )
        if period_idx > 0 and float(np.max(np.abs(curve - previous))) < tol_k:
            converged = True
            break
        previous = curve.copy()
    return phases, curve.copy(), converged, periods_run


def diurnal_hash(temperatures_k: NDArray[np.float64], meta: dict[str, Any]) -> str:
    """A deterministic ``sha256:`` digest over the curve plus canonical metadata.

    Same class params + solver settings reproduce the same hash (conventions.md §5, §11), matching
    the other Worlds products. ``meta``'s toolchain is provenance and is not covered — see
    :mod:`astro_mine.worlds._hashing`.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(temperatures_k, dtype=F64).tobytes())
    h.update(canonical_meta_bytes(meta))
    return f"sha256:{h.hexdigest()}"
