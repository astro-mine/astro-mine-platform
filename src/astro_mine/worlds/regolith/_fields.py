"""Regolith terramechanics parameter fields — pure NumPy kernels (RM-P0-WORLDS-05).

Spatial fields of the five regolith mechanical parameters — bulk density, cohesion,
friction angle, bearing capacity, thermal inertia — each with a **companion uncertainty**
layer. These are the *inputs* Sim's contact/excavation constitutive law consumes
(RM-P0-SIM-03); the law itself lives in Sim, never here (worlds.md §6, "parameters here,
physics there").

Engine-neutral array kernels (no rasterio, no IO), mirroring ``terrain/_layers.py``.
Uncertainty is first-class (conventions.md §1.6): the Phase-0 mean field is the documented
lunar prior, spatially **uniform** (there is no per-pixel lunar regolith map yet), while the
uncertainty varies spatially — inflated where the source DEM is void, exactly as terrain
inflates its vertical uncertainty there. A documented, off-by-default ``slope_sensitivity``
hook can modulate the means by terrain slope for callers that want it.

The nominal values in :data:`DEFAULT_LUNAR_PRIOR` are **illustrative baselines** with
explicit uncertainty (Lunar Sourcebook nominal ranges), not authoritative per-site data;
validation against analytic/lab terramechanics is Sim's job (RM-P0-SIM-10).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.worlds._hashing import canonical_meta_bytes

__all__ = [
    "DEFAULT_LUNAR_PRIOR",
    "PARAM_NAMES",
    "ParamPrior",
    "RegolithPrior",
    "regolith_hash",
    "regolith_layers",
]

F32 = np.float32

#: The five terramechanics parameters, in a fixed order (worlds.md §6; Core ``RegolithParams``).
PARAM_NAMES: tuple[str, ...] = (
    "bulk_density",
    "cohesion",
    "friction_angle",
    "bearing_capacity",
    "thermal_inertia",
)


@dataclass(frozen=True)
class ParamPrior:
    """A single parameter's prior: a mean, a 1-sigma uncertainty, and a slope-modulation hook.

    ``slope_sensitivity`` is the fractional change in the mean per degree of terrain slope
    (default ``0.0`` — the mean is spatially uniform). It is an explicit, off-by-default hook,
    not an asserted physical law.
    """

    mean: float
    uncertainty: float
    slope_sensitivity: float = 0.0


@dataclass(frozen=True)
class RegolithPrior:
    """The prior for all five parameters plus the void-uncertainty inflation factor."""

    bulk_density: ParamPrior
    cohesion: ParamPrior
    friction_angle: ParamPrior
    bearing_capacity: ParamPrior
    thermal_inertia: ParamPrior
    void_uncertainty_factor: float = 5.0

    def items(self) -> Iterator[tuple[str, ParamPrior]]:
        """Yield ``(param_name, ParamPrior)`` in :data:`PARAM_NAMES` order."""
        for name in PARAM_NAMES:
            yield name, getattr(self, name)

    def uses_slope(self) -> bool:
        """Whether any parameter modulates its mean by slope (so a slope layer is needed)."""
        return any(p.slope_sensitivity != 0.0 for _, p in self.items())


#: Illustrative lunar regolith nominal values with uncertainty (Lunar Sourcebook ranges):
#: bulk density ~1500 kg/m^3, cohesion ~0.5 kPa, friction ~40 deg, bearing ~10 kPa, thermal
#: inertia ~55 tiu. Baselines, not authoritative per-site data (conventions.md §1.6).
DEFAULT_LUNAR_PRIOR = RegolithPrior(
    bulk_density=ParamPrior(mean=1500.0, uncertainty=200.0),
    cohesion=ParamPrior(mean=500.0, uncertainty=300.0),
    friction_angle=ParamPrior(mean=40.0, uncertainty=5.0),
    bearing_capacity=ParamPrior(mean=1.0e4, uncertainty=5.0e3),
    thermal_inertia=ParamPrior(mean=55.0, uncertainty=15.0),
)


def regolith_layers(
    prior: RegolithPrior,
    void_mask: NDArray[np.bool_],
    slope_deg: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float32]]:
    """Build the mean + companion-uncertainty layers for all five parameters on the grid.

    Each parameter yields ``<name>`` (mean) and ``<name>_uncertainty`` (1-sigma) ``float32``
    layers shaped like ``void_mask``. The mean is the prior mean — optionally modulated by
    ``slope_deg`` when a parameter declares a ``slope_sensitivity`` (clamped non-negative);
    the uncertainty is the prior uncertainty, inflated by ``void_uncertainty_factor`` where
    the DEM is void (the only spatial structure in the default, data-driven not invented).
    """
    if slope_deg is not None and slope_deg.shape != void_mask.shape:
        raise ValueError(f"slope_deg shape {slope_deg.shape} != void_mask shape {void_mask.shape}")
    layers: dict[str, NDArray[np.float32]] = {}
    for name, param in prior.items():
        mean = np.full(void_mask.shape, param.mean, dtype=np.float64)
        if slope_deg is not None and param.slope_sensitivity != 0.0:
            mean = np.clip(mean * (1.0 + param.slope_sensitivity * slope_deg), 0.0, None)
        uncertainty = np.full(void_mask.shape, param.uncertainty, dtype=np.float64)
        uncertainty[void_mask] *= prior.void_uncertainty_factor
        layers[name] = mean.astype(F32)
        layers[f"{name}_uncertainty"] = uncertainty.astype(F32)
    return layers


def regolith_hash(layers: dict[str, NDArray[np.float32]], meta: dict[str, Any]) -> str:
    """A deterministic ``sha256:`` digest over the layer arrays + canonical metadata.

    ``meta``'s toolchain is provenance and is not covered — see :mod:`astro_mine.worlds._hashing`.
    """
    h = hashlib.sha256()
    for name in sorted(layers):
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(layers[name]).tobytes())
    h.update(canonical_meta_bytes(meta))
    return f"sha256:{h.hexdigest()}"
