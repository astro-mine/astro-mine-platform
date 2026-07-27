"""Regolith terramechanics parameter field — parameters only (RM-P0-WORLDS-05).

A grid-aligned spatial field of the five regolith mechanical parameters — bulk density,
cohesion, friction angle, bearing capacity, thermal inertia — each with a **companion
uncertainty** layer, content-addressed and georeferenced to a terrain product's CRS/grid so
Sim consumes it **without re-projection** (worlds.md §6; ``LUNAR-FR-003``). These are the
*inputs* Sim's contact/excavation constitutive law (RM-P0-SIM-03) reads at contact points;
the constitutive law itself is Sim's — **no physics here** (separation of concerns).

``RegolithField.params(x, y)`` returns Core's :class:`~astro_mine.core.world.RegolithParams`
(the means Sim consumes); ``RegolithField.uncertainty(x, y)`` returns the per-parameter
1-sigma values in the same fields. The Phase-0 mean field is the documented lunar prior
(spatially uniform — there is no per-pixel regolith map yet); spatial structure lives in the
uncertainty, inflated where the DEM is void. Nominal values are illustrative baselines with
uncertainty (conventions.md §1.6); validation belongs to Sim (RM-P0-SIM-10).

Backlog: RM-P0-WORLDS-05 — https://github.com/astro-mine/astro-mine-worlds/issues/5
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from astro_mine.core.units import PlanetaryCRS, require_crs
from astro_mine.core.world import RegolithParams
from astro_mine.worlds.crs import to_rasterio_crs
from astro_mine.worlds.regolith._fields import (
    DEFAULT_LUNAR_PRIOR,
    PARAM_NAMES,
    ParamPrior,
    RegolithPrior,
    regolith_hash,
    regolith_layers,
)
from astro_mine.worlds.terrain import TerrainModel, TerrainProduct
from astro_mine.worlds.terrain._ingest import RasterLayer, write_cog

__all__ = [
    "BUNDLE_SCHEMA",
    "DEFAULT_LUNAR_PRIOR",
    "PARAM_NAMES",
    "PARAM_UNITS",
    "ParamPrior",
    "RegolithField",
    "RegolithPrior",
    "RegolithProduct",
    "build_regolith_field",
]

BUNDLE_SCHEMA = "astro-mine-worlds/regolith/v0.1"

#: SI unit label per parameter (shared by a parameter's mean and its uncertainty layer).
PARAM_UNITS: dict[str, str] = {
    "bulk_density": "kg/m^3",
    "cohesion": "Pa",
    "friction_angle": "degree",
    "bearing_capacity": "Pa",
    "thermal_inertia": "tiu",
}

#: Map a parameter name to its Core ``RegolithParams`` field (Worlds owns the data, Core the type).
_PARAM_TO_FIELD: dict[str, str] = {
    "bulk_density": "bulk_density_kg_m3",
    "cohesion": "cohesion_pa",
    "friction_angle": "friction_angle_deg",
    "bearing_capacity": "bearing_capacity_pa",
    "thermal_inertia": "thermal_inertia_tiu",
}

_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class RegolithProduct:
    """A content-addressed regolith parameter field on disk, aligned to a terrain grid."""

    path: Path
    crs: PlanetaryCRS
    width: int
    height: int
    resolution_m: float
    transform: tuple[float, float, float, float, float, float]
    regolith_hash: str
    layers: dict[str, Path]
    manifest: dict[str, Any]


def _worlds_version() -> str:
    try:
        return importlib.metadata.version("astro-mine-platform")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed in dev/CI
        return "0+unknown"


def _read_band(path: Path) -> np.ndarray[Any, Any]:
    with rasterio.open(path) as ds:
        return np.asarray(ds.read(1))


def build_regolith_field(
    terrain: TerrainModel | TerrainProduct | str | Path,
    out_dir: str | Path,
    *,
    prior: RegolithPrior = DEFAULT_LUNAR_PRIOR,
) -> RegolithProduct:
    """Build a regolith parameter field on ``terrain``'s grid and write it as COG layers.

    Reads the void mask (and slope, only if ``prior`` modulates means by slope) from the
    terrain product, applies ``prior`` to produce mean + companion-uncertainty layers on the
    same CRS/grid/transform, writes one COG per layer plus a deterministic manifest, and
    returns the :class:`RegolithProduct`.
    """
    model = terrain if isinstance(terrain, TerrainModel) else TerrainModel.open(terrain)
    require_crs(model.crs)
    grid = model.manifest["grid"]
    transform: tuple[float, float, float, float, float, float] = tuple(grid["transform"])

    void_mask = _read_band(model.path / "void_mask.tif") > 0
    slope_deg = (
        _read_band(model.path / "slope.tif").astype(np.float64) if prior.uses_slope() else None
    )
    layers_arr = regolith_layers(prior, void_mask, slope_deg)
    height, width = void_mask.shape

    meta: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "terrain_hash": str(model.manifest.get("terrain_hash", "")),
        "crs": model.crs.model_dump(mode="json"),
        "grid": {
            "width": int(width),
            "height": int(height),
            "resolution_m": grid["resolution_m"],
            "transform": list(transform),
        },
        "layers": {
            name: {"units": PARAM_UNITS[name.removesuffix("_uncertainty")]} for name in layers_arr
        },
        "prior": {
            name: {
                "mean": p.mean,
                "uncertainty": p.uncertainty,
                "slope_sensitivity": p.slope_sensitivity,
            }
            for name, p in prior.items()
        }
        | {"void_uncertainty_factor": prior.void_uncertainty_factor},
        "toolchain": {"astro_mine_worlds": _worlds_version(), "numpy": np.__version__},
    }
    digest = regolith_hash(layers_arr, meta)
    meta = {**meta, "regolith_hash": digest}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst_crs = to_rasterio_crs(model.crs)
    layer_paths: dict[str, Path] = {}
    for name, arr in layers_arr.items():
        layer_path = out / f"{name}.tif"
        write_cog(layer_path, arr, transform, dst_crs, None)
        layer_paths[name] = layer_path
    (out / _MANIFEST_NAME).write_text(
        json.dumps(meta, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return RegolithProduct(
        path=out,
        crs=model.crs,
        width=int(width),
        height=int(height),
        resolution_m=float(grid["resolution_m"]),
        transform=transform,
        regolith_hash=digest,
        layers=layer_paths,
        manifest=meta,
    )


class RegolithField:
    """Read access to a regolith product: per-point parameter means and their uncertainty.

    ``params`` is the Sim-facing surface (Core ``RegolithParams``); the WORLDS-06 Env-API
    provider samples it into a ``SurfacePoint``. Out-of-bounds queries return all-``None``
    params (every Core field is optional).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest: dict[str, Any] = json.loads(
            (self.path / _MANIFEST_NAME).read_text(encoding="utf-8")
        )
        self.crs = PlanetaryCRS.model_validate(self.manifest["crs"])
        self._layers: dict[str, RasterLayer] = {}

    @classmethod
    def open(cls, product: RegolithProduct | str | Path) -> RegolithField:
        """Open a regolith product from a :class:`RegolithProduct` or its directory path."""
        path = product.path if isinstance(product, RegolithProduct) else product
        return cls(path)

    def _layer(self, stem: str) -> RasterLayer:
        """The named layer, materialized in memory once and reused — no per-query reopen (#48)."""
        layer = self._layers.get(stem)
        if layer is None:
            layer = RasterLayer.open(self.path / f"{stem}.tif")
            self._layers[stem] = layer
        return layer

    def _read(self, x: float, y: float, suffix: str) -> RegolithParams:
        values: dict[str, float | None] = {}
        for name in PARAM_NAMES:
            value, in_bounds = self._layer(f"{name}{suffix}").sample(x, y)
            values[name] = float(value) if in_bounds else None
        return RegolithParams(
            bulk_density_kg_m3=values["bulk_density"],
            cohesion_pa=values["cohesion"],
            friction_angle_deg=values["friction_angle"],
            bearing_capacity_pa=values["bearing_capacity"],
            thermal_inertia_tiu=values["thermal_inertia"],
        )

    def params(self, x: float, y: float) -> RegolithParams:
        """The regolith parameter **means** at world ``(x, y)`` (the Sim-facing contract)."""
        return self._read(x, y, "")

    def uncertainty(self, x: float, y: float) -> RegolithParams:
        """The per-parameter **1-sigma uncertainty** at world ``(x, y)``, in the same fields."""
        return self._read(x, y, "_uncertainty")
