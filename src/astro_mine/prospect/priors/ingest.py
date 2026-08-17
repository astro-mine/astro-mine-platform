# SPDX-License-Identifier: Apache-2.0
"""Real PDS raster-ingest for the water-ice prior-recipe (RM-P1-PROSPECT-12).

Reads the public conditioning rasters the water-ice prior conditions on — Diviner bolometric
temperature, LEND epithermal-neutron suppression, M³ surficial-hydration band depth, and the
LOLA + SPICE-derived permanently-shadowed-region (PSR) mask — reprojects each onto the Shackleton
prior grid (``prospect.md §6``: "ice priors conditioned on Diviner temperature and PSR geometry"),
and materializes a small, deterministic, **content-addressed conditioning bundle** the offline
recipe fits from.

Split of responsibilities (``LUNAR-TR-004``): GDAL/rasterio is a **build-time-only** dependency
(the ``[ingest]`` extra). The offline recipe (:mod:`astro_mine.prospect.priors.pds`) and the
publish ``from_bundle`` path read the materialized bundle with **numpy alone**, so the offline
local tier never pulls GDAL. The multi-GB PDS raster fetch is a one-time, documented, cached step
(``scripts/fetch_pds_conditioning.py``); the fitted conditioning bundle is what the local tier and
Hub consume.

The reprojection warps each source (Diviner/M³ are polar-stereographic, LEND is a coarse global
grid) onto the exact Shackleton prior grid — the same reprojection Worlds performs for its
conditioning layers (RM-P1-WORLDS-14), applied here to the prior's own CRS/grid so the layers
co-register cell-for-cell with the belief field. Sources may equally be Worlds' pre-ingested
conditioning COGs — the read path is identical.

Backlog: RM-P1-PROSPECT-12 — astro-mine-prospect#11
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from numpy.typing import NDArray

from astro_mine.core import schema_registry
from astro_mine.core.schemas import core_schema
from astro_mine.core.units import PlanetaryCRS, require_crs
from astro_mine.prospect.field.metadata import FieldGrid

__all__ = [
    "BUNDLE_SCHEMA",
    "CONDITIONING_MEMBER",
    "MANIFEST_MEMBER",
    "ConditioningLayer",
    "ConditioningLayerSet",
    "RasterInput",
    "bundle_content_hash",
    "ingest_conditioning",
    "materialize_conditioning_bundle",
    "validate_manifest_crs",
]

#: On-disk schema tag of a materialized conditioning bundle (SemVer-tagged; conventions.md §5).
BUNDLE_SCHEMA = "astro-mine-prospect/conditioning/v0.1"
#: The two members of a materialized conditioning bundle directory.
CONDITIONING_MEMBER = "conditioning.npz"
MANIFEST_MEMBER = "manifest.json"

F32 = np.float32

#: Core's canonical units schema, read from the installed Core package through its public accessor
#: — so the ``$id`` below tracks the pinned Core rev instead of a hand-copied string (RFC-0009 §1).
_UNITS_SCHEMA_ID: str = str(core_schema("astro_mine.core.units", "units.schema.json")["$id"])

# A tiny consumer schema whose sole job is to pin a manifest's raw ``crs`` block to Core's canonical
# ``PlanetaryCRS`` ``$def``. The cross-file ``$ref`` names Core's schema by its absolute ``$id`` —
# public, append-only API — and :func:`astro_mine.core.schema_registry` resolves it offline
# (RFC-0009 §1, §2), so a manifest written by a *non-Python* producer is checked against the one
# authority — an unschematized dict no longer reaches ``rasterio`` (conventions.md §5).
_MANIFEST_CRS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://schemas.astro-mine.org/prospect/conditioning/manifest.crs.schema.json",
    "$ref": f"{_UNITS_SCHEMA_ID}#/$defs/PlanetaryCRS",
}


@lru_cache(maxsize=1)
def _manifest_crs_validator() -> Draft202012Validator:
    registry = schema_registry(_MANIFEST_CRS_SCHEMA)
    return Draft202012Validator(_MANIFEST_CRS_SCHEMA, registry=registry)


def validate_manifest_crs(raw: object) -> PlanetaryCRS:
    """Validate a conditioning manifest's raw ``crs`` block into a Core :class:`PlanetaryCRS`.

    A conditioning manifest's ``crs`` arrives as an untrusted dict — possibly written by a
    non-Python producer — so it is checked in two stages before any spatial machinery sees it
    (RFC-0007):

    1. **Shape** — validated against Core's canonical ``units.schema.json`` ``PlanetaryCRS``
       ``$def`` (``RM-P1-CORE-06``). A malformed block fails here with a JSON-Schema error naming
       the offending field, not an opaque ``rasterio`` error downstream.
    2. **Guard** — coerced through :func:`~astro_mine.core.units.require_crs` (``RM-P1-CORE-08``),
       which pins the Pydantic types and applies the fail-loud rules the schema cannot express —
       notably rule 6: an Earth datum/projection marker (``WGS84`` / ``EPSG:4326``) on a
       non-``EARTH`` body is a defaulting bug and is rejected (``conventions.md §5``).

    Returns the validated :class:`PlanetaryCRS`. Raises :class:`ValueError` on a shape error and
    :class:`~astro_mine.core.units.UnitsValidationError` on a guard violation.
    """
    errors = sorted(_manifest_crs_validator().iter_errors(raw), key=lambda e: e.json_path)
    if errors:
        first = errors[0]
        raise ValueError(
            f"conditioning manifest 'crs' fails Core units.schema.json at {first.json_path}: "
            f"{first.message} (no implicit/malformed CRS; conventions.md §5)"
        )
    return require_crs(raw)


@dataclass(frozen=True)
class RasterInput:
    """One public source raster to ingest, plus the per-product provenance it carries.

    ``role`` is the semantic tag the prior recipe keys on (``psr`` / ``measured_temperature`` /
    ``neutron_suppression`` / ``band_depth``). ``scale``/``offset``/``nodata`` convert the raw
    band to its physical value (``value = raw * scale + offset``, with ``raw == nodata`` → NaN);
    identity for an already-physical raster. ``resampling`` is the GDAL/rasterio resampling name
    used when warping onto the prior grid (``"average"`` turns a binary PSR mask into a soft
    shadow fraction; ``"bilinear"`` for continuous physical fields).

    ``citation`` is the ``short_name`` of the :class:`~astro_mine.prospect.priors.provenance.\
DatasetCitation` in :mod:`~astro_mine.prospect.priors.catalog` this source fills the
    ``source_hash`` of (e.g. ``"Diviner"``); the recipe stamps the real content hash there.
    """

    path: Path
    role: str
    units: str
    citation: str
    scale: float = 1.0
    offset: float = 0.0
    nodata: float | None = None
    resampling: str = "bilinear"


@dataclass(frozen=True)
class ConditioningLayer:
    """One reprojected conditioning layer on the prior grid, with its per-product provenance."""

    name: str
    role: str
    units: str
    citation: str
    source_hash: str
    values: NDArray[np.float32]


@dataclass(frozen=True)
class ConditioningLayerSet:
    """The conditioning layers ingested for a prior, co-registered on the Shackleton prior grid."""

    grid: FieldGrid
    crs: PlanetaryCRS
    layers: dict[str, ConditioningLayer]

    def array(self, name: str) -> NDArray[np.float32]:
        """The ``(n_rows, n_cols)`` values of layer ``name`` (``KeyError`` if absent)."""
        return self.layers[name].values


def _prospect_version() -> str:
    try:
        return importlib.metadata.version("astro-mine-platform")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed in dev/CI
        return "0+unknown"


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _grid_affine(grid: FieldGrid) -> Any:
    """The rasterio affine transform of the (north-up) prior grid, in the CRS's projected metres."""
    import rasterio

    px_w = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    px_h = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    return rasterio.transform.Affine(px_w, 0.0, grid.min_x_m, 0.0, -px_h, grid.max_y_m)


def reproject_onto_prior_grid(
    source: RasterInput, grid: FieldGrid, crs: PlanetaryCRS
) -> NDArray[np.float32]:
    """Warp one source raster onto the exact ``(crs, grid)`` prior grid; uncovered cells are NaN.

    Reads the raw band, applies ``value = raw*scale + offset`` (masking ``nodata`` → NaN), then
    reprojects that physical array from the source's own CRS onto the prior grid with the source's
    resampling. The source MUST carry an explicit planetary CRS (no implicit Earth/WGS84).
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    # The destination/prior CRS must be an explicit, valid planetary CRS. Delegate that verdict to
    # Core's guard (RM-P1-CORE-08) — inheriting rule 6 (no implicit Earth/WGS84) for free — rather
    # than trusting the caller (RFC-0007 Design §3; conventions.md §5).
    crs = require_crs(crs)
    dst_crs = rasterio.crs.CRS.from_proj4(crs.projection)
    resampling = Resampling[source.resampling]
    with rasterio.open(str(source.path)) as src:
        if src.crs is None:
            # A source raster that declares no CRS is precisely the implicit-Earth defaulting bug
            # the guard forbids. Route the fail-loud verdict through require_crs so this ingest
            # boundary inherits Core's rule, not a re-implemented message inline (RM-P1-CORE-08).
            require_crs(None)
        raw = src.read(1).astype(np.float64)
        physical = raw * source.scale + source.offset
        if source.nodata is not None:
            physical = np.where(raw == source.nodata, np.nan, physical)
        physical = np.where(np.isfinite(physical), physical, np.nan).astype(F32)
        destination = np.full((grid.n_rows, grid.n_cols), np.nan, dtype=F32)
        reproject(
            source=physical,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=float("nan"),
            dst_transform=_grid_affine(grid),
            dst_crs=dst_crs,
            dst_nodata=float("nan"),
            resampling=resampling,
        )
    return destination


def ingest_conditioning(
    inputs: Mapping[str, RasterInput],
    *,
    grid: FieldGrid,
    crs: PlanetaryCRS,
) -> ConditioningLayerSet:
    """Reproject each named source raster onto the prior ``grid`` and record its content hash.

    ``inputs`` maps a layer name to its :class:`RasterInput`. Each is warped onto ``(crs, grid)``
    (co-registered cell-for-cell), hashed by its raw source bytes, and returned as a
    :class:`ConditioningLayer`. The result feeds :func:`materialize_conditioning_bundle`.
    """
    layers: dict[str, ConditioningLayer] = {}
    for name, source in inputs.items():
        values = reproject_onto_prior_grid(source, grid, crs)
        layers[name] = ConditioningLayer(
            name=name,
            role=source.role,
            units=source.units,
            citation=source.citation,
            source_hash=_hash_file(source.path),
            values=values,
        )
    return ConditioningLayerSet(grid=grid, crs=crs, layers=layers)


def bundle_content_hash(arrays: Mapping[str, NDArray[np.float32]]) -> str:
    """A deterministic SHA-256 over the co-registered layer arrays (sorted names, raw bytes).

    Numpy-only, so the offline recipe can recompute it to verify a materialized bundle before
    fitting (fail-closed, hub.md principle 7) without pulling GDAL.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays[name], dtype=F32).tobytes())
    return f"sha256:{digest.hexdigest()}"


def materialize_conditioning_bundle(layer_set: ConditioningLayerSet, out_dir: str | Path) -> Path:
    """Write ``layer_set`` as a content-addressed conditioning bundle (``.npz`` + ``manifest``).

    The offline recipe reads this bundle with numpy alone (no GDAL). The arrays are packed into a
    deterministic ``conditioning.npz`` and a ``manifest.json`` records the grid/CRS, each layer's
    per-product provenance (role, units, citation, ``source_hash``), the toolchain, and a
    ``content_hash`` over the arrays — so a Bench scenario can pin the conditioning inputs and the
    fit reproduces from cited public rasters (conventions.md §5, ``LUNAR-DR-004``).
    """
    import rasterio

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {name: layer.values for name, layer in layer_set.layers.items()}
    # Deterministic .npz: sorted keys, fixed float32 dtype (np.savez writes members in call order).
    # (np.savez's stub types **kwds against its `allow_pickle: bool` param — a false positive here.)
    with (out / CONDITIONING_MEMBER).open("wb") as fh:
        np.savez(fh, **{name: arrays[name] for name in sorted(arrays)})  # type: ignore[arg-type]
    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "crs": layer_set.crs.model_dump(mode="json"),
        "grid": layer_set.grid.model_dump(mode="json"),
        "layers": {
            name: {
                "role": layer.role,
                "units": layer.units,
                "citation": layer.citation,
                "source_hash": layer.source_hash,
            }
            for name, layer in sorted(layer_set.layers.items())
        },
        "content_hash": bundle_content_hash(arrays),
        "toolchain": {
            "astro_mine_prospect": _prospect_version(),
            "gdal": rasterio.__gdal_version__,
            "numpy": np.__version__,
        },
    }
    (out / MANIFEST_MEMBER).write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out
