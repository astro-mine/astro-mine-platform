"""Illumination + PSR detection — horizon maps and permanently-shadowed-region masks.

Precomputed per-azimuth horizon maps (O(1) per-epoch Sun visibility) and PSR masks over a
defined epoch window — the comms/sun-denied core of the anchor scenario (scenario §5;
``LUNAR-FR-001``). Geometry is ground truth: the terrain skyline (the ``_horizon`` kernels)
plus SPICE Sun geometry (the ``spice`` backbone) decide visibility; nothing here invents an
RF or thermal layer (those are [Link] / RM-P0-WORLDS-04).

The horizon map is computed in **world (grid) azimuth**; the SPICE Sun azimuth is
topocentric and is converted into that frame via the south-polar-stereographic grid
convergence. That convergence is exact only for the lunar polar projection used here, so
:class:`IlluminationModel` fails loudly on any other CRS; the rigorous per-cell topocentric
horizon that lifts the restriction is RM-P1-WORLDS-12 (issue #11).

**PSR epoch semantics** (scenario open question §16): a PSR mask is "never sunlit over a
sampled epoch window", and *which* window defines "permanent" is explicit on the result via
:class:`PsrEpochSemantics` — ``DIURNAL`` (one lunar day), ``SEASONAL`` (a lunar-obliquity
season; the physically meaningful timescale for true permanence), or ``MISSION`` (the
caller's mission window). The MVP samples whatever window it is given and labels it; it does
not assert permanence beyond that window.

Backlog: RM-P0-WORLDS-03 — https://github.com/astro-mine/astro-mine-worlds/issues/3
"""

from __future__ import annotations

import enum
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
import rasterio.crs
import rasterio.transform
import rasterio.warp
from numpy.typing import NDArray

from astro_mine.core.units import MOON, MOON_BODY_FIXED, Epoch, EpochWindow, ReferenceFrame
from astro_mine.spice import DEFAULT_ABCORR, Site, body_position, epoch_range, sun_geometry
from astro_mine.worlds.crs import MOON_RADIUS_M, lunar_geographic_proj4
from astro_mine.worlds.fields import FieldArray, FieldStore, read_field_zarr, write_field_zarr
from astro_mine.worlds.illumination._backend import DEFAULT_BACKEND
from astro_mine.worlds.illumination._horizon import (
    azimuth_bin,
    horizon_field,
    horizon_hash,
    psr_mask_hash,
    sun_visibility_raster,
    topocentric_to_world_azimuth,
)
from astro_mine.worlds.illumination._topocentric import (
    body_fixed_positions,
    horizon_frame_delta,
    topocentric_elevation_azimuth_grid,
    topocentric_horizon_field,
    topocentric_horizon_hash,
)
from astro_mine.worlds.illumination._validation import (
    VALIDATION_SCHEMA,
    PsrReference,
    PsrValidation,
    psr_statistics,
    validate_psr,
)
from astro_mine.worlds.terrain import TerrainModel, TerrainProduct
from astro_mine.worlds.terrain._layers import fill_voids

if TYPE_CHECKING:
    # Runtime import would cycle (spec._bundle imports this module); from_spec only needs the
    # instance's `.layers`, which is duck-typed, so the class is a type-checking-only reference.
    from astro_mine.worlds.spec._model import WorldSpec

__all__ = [
    "BUNDLE_SCHEMA",
    "DEFAULT_BACKEND",
    "FIELD_MODEL_ENTRY_POINT_GROUP",
    "FIELD_MODEL_INTERFACE",
    "FIELD_MODEL_INTERFACE_VERSION",
    "HORIZON_ARRAY",
    "HORIZON_DIMS",
    "HORIZON_STORE_NAME",
    "RAYCAST_CPU_BACKEND",
    "RAYCAST_GPU_BACKEND",
    "SURROGATE_BACKEND",
    "VALIDATION_SCHEMA",
    "HorizonFrame",
    "IlluminationError",
    "IlluminationModel",
    "PsrEpochSemantics",
    "PsrReference",
    "PsrResult",
    "PsrValidation",
    "RayCastGpuIlluminationModel",
    "RayCastIlluminationModel",
    "SunVisibilityModel",
    "SurrogateAdmissionError",
    "SurrogateIlluminationModel",
    "available_backends",
    "build_illumination_field_manifest",
    "build_illumination_model",
    "field_model_kind_for_domain",
    "illumination_params_from_manifest",
    "known_backends",
    "psr_statistics",
    "validate_psr",
]

BUNDLE_SCHEMA = "astro-mine-worlds/illumination/v0.1"

F32 = np.float32

#: NAIF name for the Sun — the body whose per-cell topocentric geometry lights the surface.
_SUN = "SUN"

#: The Zarr field store the ``(H, W, n_azimuth)`` horizon map is persisted to inside a world
#: bundle's ``illumination/`` directory (worlds.md §5: "Horizon maps / PSR masks — Zarr + COG").
#: Persisting it is what stops every :class:`IlluminationModel` construction from re-deriving the
#: full skyline in-process, and gives the §7 cloud precompute/serve tier an artifact to read.
HORIZON_STORE_NAME = "horizon.zarr"
#: The array name inside that store, and its axis names.
HORIZON_ARRAY = "horizon"
HORIZON_DIMS = ("y", "x", "azimuth")

#: The manifest ``params`` that determine the horizon **array** itself. ``abcorr`` (a Sun-geometry
#: correction) and ``backend`` (a Sun-visibility strategy) are recorded in the manifest — and so in
#: :attr:`IlluminationModel.illumination_hash` — but do not enter the skyline computation, so a
#: persisted map stays reusable across them. These are the fields a stored map is checked against.
_HORIZON_PARAMS = ("n_azimuth", "max_radius_m", "body_radius_m", "horizon_frame", "body")


class IlluminationError(Exception):
    """Raised on an unsupported CRS or an out-of-bounds query."""


class PsrEpochSemantics(enum.StrEnum):
    """What "permanent" means for a PSR mask — the window over which shadow was sampled."""

    DIURNAL = "diurnal"  # one lunar day (synodic month)
    SEASONAL = "seasonal"  # a lunar-obliquity season — the meaningful "permanent" timescale
    MISSION = "mission"  # the caller-supplied mission epoch window


class HorizonFrame(enum.StrEnum):
    """The frame the per-cell horizon map is computed in (the RM-P1-WORLDS-12 fidelity dial).

    ``GRID`` is the RM-P0-WORLDS-03 default: per-azimuth skyline in the projected grid frame,
    reconciled with the SPICE topocentric Sun azimuth via the south-polar-stereographic
    grid-convergence correction (exact only for that spherical polar CRS). ``TOPOCENTRIC`` is
    the rigorous upgrade: the skyline is computed directly in each cell's local topocentric
    frame from true 3-D body-fixed geometry, so the SPICE Sun azimuth is used with **no**
    projection correction and the model works on any CRS or body (e.g. the Mars pack).
    """

    GRID = "grid"
    TOPOCENTRIC = "topocentric"


@dataclass(frozen=True)
class PsrResult:
    """A permanently-shadowed-region mask over a sampled epoch window, with provenance."""

    mask: NDArray[np.bool_]  # True where the Sun is never visible over the window
    ever_lit_fraction: float  # fraction of cells lit at least once
    void_mask: NDArray[np.bool_]  # DEM voids — PSR-ness here is not trustworthy
    window: EpochWindow
    step_s: float
    n_epochs: int
    semantics: PsrEpochSemantics
    illumination_hash: str
    #: A snapshot of the producing :meth:`IlluminationModel.to_manifest` (frame, radius, abcorr,
    #: n_azimuth, body, backend, terrain hash, grid, toolchain), captured at :meth:`psr_mask` time.
    #: The bundle build no longer holds the model, so this lets it write a self-describing
    #: illumination manifest and check the mask against the WorldSpec declaration (issue #36).
    #: Defaults empty so a directly-constructed result (a test double) is still valid.
    illumination_manifest: dict[str, Any] = field(default_factory=dict)
    #: The producing model's ``(H, W, n_azimuth)`` horizon map — carried (by reference, not copied)
    #: so the bundle build can persist it to Zarr without holding the model itself (issue #39).
    #: ``None`` for a directly-constructed result, in which case the bundle simply ships no horizon
    #: layer and a consumer recomputes it in-process.
    horizon: NDArray[np.float32] | None = None

    @property
    def psr_hash(self) -> str:
        """The illumination component's content hash for a world bundle: the terrain horizon
        (:attr:`illumination_hash`) folded with the **SPICE-derived** PSR + void masks and the
        epoch-window sampling. Folding the mask in means a kernel/ephemeris change that alters the
        shipped PSR mask changes the ``world_hash`` — the horizon alone did not pin the PSR-ness the
        anchor scores (RM-P1-WORLDS-15). ``ever_lit_fraction`` is a derived summary of ``mask``, so
        it need not be hashed separately."""
        meta = {
            "illumination_hash": self.illumination_hash,
            "semantics": self.semantics.value,
            "step_s": self.step_s,
            "n_epochs": self.n_epochs,
            "window": {
                "start_tdb_s": self.window.start.tdb_seconds,
                "end_tdb_s": self.window.end.tdb_seconds,
            },
        }
        return psr_mask_hash(self.mask, self.void_mask, meta)

    def to_manifest(self) -> dict[str, Any]:
        """The illumination provenance for ``illumination/manifest.json`` in a world bundle.

        The producing model's manifest (frame / max_radius_m / abcorr / n_azimuth / toolchain) plus
        the PSR sampling that turned per-epoch Sun visibility into a permanent-shadow mask — so a
        pulled bundle records *how* its PSR mask was computed, not just the mask bytes (worlds.md §5
        provenance; issue #36). ``psr_hash`` is included so the manifest ties to the bundle's
        ``component_hashes["illumination"]``.
        """
        doc = dict(self.illumination_manifest)
        doc["psr"] = {
            "psr_hash": self.psr_hash,
            "illumination_hash": self.illumination_hash,
            "semantics": self.semantics.value,
            "step_s": self.step_s,
            "n_epochs": self.n_epochs,
            "ever_lit_fraction": self.ever_lit_fraction,
            "window": {
                "start_tdb_s": self.window.start.tdb_seconds,
                "end_tdb_s": self.window.end.tdb_seconds,
            },
        }
        return doc


def _worlds_version() -> str:
    try:
        return importlib.metadata.version("astro-mine-platform")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed in dev/CI
        return "0+unknown"


def _require_polar_stereographic(projection: str | None) -> None:
    """Fail loudly unless ``projection`` is the lunar south-polar stereographic CRS.

    The topocentric->world azimuth conversion assumes ``lon_0 = 0`` south-polar
    stereographic (grid convergence == longitude). Any other CRS needs the rigorous
    per-cell topocentric horizon of RM-P1-WORLDS-12, so it is rejected rather than
    silently mis-projected.
    """
    proj = projection or ""
    if not all(token in proj for token in ("+proj=stere", "+lat_0=-90", "+lon_0=0")):
        raise IlluminationError(
            "illumination requires the lunar south-polar stereographic CRS "
            "(+proj=stere +lat_0=-90 +lon_0=0); the per-cell topocentric horizon for "
            f"other CRSs is RM-P1-WORLDS-12. Got projection: {projection!r}"
        )


@dataclass(frozen=True)
class _StoredHorizon:
    """A horizon map read back from a Zarr store, with the provenance it was written with."""

    horizon: NDArray[np.float32]
    params: dict[str, Any]
    grid: dict[str, Any]
    terrain_hash: str


def _read_horizon_store(path: str | Path) -> _StoredHorizon | None:
    """Read a persisted horizon map, or ``None`` if the store is not there.

    The **load-path fallback**: a bundle built before horizon maps were persisted (or one that
    deliberately omits the layer) simply has no store, and the caller recomputes in-process —
    the pre-existing behaviour. A store that *is* present but unreadable/foreign is an error,
    not a miss (:func:`~astro_mine.worlds.fields.read_field_zarr` fails loudly).
    """
    store = Path(path)
    if not store.exists():
        return None
    arrays, attrs = read_field_zarr(store)
    try:
        horizon = arrays[HORIZON_ARRAY]
    except KeyError:
        raise IlluminationError(
            f"{store} has no {HORIZON_ARRAY!r} array; it is not a horizon-map store"
        ) from None
    manifest = attrs.get("manifest", {})
    return _StoredHorizon(
        horizon=np.ascontiguousarray(horizon, dtype=F32),
        params=dict(manifest.get("params", {})),
        grid=dict(manifest.get("grid", {})),
        terrain_hash=str(manifest.get("terrain_hash", "")),
    )


def illumination_params_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """The :class:`IlluminationModel` kwargs recorded in a bundle's ``illumination/manifest.json``.

    A world bundle records *how* its PSR mask was computed (issue #36); this reads those params
    back so :meth:`~astro_mine.worlds.provider.DemWorldProvider.from_bundle` rebuilds the **same**
    model rather than one with library defaults — without which a persisted horizon map could never
    be adopted (its parameters would not match) and the rebuilt ``illumination_hash`` would not be
    the bundle's. Unknown/absent keys are simply omitted, so an old manifest degrades to defaults.
    """
    params = manifest.get("params", {})
    kwargs: dict[str, Any] = {}
    for key in ("n_azimuth", "max_radius_m", "body_radius_m", "abcorr", "body"):
        if params.get(key) is not None:
            kwargs[key] = params[key]
    if params.get("horizon_frame") is not None:
        kwargs["horizon_frame"] = HorizonFrame(params["horizon_frame"])
    return kwargs


class IlluminationModel:
    """Horizon-map-backed Sun visibility and PSR detection over a terrain product.

    Builds the per-azimuth horizon map once at construction; thereafter
    :meth:`sun_visible` is an O(1) lookup, :meth:`illuminated_mask` an O(cells) raster, and
    :meth:`psr_mask` an OR across a sampled epoch window. The Core Environment-API provider
    that exposes these through the world contract is RM-P0-WORLDS-06.
    """

    def __init__(
        self,
        terrain: TerrainModel | TerrainProduct | str | Path,
        *,
        n_azimuth: int = 180,
        max_radius_m: float | None = None,
        body_radius_m: float = MOON_RADIUS_M,
        abcorr: str = DEFAULT_ABCORR,
        horizon_frame: HorizonFrame = HorizonFrame.GRID,
        body: str = MOON,
        body_fixed_frame: ReferenceFrame = MOON_BODY_FIXED,
        backend: str = DEFAULT_BACKEND,
        horizon_store: str | Path | None = None,
    ) -> None:
        model = terrain if isinstance(terrain, TerrainModel) else TerrainModel.open(terrain)
        # The grid-convergence conversion is projection-specific, so the GRID frame keeps its
        # south-polar-stereographic guard; the TOPOCENTRIC frame is CRS-agnostic by design
        # (true 3-D geometry), so it lifts that restriction — the point of RM-P1-WORLDS-12.
        if horizon_frame is HorizonFrame.GRID:
            _require_polar_stereographic(model.crs.projection)
        self.path = model.path
        self.crs = model.crs
        self.n_azimuth = n_azimuth
        self.body_radius_m = body_radius_m
        self.abcorr = abcorr
        self.horizon_frame = horizon_frame
        self.body = body
        self.body_fixed_frame = body_fixed_frame
        #: The active Sun-visibility backend (RM-P1-WORLDS-10). The horizon-map default is
        #: :data:`DEFAULT_BACKEND`; a ray-cast / surrogate subclass records its own selector so a
        #: backend swap moves :attr:`illumination_hash` (and thus the world hash) honestly, while
        #: the default stays out of the manifest — existing world hashes are unperturbed.
        self.backend = backend
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.terrain_hash = str(manifest.get("terrain_hash", ""))

        with rasterio.open(self.path / "elevation.tif") as ds:
            elevation = ds.read(1).astype(np.float64)
            self.transform = ds.transform
            self._src_crs = ds.crs
        self.height, self.width = elevation.shape
        self.void_mask = np.isnan(elevation)
        filled = fill_voids(elevation.astype(F32), self.void_mask).astype(np.float64)
        #: The void-filled DEM (float64) the fine ray-cast backends (RM-P1-WORLDS-10) march over;
        #: kept once here so a ray-cast subclass reuses it rather than re-reading the raster.
        self._filled_elevation = filled

        a, e = float(self.transform.a), float(self.transform.e)
        if max_radius_m is None:
            max_radius_m = min(self.height * abs(e), self.width * abs(a))
        self.max_radius_m = max_radius_m

        self._geo_crs = rasterio.crs.CRS.from_proj4(
            lunar_geographic_proj4(float(self.crs.reference_radius_m))
        )
        #: Where the horizon map came from — a persisted Zarr store, or an in-process rebuild.
        #: Recorded on the model (and in the bundle manifest) so a consumer can tell which it got.
        self.horizon_source = "recomputed"
        stored = _read_horizon_store(horizon_store) if horizon_store is not None else None
        if horizon_frame is HorizonFrame.TOPOCENTRIC:
            self._positions = self._cell_positions(filled)
            if stored is not None:
                self.horizon = self._adopt_horizon(stored)
            else:
                self.horizon = topocentric_horizon_field(
                    self._positions,
                    pixel_size_m=(a, e),
                    n_azimuth=n_azimuth,
                    max_radius_m=max_radius_m,
                )
            self.illumination_hash = topocentric_horizon_hash(self.horizon, self.to_manifest())
        else:
            if stored is not None:
                self.horizon = self._adopt_horizon(stored)
            else:
                self.horizon = horizon_field(
                    filled,
                    pixel_size_m=(a, e),
                    n_azimuth=n_azimuth,
                    max_radius_m=max_radius_m,
                    body_radius_m=body_radius_m,
                )
            self.illumination_hash = horizon_hash(self.horizon, self.to_manifest())

    def _adopt_horizon(self, stored: _StoredHorizon) -> NDArray[np.float32]:
        """Validate a persisted horizon map against this model and adopt it (no recompute).

        A stored map is only reusable if it was built over **this** terrain (``terrain_hash``), on
        **this** grid, with the same skyline-determining parameters (:data:`_HORIZON_PARAMS`) — a
        stale or foreign map silently substituted for the real skyline would corrupt every PSR and
        line-of-sight answer downstream, which worlds.md §9 calls out as safety-relevant. So the
        check **fails loudly** rather than falling back to a recompute: an explicitly-pointed-at
        store that does not match is a bug in the caller, not a cache miss. (A bundle with *no*
        store at all is the cache miss, and is handled by not calling this at all.)
        """
        manifest = self.to_manifest()
        mismatches = []
        if stored.terrain_hash != self.terrain_hash:
            mismatches.append(
                f"terrain_hash: store {stored.terrain_hash!r}, model {self.terrain_hash!r}"
            )
        if stored.grid != manifest["grid"]:
            mismatches.append(f"grid: store {stored.grid!r}, model {manifest['grid']!r}")
        for name in _HORIZON_PARAMS:
            want = manifest["params"].get(name)
            got = stored.params.get(name)
            if got != want:
                mismatches.append(f"{name}: store {got!r}, model {want!r}")
        if mismatches:
            raise IlluminationError(
                "persisted horizon map does not match this illumination model: "
                + "; ".join(mismatches)
            )
        expected = (self.height, self.width, self.n_azimuth)
        if stored.horizon.shape != expected:
            raise IlluminationError(
                f"persisted horizon map has shape {stored.horizon.shape}, expected {expected}"
            )
        self.horizon_source = "stored"
        return stored.horizon

    def write_horizon_zarr(self, path: str | Path) -> FieldStore:
        """Persist the ``(H, W, n_azimuth)`` horizon map to a chunked Zarr store at ``path``.

        The store is the world bundle's ``illumination/horizon.zarr`` member (worlds.md §5) — the
        artifact that lets a pulled bundle skip the O(n_azimuth · n_radius_steps) skyline rebuild,
        and the object the §7 cloud precompute tier writes once and the serve tier range-reads.
        Its content hash goes into ``world_hash`` (:mod:`~astro_mine.worlds.spec._bundle`).
        """
        return write_field_zarr(
            path,
            [
                FieldArray(
                    name=HORIZON_ARRAY,
                    values=self.horizon,
                    units="degree",
                    dims=HORIZON_DIMS,
                )
            ],
            attrs={
                "layer": "illumination/horizon",
                "illumination_hash": self.illumination_hash,
                "manifest": self.to_manifest(),
            },
        )

    @classmethod
    def from_product(cls, product: TerrainProduct | str | Path, **kwargs: Any) -> IlluminationModel:
        """Build an illumination model from a terrain product (or its directory path)."""
        return cls(product, **kwargs)

    @classmethod
    def from_spec(
        cls,
        spec: WorldSpec,
        terrain: TerrainModel | TerrainProduct | str | Path,
        **overrides: Any,
    ) -> IlluminationModel:
        """Construct a model from a :class:`WorldSpec`'s recorded illumination parameters.

        Reads ``spec.layers`` — ``illumination_horizon_frame`` / ``illumination_max_radius_m`` /
        ``illumination_abcorr`` / ``illumination_n_azimuth`` — so the *spec* is the single source of
        truth for the PSR mask: rebuilding from a spec reconstructs the identical model, hence the
        identical mask and ``world_hash`` (worlds.md §10 determinism gate; issue #36). A ``None``
        field selects the constructor default; ``overrides`` win over both (for callers threading
        e.g. ``body`` / ``body_fixed_frame`` for a non-lunar pack). Backend selection stays with
        :func:`build_illumination_model`, not here — this builds the horizon-map model.
        """
        layers = spec.layers
        kwargs: dict[str, Any] = {}
        if layers.illumination_n_azimuth is not None:
            kwargs["n_azimuth"] = layers.illumination_n_azimuth
        if layers.illumination_max_radius_m is not None:
            kwargs["max_radius_m"] = layers.illumination_max_radius_m
        if layers.illumination_abcorr is not None:
            kwargs["abcorr"] = layers.illumination_abcorr
        if layers.illumination_horizon_frame is not None:
            kwargs["horizon_frame"] = HorizonFrame(layers.illumination_horizon_frame)
        kwargs.update(overrides)
        return cls(terrain, **kwargs)

    def to_manifest(self) -> dict[str, Any]:
        """Provenance manifest: schema, source terrain hash, grid, params, and toolchain."""
        params: dict[str, Any] = {
            "n_azimuth": self.n_azimuth,
            "max_radius_m": self.max_radius_m,
            "body_radius_m": self.body_radius_m,
            "abcorr": self.abcorr,
            "horizon_frame": self.horizon_frame.value,
            "body": self.body,
        }
        # A non-default backend (RM-P1-WORLDS-10) is folded into the hash so a swap moves the world
        # hash; the horizon default is omitted so existing world hashes are byte-for-byte unchanged.
        if self.backend != DEFAULT_BACKEND:
            params["backend"] = self.backend
        return {
            "schema": BUNDLE_SCHEMA,
            "terrain_hash": self.terrain_hash,
            "grid": {"width": self.width, "height": self.height},
            "params": params,
            "toolchain": {"astro_mine_worlds": _worlds_version(), "numpy": np.__version__},
        }

    def _lonlat(self, x: float, y: float) -> tuple[float, float]:
        """Body-fixed (longitude, latitude) in degrees for a projected ``(x, y)``."""
        xs, ys = rasterio.warp.transform(self._src_crs, self._geo_crs, [x], [y])
        return float(xs[0]), float(ys[0])

    def _site(self, lat_deg: float, lon_deg: float) -> Site:
        """A body-fixed :class:`~astro_mine.spice.Site` on the body sphere at ``(lat, lon)``.

        Generalises ``Site.lunar_from_latlon`` to the model's ``body`` / ``body_fixed_frame``
        / ``body_radius_m`` — identical to the lunar helper for the Moon defaults, and the
        seam that lets the topocentric frame serve a non-lunar body (the Mars pack).
        """
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        r = self.body_radius_m
        position = (
            r * np.cos(lat) * np.cos(lon),
            r * np.cos(lat) * np.sin(lon),
            r * np.sin(lat),
        )
        return Site(body=self.body, position_m=position, frame=self.body_fixed_frame)

    def _cell_positions(self, elevation: NDArray[np.float64]) -> NDArray[np.float64]:
        """Body-fixed Cartesian positions (H, W, 3) for every terrain cell centre."""
        rows, cols = np.mgrid[0 : self.height, 0 : self.width]
        xs, ys = rasterio.transform.xy(self.transform, rows.ravel().tolist(), cols.ravel().tolist())
        lon, lat = rasterio.warp.transform(self._src_crs, self._geo_crs, list(xs), list(ys))
        lon_grid = np.asarray(lon, dtype=np.float64).reshape(self.height, self.width)
        lat_grid = np.asarray(lat, dtype=np.float64).reshape(self.height, self.width)
        return body_fixed_positions(lon_grid, lat_grid, elevation, self.body_radius_m)

    def _sun(self, x: float, y: float, epoch: Epoch) -> tuple[float, float]:
        """Sun (elevation_deg, azimuth_deg) at projected ``(x, y)`` and ``epoch``.

        The azimuth is the SPICE **topocentric** azimuth for the topocentric frame (used
        directly against the topocentric horizon) or the grid-convergence-corrected **world**
        azimuth for the grid frame (RM-P0-WORLDS-03 behaviour).
        """
        lon, lat = self._lonlat(x, y)
        geom = sun_geometry(self._site(lat, lon), epoch, abcorr=self.abcorr)
        if self.horizon_frame is HorizonFrame.TOPOCENTRIC:
            return geom.elevation_deg, geom.azimuth_deg
        return geom.elevation_deg, topocentric_to_world_azimuth(geom.azimuth_deg, lon)

    def sun_visible(self, x: float, y: float, epoch: Epoch) -> bool:
        """Is the Sun above the terrain horizon at projected ``(x, y)``, ``epoch``? (O(1))."""
        return self.illumination_at(x, y, epoch)[0]

    def illumination_at(self, x: float, y: float, epoch: Epoch) -> tuple[bool, float]:
        """``(sun_visible, sun_elevation_deg)`` at a cell/epoch — one geometry evaluation.

        The per-cell insolation primitive RM-P1-WORLDS-13's thermal forcing samples over a
        diurnal window: whether the Sun clears the local terrain horizon, and its elevation
        for the incidence-weighted flux. Raises :class:`IlluminationError` off the grid.
        """
        row, col = rasterio.transform.rowcol(self.transform, x, y)
        row, col = int(row), int(col)
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IlluminationError(f"({x}, {y}) is outside the terrain grid")
        elevation_deg, world_az = self._sun(x, y, epoch)
        lit = bool(elevation_deg > self.horizon[row, col, azimuth_bin(world_az, self.n_azimuth)])
        return lit, elevation_deg

    def illuminated_mask(self, epoch: Epoch) -> NDArray[np.bool_]:
        """Cells lit at ``epoch``.

        The GRID frame uses one region-centre Sun (the RM-P0-WORLDS-03 approximation); the
        TOPOCENTRIC frame evaluates the Sun's **per-cell** topocentric elevation/azimuth
        against each cell's topocentric horizon — the rigorous raster of RM-P1-WORLDS-12.
        """
        if self.horizon_frame is HorizonFrame.TOPOCENTRIC:
            return self._illuminated_mask_topocentric(epoch)
        cx, cy = rasterio.transform.xy(self.transform, self.height // 2, self.width // 2)
        elevation_deg, world_az = self._sun(float(cx), float(cy), epoch)
        return sun_visibility_raster(self.horizon, elevation_deg, world_az)

    def _illuminated_mask_topocentric(self, epoch: Epoch) -> NDArray[np.bool_]:
        """Per-cell lit mask: the Sun's per-cell topocentric el/az vs each cell's horizon.

        The Sun's body-fixed position is resolved once (it is ~1 AU away, so its *direction*
        is common to the whole region) and its true topocentric elevation/azimuth is computed
        per cell, then thresholded against the per-cell topocentric horizon bin.
        """
        sun = body_position(_SUN, self.body, epoch, frame=self.body_fixed_frame, abcorr=self.abcorr)
        sun_grid = np.broadcast_to(np.asarray(sun, dtype=np.float64), self._positions.shape)
        elevation, azimuth = topocentric_elevation_azimuth_grid(self._positions, sun_grid)
        width_deg = 360.0 / self.n_azimuth
        bins = np.clip(
            np.floor((np.nan_to_num(azimuth) % 360.0) / width_deg), 0, self.n_azimuth - 1
        ).astype(np.intp)
        horizon_at = np.take_along_axis(self.horizon, bins[..., None], axis=2)[..., 0]
        elevation = np.nan_to_num(elevation, nan=-np.inf)
        return elevation > horizon_at.astype(np.float64)

    def frame_delta(self) -> dict[str, float]:
        """Error budget of this TOPOCENTRIC model vs the P0 grid-convergence horizon.

        Builds the grid-frame horizon over the same terrain and reports the max/mean
        per-cell skyline-elevation discrepancy (and the high-latitude worst case, where the
        grid convergence is least accurate) via
        :func:`~astro_mine.worlds.illumination._topocentric.horizon_frame_delta`. Requires the
        TOPOCENTRIC frame on the polar-stereographic CRS the grid frame supports.
        """
        if self.horizon_frame is not HorizonFrame.TOPOCENTRIC:
            raise IlluminationError(
                "frame_delta compares a TOPOCENTRIC model against the grid frame"
            )
        _require_polar_stereographic(self.crs.projection)
        with rasterio.open(self.path / "elevation.tif") as ds:
            elevation = ds.read(1).astype(np.float64)
        filled = fill_voids(elevation.astype(F32), np.isnan(elevation)).astype(np.float64)
        a, e = float(self.transform.a), float(self.transform.e)
        grid = horizon_field(
            filled,
            pixel_size_m=(a, e),
            n_azimuth=self.n_azimuth,
            max_radius_m=self.max_radius_m,
            body_radius_m=self.body_radius_m,
        )
        return horizon_frame_delta(grid, self.horizon, self._latitude_grid())

    def _latitude_grid(self) -> NDArray[np.float64]:
        """Body-fixed latitude (deg) at every terrain cell centre (H, W)."""
        rows, cols = np.mgrid[0 : self.height, 0 : self.width]
        xs, ys = rasterio.transform.xy(self.transform, rows.ravel().tolist(), cols.ravel().tolist())
        _lon, lat = rasterio.warp.transform(self._src_crs, self._geo_crs, list(xs), list(ys))
        return np.asarray(lat, dtype=np.float64).reshape(self.height, self.width)

    def psr_mask(
        self,
        window: EpochWindow,
        step_s: float,
        *,
        semantics: PsrEpochSemantics = PsrEpochSemantics.MISSION,
    ) -> PsrResult:
        """Cells never lit across ``window`` sampled at ``step_s`` — the PSR mask.

        ``EpochWindow`` guarantees ``end > start``, so a sampled window always yields at
        least the start epoch; ``step_s <= 0`` is rejected by ``epoch_range``.
        """
        epochs = list(epoch_range(window, step_s))
        ever_lit = np.zeros((self.height, self.width), dtype=np.bool_)
        for epoch in epochs:
            ever_lit |= self.illuminated_mask(epoch)
        return PsrResult(
            mask=~ever_lit,
            ever_lit_fraction=float(ever_lit.mean()),
            void_mask=self.void_mask,
            window=window,
            step_s=step_s,
            n_epochs=len(epochs),
            semantics=semantics,
            illumination_hash=self.illumination_hash,
            illumination_manifest=self.to_manifest(),
            horizon=self.horizon,
        )


# The RM-P1-WORLDS-10 field-model backends (ray-cast CPU/GPU, learned surrogate) and the field-model
# factory/manifest are re-exported here so consumers select a backend through the illumination
# package surface. Imported at the bottom because each subclass extends ``IlluminationModel`` above;
# this is the standard "import the extenders after the base is defined" package idiom.
from astro_mine.worlds.illumination._backend import SunVisibilityModel  # noqa: E402
from astro_mine.worlds.illumination._raycast import (  # noqa: E402
    RAYCAST_CPU_BACKEND,
    RayCastIlluminationModel,
)
from astro_mine.worlds.illumination._raycast_gpu import (  # noqa: E402
    RAYCAST_GPU_BACKEND,
    RayCastGpuIlluminationModel,
)
from astro_mine.worlds.illumination._registry import (  # noqa: E402
    FIELD_MODEL_ENTRY_POINT_GROUP,
    FIELD_MODEL_INTERFACE,
    FIELD_MODEL_INTERFACE_VERSION,
    SURROGATE_BACKEND,
    available_backends,
    build_illumination_field_manifest,
    build_illumination_model,
    known_backends,
)
from astro_mine.worlds.illumination._surrogate import (  # noqa: E402
    SurrogateAdmissionError,
    SurrogateIlluminationModel,
    field_model_kind_for_domain,
)
