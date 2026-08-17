# SPDX-License-Identifier: Apache-2.0
"""Environment-API world provider + terrain occlusion / LOS service (RM-P0-WORLDS-06).

:class:`DemWorldProvider` implements the Core
:class:`~astro_mine.core.world.WorldProvider` contract: given a body-fixed position (and
epoch) it returns a :class:`~astro_mine.core.world.SurfacePoint` — ground geometry, surface
frame, local gravity, illumination/solar flux, surface temperature, and the regolith
terramechanics tuple — by composing the Phase-0 lunar field models (terrain
RM-P0-WORLDS-01, illumination/PSR RM-P0-WORLDS-03, regolith RM-P0-WORLDS-05). It also
exposes ``ray_intersect`` (DEM ray-cast) and the horizon-map ``line_of_sight``, the
occlusion machinery [Link] queries for inter-agent and Earth visibility (worlds.md §6).

**Frames.** Queried positions and rays are body-fixed Cartesian metres in :attr:`frame`
(:data:`~astro_mine.core.units.MOON_BODY_FIXED`); each :class:`SurfacePoint` is returned in
the local **topocentric** surface frame, where the terrain normal is grid-native and
point-mass gravity is ``(0, 0, -g)``. The body-fixed⇄map bridge uses the same SPICE-sphere
+ PROJ machinery the rest of Worlds uses (no implicit Earth/WGS84; conventions.md §5).

**Surface temperature** is sourced from an optional injected :class:`ThermalSource`; with
none, the provider falls back to a coarse radiative-equilibrium first-cut. The per-cell,
illumination-driven model that feeds this hook is RM-P1-WORLDS-13 (#16); the standalone
per-terrain-class curves are RM-P0-WORLDS-04.

Backlog: RM-P0-WORLDS-06 — astro-mine-worlds#6
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import rasterio.crs
import rasterio.transform
import rasterio.warp

from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.core.units import Epoch, ReferenceFrame
from astro_mine.core.world import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
    Vector,
)
from astro_mine.spice import sun_geometry
from astro_mine.worlds.crs import lunar_geographic_proj4, to_rasterio_crs
from astro_mine.worlds.illumination import (
    HORIZON_STORE_NAME,
    HorizonFrame,
    IlluminationModel,
    SunVisibilityModel,
    illumination_params_from_manifest,
)
from astro_mine.worlds.illumination._horizon import azimuth_bin, topocentric_to_world_azimuth
from astro_mine.worlds.provider._geometry import (
    add_scaled,
    equilibrium_temperature,
    norm,
    solar_flux,
    topocentric_elevation_azimuth,
    unit,
)
from astro_mine.worlds.regolith import RegolithField, RegolithProduct
from astro_mine.worlds.terrain import TerrainModel, TerrainProduct

if TYPE_CHECKING:
    from astro_mine.worlds.bodies import BodyPack

__all__ = ["ConditioningSource", "DemWorldProvider", "ThermalSource"]


class ConditioningSource(Protocol):
    """A pluggable set of co-registered conditioning field layers the provider can serve.

    Structurally satisfied by :class:`~astro_mine.worlds.ingest.ConditioningField`
    (RM-P1-WORLDS-14): given a projected map ``(x, y)`` on the world grid, return each layer's
    value (Diviner temperature, LEND WEH, M³ water). Injected into the provider so Prospect can
    reach the conditioning layers through the Environment-API surface, not a private channel.
    """

    @property
    def layers(self) -> tuple[str, ...]:
        """The conditioning-layer names available."""
        ...

    def sample(self, x: float, y: float) -> dict[str, float]:
        """Every conditioning layer's value at projected map ``(x, y)``."""
        ...


class ThermalSource(Protocol):
    """A pluggable surface-temperature model the provider consults when one is injected.

    Given a cell (projected map ``x``/``y``) and ``epoch`` — plus the provider's already
    computed solar flux for convenience — return the surface temperature (K).
    RM-P0-WORLDS-04 ships per-terrain-class diurnal curves; the per-cell, illumination-driven
    wiring that drives this hook is RM-P1-WORLDS-13 (#16). With no source injected the
    provider uses a coarse radiative-equilibrium first-cut instead.
    """

    def temperature_k(
        self, *, map_x: float, map_y: float, epoch: Epoch, solar_flux_w_m2: float
    ) -> float:
        """Surface temperature (K) at projected ``(map_x, map_y)`` and ``epoch``."""
        ...


#: Iterations for the ray-surface crossing bisection (sub-millimetre over the grid extent).
_BISECT_ITERS = 40


def _bundle_illumination_kwargs(bundle_dir: Path) -> dict[str, Any]:
    """The illumination kwargs to rebuild a bundle's model — its recorded params + stored horizon.

    A pulled bundle records *how* its illumination was computed in ``illumination/manifest.json``
    (issue #36) and, since issue #39, ships the derived horizon map as ``illumination/horizon.zarr``
    (worlds.md §5). Reading both back means ``from_bundle`` reconstructs the **bundle's** model —
    its ``illumination_hash``, not a library-default one — and adopts the persisted skyline instead
    of re-deriving it. Both are optional: a bundle without the manifest falls back to the
    constructor defaults, and one without the store recomputes the horizon in-process, which is the
    pre-existing behaviour.
    """
    manifest_path = bundle_dir / "illumination" / "manifest.json"
    kwargs: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kwargs.update(illumination_params_from_manifest(manifest))
    store = bundle_dir / "illumination" / HORIZON_STORE_NAME
    if store.exists():
        kwargs["horizon_store"] = store
    return kwargs


class DemWorldProvider:
    """DEM-backed implementation of the Core
    :class:`~astro_mine.core.world.WorldProvider` (worlds.md §6).

    Composes a :class:`~astro_mine.worlds.terrain.TerrainModel`, a
    :class:`~astro_mine.worlds.illumination.SunVisibilityModel`, and a
    :class:`~astro_mine.worlds.regolith.RegolithField` (which all share one CRS/grid) into
    the Core Environment-API world/terrain surface Sim consumes and the horizon
    line-of-sight Link queries. Determinism is contractual — the same ``(position, epoch)``
    yields an identical :class:`SurfacePoint`.

    ``illumination`` is typed to the **structural** backend contract, not to
    :class:`~astro_mine.worlds.illumination.IlluminationModel`: since issue #52 the
    ``astro_mine.field_models`` entry-point group is open, so the model here may come from a
    third-party field model that satisfies :class:`SunVisibilityModel` without subclassing anything
    of Worlds'. The provider reads only that protocol's surface.
    """

    def __init__(
        self,
        terrain: TerrainModel,
        illumination: SunVisibilityModel,
        regolith: RegolithField,
        *,
        frame: ReferenceFrame | None = None,
        thermal: ThermalSource | None = None,
        conditioning: ConditioningSource | None = None,
        body_pack: BodyPack | None = None,
    ) -> None:
        from astro_mine.worlds.bodies import MOON_PACK

        self.terrain = terrain
        self.illumination = illumination
        self.regolith = regolith
        self._pack = body_pack if body_pack is not None else MOON_PACK
        self._frame = frame if frame is not None else self._pack.body_fixed_frame
        self._thermal = thermal
        self._conditioning = conditioning
        self.crs = terrain.crs
        self._radius_m = float(self.crs.reference_radius_m)
        self._proj_crs = to_rasterio_crs(self.crs)
        self._geo_crs = rasterio.crs.CRS.from_proj4(lunar_geographic_proj4(self._radius_m))
        self._transform = illumination.transform
        self._height = illumination.height
        self._width = illumination.width
        self._resolution_m = float(terrain.manifest["grid"]["resolution_m"])
        # A generous cap so a ray from a sensible altitude always reaches the terrain.
        self._max_range_m = 2.0 * max(self._width, self._height) * self._resolution_m

    @classmethod
    def open(
        cls,
        terrain: TerrainModel | TerrainProduct | str | Path,
        regolith: RegolithField | RegolithProduct | str | Path,
        *,
        frame: ReferenceFrame | None = None,
        thermal: ThermalSource | None = None,
        conditioning: ConditioningSource | None = None,
        body_pack: BodyPack | None = None,
        illumination_backend: str | None = None,
        surrogate: Mapping[str, object] | None = None,
        **illumination_kwargs: object,
    ) -> DemWorldProvider:
        """Assemble a provider from a terrain product and a regolith product.

        Opens the terrain and regolith models and builds an
        :class:`~astro_mine.worlds.illumination.IlluminationModel` over the terrain
        (``illumination_kwargs`` forwarded — e.g. ``n_azimuth``, ``max_radius_m``,
        ``abcorr``, ``horizon_store``). An optional ``thermal`` source overrides the coarse
        temperature first-cut. For a non-lunar ``body_pack`` (RM-P1-WORLDS-11), the illumination
        model is built in the CRS-agnostic **topocentric** frame with the pack's body/frame/radius.

        ``illumination_backend`` (RM-P1-WORLDS-10; a ``WorldSpec``'s
        ``layers.illumination_backend``) selects the Sun-visibility field-model backend —
        ``None`` keeps the precomputed horizon default (the construction below is then byte-for-byte
        as before); ``"raycast_cpu"`` / ``"raycast_gpu"`` route through the fine ray-cast path, and
        ``"surrogate:<name>"`` needs the artifacts passed in ``surrogate``. The horizon map
        is always built, so ``line_of_sight`` is unaffected by the backend choice.
        """
        from astro_mine.worlds.bodies import MOON_PACK

        pack = body_pack if body_pack is not None else MOON_PACK
        tmodel = terrain if isinstance(terrain, TerrainModel) else TerrainModel.open(terrain)
        illum_kwargs = dict(illumination_kwargs)
        if pack.body != MOON_PACK.body:
            illum_kwargs.setdefault("body", pack.body)
            illum_kwargs.setdefault("body_fixed_frame", pack.body_fixed_frame)
            illum_kwargs.setdefault("body_radius_m", pack.reference_radius_m)
            illum_kwargs.setdefault("horizon_frame", HorizonFrame.TOPOCENTRIC)
        if illumination_backend:
            from astro_mine.worlds.illumination import build_illumination_model

            illum = build_illumination_model(
                tmodel, backend=illumination_backend, surrogate=surrogate, **illum_kwargs
            )
        else:
            illum = IlluminationModel(tmodel, **illum_kwargs)  # type: ignore[arg-type]
        rfield = regolith if isinstance(regolith, RegolithField) else RegolithField.open(regolith)
        return cls(
            tmodel,
            illum,
            rfield,
            frame=frame,
            thermal=thermal,
            conditioning=conditioning,
            body_pack=pack,
        )

    @classmethod
    def from_bundle(cls, manifest: PluginManifest, layers: Mapping[str, bytes]) -> DemWorldProvider:
        """Rebuild a live provider from a **pulled** world bundle — the ``world_provider`` factory.

        The ``astro_mine.providers`` → ``world_provider`` entry point Sim/Bench call after
        resolving a world **by content hash** through [Hub] (RM-P1-WORLDS-15). ``layers`` maps
        mediaType→bytes; the sole ``application/vnd.astro-mine.world.bundle.v1.tar`` layer is a
        deterministic tar of the full bundle directory (terrain/regolith COGs and their manifests,
        the PSR mask, the persisted horizon map, thermal curves, STAC, ``world.json``). It is
        unpacked into a private temp directory and the terrain + regolith products are re-opened
        directly. The temp directory is reclaimed when the returned provider is garbage-collected.

        A bundle that ships ``illumination/horizon.zarr`` has its skyline **adopted, not
        recomputed** (issue #39; the anchor got one in issue #46). That is the difference between a
        world load taking ~3 s and taking the better part of an hour: deriving a
        ``(1264, 1264, 120)`` skyline from the packaged DEM is a 192-million-entry ray-march, and
        every consumer used to pay it on every load. A bundle without the store still recomputes in
        process — the pre-existing behaviour, and the reason ``horizon_source`` is recorded on the
        model so a consumer can tell which it got.

        Consumers reach this **without importing** ``astro_mine.worlds`` (they load the entry
        point); the factory itself never imports ``astro_mine.sim`` or the Hub client.
        """
        import shutil
        import tempfile
        import weakref

        from astro_mine.worlds.spec._publish import BUNDLE_LAYER_MEDIA_TYPE, extract_bundle_tar

        if manifest.kind != PluginKind.WORLD_PROVIDER:
            raise ValueError(
                f"manifest kind {manifest.kind!r} is not a {PluginKind.WORLD_PROVIDER!r}"
            )
        try:
            payload = layers[BUNDLE_LAYER_MEDIA_TYPE]
        except KeyError:
            raise ValueError(
                f"pulled artifact has no {BUNDLE_LAYER_MEDIA_TYPE!r} layer; cannot rebuild a world"
            ) from None

        workdir = Path(tempfile.mkdtemp(prefix="astro-mine-world-"))
        try:
            extract_bundle_tar(payload, workdir)
            terrain_dir = workdir / "terrain"
            regolith_dir = workdir / "regolith"
            if (
                not (terrain_dir / "manifest.json").exists()
                or not (regolith_dir / "manifest.json").exists()
            ):
                raise ValueError(
                    "world bundle is missing a self-describing terrain/ or regolith/ product; "
                    "cannot rebuild a provider"
                )
            provider = cls.open(terrain_dir, regolith_dir, **_bundle_illumination_kwargs(workdir))
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        weakref.finalize(provider, shutil.rmtree, workdir, ignore_errors=True)
        return provider

    @property
    def frame(self) -> ReferenceFrame:
        """The body-fixed reference frame queried positions and rays resolve in."""
        return self._frame

    # --- WorldProvider contract ----------------------------------------------------

    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        """Query the surface at ``position`` (and ``epoch``) — geometry, gravity,
        illumination, temperature, and regolith in one :class:`SurfacePoint`.

        Out-of-grid positions (including the body centre) return a well-formed default
        point rather than raising, so the query surface is total.
        """
        mapped = self._to_map(position)
        if mapped is None:
            return self._unmapped_point(position)
        map_x, map_y, _radius = mapped
        sample = self.terrain.sample(map_x, map_y)
        if not sample.in_bounds:
            return self._unmapped_point(position)
        state, flux, temperature = self._solar_state(map_x, map_y, epoch)
        return SurfacePoint(
            frame=self._pack.surface_frame,
            elevation_m=float(sample.elevation_m),
            surface_normal=sample.normal,
            gravity=self._pack.gravity(position),
            illumination=Illumination(state=state, solar_flux_w_m2=flux),
            temperature_k=temperature,
            regolith=self.regolith.params(map_x, map_y),
        )

    def ray_intersect(self, origin: Vector, direction: Vector) -> Vector | None:
        """First terrain intersection of the ray from ``origin`` along ``direction``, or
        ``None`` if it misses — the DEM ray-cast occlusion primitive.

        Marches at the grid resolution detecting the first above→below-surface crossing,
        then bisects it. Detects only the first such crossing (a ray starting below the
        surface, or one that never descends, returns ``None``).
        """
        ray = unit(direction)
        if ray is None:
            return None
        previous_above: bool | None = None
        previous_t = 0.0
        t = 0.0
        while t <= self._max_range_m:
            point = add_scaled(origin, ray, t)
            surface_radius = self._surface_radius(point)
            if surface_radius is None:
                if previous_above is not None:
                    return None  # left the grid after entering it — no hit
                previous_t = t
                t += self._resolution_m
                continue
            above = norm(point) >= surface_radius
            if previous_above is True and not above:
                return self._bisect_hit(origin, ray, previous_t, t)
            previous_above = above
            previous_t = t
            t += self._resolution_m
        return None

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        """Whether ``target`` is visible from ``observer`` over the terrain horizon.

        Horizon-map-backed: the target's topocentric elevation from the observer is
        thresholded against the observer cell's precomputed per-azimuth horizon. Exact for
        the far-field geometry (surface→relay, surface→Earth); the rigorous near-field /
        per-cell topocentric treatment is RM-P1-WORLDS-12. ``observer`` outside the modelled
        grid, or a target coincident with it, yields ``False``.
        """
        mapped = self._to_map(observer)
        if mapped is None:
            return False
        map_x, map_y, _radius = mapped
        row, col = rasterio.transform.rowcol(self._transform, map_x, map_y)
        row, col = int(row), int(col)
        if not (0 <= row < self._height and 0 <= col < self._width):
            return False
        try:
            elevation_deg, azimuth_deg = topocentric_elevation_azimuth(observer, target)
        except ValueError:
            return False
        lon, _lat = self._map_to_lonlat(map_x, map_y)
        world_az = topocentric_to_world_azimuth(azimuth_deg, lon)
        horizon_elev = float(
            self.illumination.horizon[row, col, azimuth_bin(world_az, self.illumination.n_azimuth)]
        )
        return bool(elevation_deg > horizon_elev)

    def conditioning_at(self, position: Vector) -> dict[str, float]:
        """The conditioning-layer values (Diviner/LEND/M³) at a body-fixed ``position``.

        Serves the RM-P1-WORLDS-14 conditioning layers through the world-provider surface so
        [Prospect] can condition real priors on them (RM-P1-PROSPECT-12). Returns each layer's
        co-registered value; ``NaN`` per layer off the modelled grid. Raises
        :class:`LookupError` when no conditioning source was injected.
        """
        if self._conditioning is None:
            raise LookupError("no conditioning source injected; pass conditioning= to the provider")
        mapped = self._to_map(position)
        if mapped is None:
            return dict.fromkeys(self._conditioning.layers, float("nan"))
        map_x, map_y, _radius = mapped
        return self._conditioning.sample(map_x, map_y)

    # --- internals -----------------------------------------------------------------

    def _to_map(self, position: Vector) -> tuple[float, float, float] | None:
        """Body-fixed Cartesian ``position`` → projected map ``(x, y)`` and body radius.

        Returns ``None`` at the body centre, where latitude/longitude are undefined.
        """
        x, y, z = position
        r = math.sqrt(x * x + y * y + z * z)
        if r == 0.0:
            return None
        lat = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
        lon = math.degrees(math.atan2(y, x))
        xs, ys = rasterio.warp.transform(self._geo_crs, self._proj_crs, [lon], [lat])
        return float(xs[0]), float(ys[0]), r

    def _map_to_lonlat(self, map_x: float, map_y: float) -> tuple[float, float]:
        """Projected map ``(x, y)`` → body-fixed geographic ``(lon, lat)`` degrees."""
        lons, lats = rasterio.warp.transform(self._proj_crs, self._geo_crs, [map_x], [map_y])
        return float(lons[0]), float(lats[0])

    def _surface_radius(self, point: Vector) -> float | None:
        """Body radius (m) of the terrain surface directly under ``point``, or ``None`` if
        ``point`` is off the grid (or over a void where elevation is undefined)."""
        mapped = self._to_map(point)
        if mapped is None:
            return None
        map_x, map_y, _radius = mapped
        sample = self.terrain.sample(map_x, map_y)
        if not sample.in_bounds or math.isnan(sample.elevation_m):
            return None
        return self._radius_m + float(sample.elevation_m)

    def _bisect_hit(self, origin: Vector, ray: Vector, above_t: float, below_t: float) -> Vector:
        """Refine the ray-surface crossing between an above-surface and below-surface ``t``."""
        lo, hi = above_t, below_t
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            point = add_scaled(origin, ray, mid)
            surface_radius = self._surface_radius(point)
            if surface_radius is None or norm(point) >= surface_radius:
                lo = mid
            else:
                hi = mid
        return add_scaled(origin, ray, hi)

    def _solar_state(
        self, map_x: float, map_y: float, epoch: Epoch | None
    ) -> tuple[IlluminationState, float, float]:
        """Illumination state, solar flux, and surface temperature at a cell.

        Temperature comes from the injected :class:`ThermalSource` when present, else a coarse
        radiative-equilibrium first-cut. With no ``epoch`` the time-varying solar state is
        unknown, so the static dark baseline (shadow, zero flux, floor temperature) is
        returned without consulting the thermal source (which needs an epoch).
        """
        if epoch is None:
            return IlluminationState.SHADOW, 0.0, self._pack.shadow_floor_k
        lit = self.illumination.sun_visible(map_x, map_y, epoch)
        lon, lat = self._map_to_lonlat(map_x, map_y)
        elevation_deg = sun_geometry(
            self._pack.site(lat, lon), epoch, abcorr=self.illumination.abcorr
        ).elevation_deg
        flux = solar_flux(
            elevation_deg, lit=lit, solar_constant_w_m2=self._pack.solar_constant_w_m2
        )
        state = IlluminationState.LIT if flux > 0.0 else IlluminationState.SHADOW
        if self._thermal is not None:
            temperature = self._thermal.temperature_k(
                map_x=map_x, map_y=map_y, epoch=epoch, solar_flux_w_m2=flux
            )
        else:
            temperature = equilibrium_temperature(
                flux,
                bond_albedo=self._pack.bond_albedo,
                emissivity=self._pack.emissivity,
                shadow_floor_k=self._pack.shadow_floor_k,
            )
        return state, flux, temperature

    def _unmapped_point(self, position: Vector) -> SurfacePoint:
        """A well-formed default :class:`SurfacePoint` for a position off the modelled grid.

        Radial gravity still resolves (it needs only the position); geometry/illumination/
        regolith fall back to flat/dark/unknown defaults.
        """
        return SurfacePoint(
            frame=self._pack.surface_frame,
            elevation_m=0.0,
            surface_normal=(0.0, 0.0, 1.0),
            gravity=self._pack.gravity(position),
            illumination=Illumination(state=IlluminationState.SHADOW, solar_flux_w_m2=0.0),
            temperature_k=self._pack.shadow_floor_k,
            regolith=RegolithParams(),
        )
