# SPDX-License-Identifier: Apache-2.0
"""The content-addressed world bundle (RM-P0-WORLDS-07; worlds.md §3, §5).

:func:`build_world_bundle` composes the already-built, content-addressed layer products
(terrain RM-P0-WORLDS-01, regolith RM-P0-WORLDS-05, the illumination/PSR mask RM-P0-WORLDS-03,
the thermal curves RM-P0-WORLDS-04) described by a :class:`~astro_mine.worlds.spec._model.WorldSpec`
into a single distributable bundle: the layer COGs, a STAC catalog, a 3D-Tiles export, and a
``world.json`` manifest. The bundle's :attr:`WorldBundle.world_hash` is a ``sha256`` over the
canonical spec, the resolved component hashes, and the pinned toolchain — so it is **reproducible
from the WorldSpec + toolchain** (the components are themselves deterministic, established by
WORLDS-01..06), and Bench can pin a world by that hash (worlds.md §5; RM-P0-BENCH-01).

Backlog: RM-P0-WORLDS-07 — astro-mine-worlds#7
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import shutil
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from astro_mine.worlds.crs import to_rasterio_crs
from astro_mine.worlds.fields import (
    ZARR_MEDIA_TYPE,
    FieldArray,
    FieldStore,
    write_field_zarr,
    zarr_version,
)
from astro_mine.worlds.illumination import (
    HORIZON_ARRAY,
    HORIZON_DIMS,
    HORIZON_STORE_NAME,
    PsrResult,
)
from astro_mine.worlds.regolith import RegolithField, RegolithProduct
from astro_mine.worlds.spec._model import WorldSpec
from astro_mine.worlds.spec._schema import (
    PLANETARY_CRS_DEF,
    REFERENCE_FRAME_DEF,
    UNITS_SCHEMA_ID,
    units_def_ref,
    validate_units_object,
)
from astro_mine.worlds.spec._stac import StacLayer, write_stac_catalog
from astro_mine.worlds.spec._tiles import TILESET_NAME, export_3d_tiles
from astro_mine.worlds.terrain import TerrainModel, TerrainProduct
from astro_mine.worlds.terrain._ingest import write_cog
from astro_mine.worlds.thermal import DiurnalCurve

__all__ = ["WorldBundle", "build_world_bundle"]

BUNDLE_SCHEMA = "astro-mine-worlds/world/v0.1"

_MANIFEST_NAME = "world.json"
#: Each layer product (terrain, regolith) writes its own ``manifest.json``; the bundle copies it
#: alongside the COGs so a pulled bundle is self-describing and re-openable (RM-P1-WORLDS-15).
_PRODUCT_MANIFEST_NAME = "manifest.json"
_Transform = tuple[float, float, float, float, float, float]

#: The Zarr field store holding the per-class diurnal temperature curves (worlds.md §5's format
#: table puts **thermal** among the "Field models -> Zarr" N-D layers). ``thermal.json`` keeps the
#: human-readable summary (the night floor / peak / period each class is scored on); the *curves*
#: themselves — the ``(n_class, n_phase)`` stack that RM-P0-SIM-07's power/thermal model integrates
#: over — were previously computed and then dropped on the floor. Now they ship.
_THERMAL_STORE_NAME = "curves.zarr"
_THERMAL_ARRAY = "temperature_k"
_THERMAL_PHASE_ARRAY = "phase"


@dataclass(frozen=True)
class WorldBundle:
    """A content-addressed world bundle on disk — the distributable unit (worlds.md §5)."""

    path: Path
    world_id: str
    world_hash: str
    spec: WorldSpec
    component_hashes: dict[str, str]
    manifest: dict[str, Any]
    stac_catalog: Path
    tileset: Path

    @classmethod
    def load(cls, path: str | Path) -> WorldBundle:
        """Reconstruct a :class:`WorldBundle` from a built bundle directory on disk.

        Reads ``world.json`` and rebuilds the value object without re-running the (expensive,
        raster-touching) build — the entry the ``worlds publish`` CLI and any consumer that
        already has the bundle bytes use to hand it to :func:`publish_world_bundle`.
        """
        root = Path(path)
        manifest: dict[str, Any] = json.loads((root / _MANIFEST_NAME).read_text(encoding="utf-8"))
        spec = WorldSpec.model_validate(manifest["spec"])
        return cls(
            path=root,
            world_id=str(manifest["world_id"]),
            world_hash=str(manifest["world_hash"]),
            spec=spec,
            component_hashes=dict(manifest["components"]),
            manifest=manifest,
            stac_catalog=root / str(manifest["stac"]),
            tileset=root / str(manifest["tiles"]),
        )


def _worlds_version() -> str:
    try:
        return importlib.metadata.version("astro-mine-platform")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed in dev/CI
        return "0+unknown"


def _combined_thermal_hash(curves: Sequence[DiurnalCurve]) -> str:
    digest = hashlib.sha256()
    for curve in sorted(curves, key=lambda c: c.terrain_class):
        digest.update(curve.terrain_class.encode("utf-8"))
        digest.update(curve.thermal_hash.encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def _write_horizon_map(path: Path, psr: PsrResult) -> FieldStore:
    """Persist the PSR result's ``(H, W, n_azimuth)`` horizon map as the bundle's Zarr field layer.

    Written from the :class:`PsrResult` — which carries both the array and the producing model's
    manifest — so the bundle build stays decoupled from the live
    :class:`~astro_mine.worlds.illumination.IlluminationModel` (issue #36's separation), while the
    store still records the full provenance the load path validates a reuse against.
    """
    assert psr.horizon is not None  # guarded by the caller
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_field_zarr(
        path,
        [FieldArray(name=HORIZON_ARRAY, values=psr.horizon, units="degree", dims=HORIZON_DIMS)],
        attrs={
            "layer": "illumination/horizon",
            "illumination_hash": psr.illumination_hash,
            "manifest": psr.illumination_manifest,
        },
    )


def _write_thermal_curves(path: Path, curves: Sequence[DiurnalCurve]) -> FieldStore:
    """Persist the per-class diurnal temperature curves as a chunked ``(n_class, n_phase)`` Zarr.

    worlds.md §5 puts **thermal** among the "Field models -> Zarr" N-D layers. The bundle already
    shipped a ``thermal.json`` *summary* (night floor / peak / period); the solved curves themselves
    — what Sim's power/thermal model actually integrates over — were discarded at bundle time. All
    classes share the same phase grid (they are solved on one), so the phases are one 1-D array and
    the temperatures a class-major stack.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    phases = np.asarray(curves[0].phases, dtype=np.float64)
    if any(curve.phases.shape != phases.shape for curve in curves):
        raise ValueError("diurnal curves must share one phase grid to stack into a field layer")
    temperatures = np.stack([np.asarray(c.temperatures_k, dtype=np.float64) for c in curves])
    return write_field_zarr(
        path,
        [
            FieldArray(name=_THERMAL_PHASE_ARRAY, values=phases, units="1", dims=("phase",)),
            FieldArray(
                name=_THERMAL_ARRAY,
                values=temperatures,
                units="K",
                dims=("terrain_class", "phase"),
            ),
        ],
        attrs={
            "layer": "thermal/curves",
            "terrain_classes": [curve.terrain_class for curve in curves],
            "period_s": [curve.period_s for curve in curves],
            "thermal_hashes": [curve.thermal_hash for curve in curves],
        },
    )


def _copy_layers(manifest: dict[str, Any], src: Path, dst: Path) -> dict[str, str]:
    """Copy a product's COG layers **and its ``manifest.json``** into ``dst``.

    Returns ``{layer_name: units}``. The product manifest is copied so the bundle sub-directory
    is a self-describing terrain/regolith product that :class:`~astro_mine.worlds.terrain.\
TerrainModel`/:class:`~astro_mine.worlds.regolith.RegolithField` can re-open directly from a
    pulled bundle — the gap RM-P1-WORLDS-15 closes.
    """
    dst.mkdir(parents=True, exist_ok=True)
    units: dict[str, str] = {}
    for name, info in sorted(manifest["layers"].items()):
        shutil.copy2(src / f"{name}.tif", dst / f"{name}.tif")
        units[name] = str(info["units"])
    shutil.copy2(src / _PRODUCT_MANIFEST_NAME, dst / _PRODUCT_MANIFEST_NAME)
    return units


def build_world_bundle(
    spec: WorldSpec,
    *,
    terrain: TerrainModel | TerrainProduct | str | Path,
    out_dir: str | Path,
    regolith: RegolithField | RegolithProduct | str | Path | None = None,
    psr: PsrResult | None = None,
    thermal: Sequence[DiurnalCurve] | None = None,
    tiles_max_dim: int = 64,
) -> WorldBundle:
    """Assemble a content-addressed world bundle from a spec and its built layer products.

    Validates the components share the spec's CRS, copies their COG layers into the bundle, writes
    a STAC catalog and a 3D-Tiles terrain export, computes the ``world_hash``, and writes the
    ``world.json`` manifest. Returns the :class:`WorldBundle`.
    """
    terrain_model = terrain if isinstance(terrain, TerrainModel) else TerrainModel.open(terrain)
    crs = terrain_model.crs
    if crs != spec.crs:
        raise ValueError("terrain CRS does not match the WorldSpec CRS")
    grid = terrain_model.manifest["grid"]
    transform: _Transform = tuple(grid["transform"])
    width, height = int(grid["width"]), int(grid["height"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    component_hashes: dict[str, str] = {"terrain": str(terrain_model.manifest["terrain_hash"])}
    stac_layers: list[StacLayer] = []
    #: The chunked Zarr field layers this bundle ships (worlds.md §5) — name -> written store.
    field_stores: dict[str, FieldStore] = {}

    terrain_units = _copy_layers(terrain_model.manifest, terrain_model.path, out / "terrain")
    for name, units in sorted(terrain_units.items()):
        stac_layers.append(
            StacLayer(f"terrain-{name}", f"../terrain/{name}.tif", f"terrain/{name}", units)
        )

    if regolith is not None:
        reg = regolith if isinstance(regolith, RegolithField) else RegolithField.open(regolith)
        component_hashes["regolith"] = str(reg.manifest["regolith_hash"])
        reg_units = _copy_layers(reg.manifest, reg.path, out / "regolith")
        for name, units in sorted(reg_units.items()):
            stac_layers.append(
                StacLayer(f"regolith-{name}", f"../regolith/{name}.tif", f"regolith/{name}", units)
            )

    if psr is None:
        _warn_if_psr_declared_but_absent(spec)
    if psr is not None:
        if psr.mask.shape != (height, width):
            raise ValueError(
                f"PSR mask shape {psr.mask.shape} does not match the terrain grid {(height, width)}"
            )
        # The WorldSpec must *determine* the PSR mask (worlds.md §10 determinism gate): a bundle
        # cannot ship a mask its own declaration cannot reproduce (issue #36).
        _check_illumination_matches_spec(spec, psr)
        # Fold the SPICE-derived PSR mask into the world hash, not just the terrain horizon, so the
        # digest pins the permanently-shadowed regions the anchor scores (RM-P1-WORLDS-15).
        component_hashes["illumination"] = psr.psr_hash
        (out / "illumination").mkdir(parents=True, exist_ok=True)
        write_cog(
            out / "illumination" / "psr_mask.tif",
            psr.mask.astype(np.uint8),
            transform,
            to_rasterio_crs(crs),
            None,
        )
        # Write the illumination provenance beside the mask (frame / radius / abcorr / window /
        # step), as terrain/ and regolith/ already do, so a pulled bundle is self-describing about
        # how its PSR mask was computed — not just the mask bytes (issue #36).
        (out / "illumination" / _PRODUCT_MANIFEST_NAME).write_text(
            json.dumps(psr.to_manifest(), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        stac_layers.append(
            StacLayer(
                "illumination-psr_mask",
                "../illumination/psr_mask.tif",
                "illumination/psr_mask",
                "bool",
            )
        )
        # The per-azimuth horizon map: an (H, W, n_azimuth) N-D field, so **Zarr**, not a flat COG
        # (worlds.md §5's format table). Persisting it is the point of issue #39 — without it every
        # `IlluminationModel` construction re-derives the whole skyline in-process, and the §7 cloud
        # precompute/serve tier has no artifact to range-read. A PsrResult built directly (a test
        # double) carries no horizon, and then the bundle simply ships none: the load path falls
        # back to recomputing, exactly as before.
        if psr.horizon is not None:
            if psr.horizon.shape[:2] != (height, width):
                raise ValueError(
                    f"horizon map shape {psr.horizon.shape} does not match the terrain grid "
                    f"{(height, width)}"
                )
            field_stores["illumination/horizon"] = _write_horizon_map(
                out / "illumination" / HORIZON_STORE_NAME, psr
            )
            stac_layers.append(
                StacLayer(
                    "illumination-horizon",
                    f"../illumination/{HORIZON_STORE_NAME}",
                    "illumination/horizon",
                    "degree",
                    media_type=ZARR_MEDIA_TYPE,
                )
            )

    if thermal:
        component_hashes["thermal"] = _combined_thermal_hash(thermal)
        (out / "thermal").mkdir(parents=True, exist_ok=True)
        curves = sorted(thermal, key=lambda c: c.terrain_class)
        thermal_doc = {
            "thermal_hash": component_hashes["thermal"],
            "classes": [
                {
                    "terrain_class": curve.terrain_class,
                    "thermal_hash": curve.thermal_hash,
                    "night_floor_k": curve.night_floor_k,
                    "peak_k": curve.peak_k,
                    "period_s": curve.period_s,
                }
                for curve in curves
            ],
        }
        (out / "thermal" / "thermal.json").write_text(
            json.dumps(thermal_doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        field_stores["thermal/curves"] = _write_thermal_curves(
            out / "thermal" / _THERMAL_STORE_NAME, curves
        )
        stac_layers.append(
            StacLayer(
                "thermal-curves",
                f"../thermal/{_THERMAL_STORE_NAME}",
                "thermal/curves",
                "K",
                media_type=ZARR_MEDIA_TYPE,
            )
        )

    # Fold every persisted Zarr layer's **store** hash into the component hashes, so the world hash
    # pins the bytes that actually ship. The horizon array's *content* already reaches the world
    # hash transitively (illumination_hash -> psr_hash), but the persisted store does not: without
    # this, a bundle whose horizon.zarr was tampered with — or one shipping no horizon at all —
    # would carry the same world_hash as the honest, fully-populated build (issue #39).
    for name, field_store in sorted(field_stores.items()):
        component_hashes[name.replace("/", "_")] = field_store.store_hash

    proj4 = crs.projection or ""
    catalog = write_stac_catalog(
        out / "stac",
        world_id=spec.world_id,
        description=spec.description or f"Astro-Mine world bundle: {spec.world_id}",
        proj4=proj4,
        shape=(height, width),
        transform=transform,
        datetime_iso=spec.reference_datetime,
        layers=stac_layers,
    )

    with rasterio.open(out / "terrain" / "elevation.tif") as ds:
        elevation = ds.read(1).astype(np.float64)
    tiles = export_3d_tiles(out / "tiles", elevation, transform, crs=crs, max_dim=tiles_max_dim)

    # Recorded as provenance in the manifest, never hashed (astro-mine-worlds#46). Zarr is listed
    # only when a Zarr layer was actually written, so the record names the libraries that truly ran.
    toolchain = {
        "astro_mine_worlds": _worlds_version(),
        "numpy": np.__version__,
    }
    if field_stores:
        toolchain["zarr"] = zarr_version()
    world_hash = _world_hash(spec, component_hashes)

    # Pin the serialized units objects to Core's canonical schema at emit (RM-P1-WORLDS-17,
    # RFC-0007 Design §1a): the ``crs`` and ``tiles_anchor.frame`` mappings that go on disk are
    # validated against ``units.schema.json`` so a non-Python consumer (View, RM-P1-VIEW-06) has a
    # schema to check them against, not just a hand-written mirror.
    crs_manifest = crs.model_dump(mode="json")
    validate_units_object(crs_manifest, PLANETARY_CRS_DEF)
    tiles_anchor_manifest = tiles.anchor.to_manifest()
    validate_units_object(tiles_anchor_manifest["frame"], REFERENCE_FRAME_DEF)

    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "world_id": spec.world_id,
        "version": spec.version,
        "spec": spec.model_dump(mode="json"),
        "spec_hash": spec.spec_hash,
        "components": dict(sorted(component_hashes.items())),
        "crs": crs_manifest,
        # The Core units schema the ``crs`` and ``tiles_anchor.frame`` objects conform to —
        # published so a consumer resolves them to Core's vocabulary, not a mirror (RFC-0007).
        "units_schema": {
            "id": UNITS_SCHEMA_ID,
            "crs": units_def_ref(PLANETARY_CRS_DEF),
            "tiles_anchor_frame": units_def_ref(REFERENCE_FRAME_DEF),
        },
        "grid": {
            "width": width,
            "height": height,
            "resolution_m": grid["resolution_m"],
            "transform": list(transform),
        },
        "stac": "stac/catalog.json",
        # `tiles` stays a string: `WorldBundle.load` and View's world-manifest reader both take it
        # as a relative path. The anchor is published beside it, additively (RM-P1-WORLDS-16).
        "tiles": f"tiles/{TILESET_NAME}",
        "tiles_anchor": tiles_anchor_manifest,
        # The chunked N-D field layers (worlds.md §5): where each Zarr store sits, what arrays it
        # holds, and its content hash — so a consumer knows a horizon map is *there* (and can skip
        # the rebuild) without opening the store, and an empty mapping honestly says it is not.
        "fields": {
            name: {**store.to_manifest(), "path": store.path.relative_to(out).as_posix()}
            for name, store in sorted(field_stores.items())
        },
        "toolchain": toolchain,
        "world_hash": world_hash,
    }
    (out / _MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return WorldBundle(
        path=out,
        world_id=spec.world_id,
        world_hash=world_hash,
        spec=spec,
        component_hashes=dict(sorted(component_hashes.items())),
        manifest=manifest,
        stac_catalog=catalog,
        tileset=tiles.tileset,
    )


#: The PSR parameters whose presence means "this spec expects a PSR layer".
_PSR_DECLARING_FIELDS = ("psr_semantics", "psr_start", "psr_days", "psr_step_hours")


def _warn_if_psr_declared_but_absent(spec: WorldSpec) -> None:
    """Say so when a spec declares PSR parameters and no PSR layer was supplied (issue #60).

    The check for a *supplied* mask is strict — a bundle may not ship a mask its own declaration
    cannot reproduce (:func:`_check_illumination_matches_spec`, issue #36) — and the omitted case
    was silent, which is the same gap from the other side: the bundle's components then contain no
    illumination while its spec says `psr_semantics: seasonal`, and nothing tells the builder.

    A **warning**, not an error, because a PSR-free build is a legitimate and useful thing to do:
    computing the mask needs a furnished SPICE kernel pool, and the shipped synthetic example
    exists precisely so a reader can reach a bundle without kernels. What is not legitimate is
    doing it without noticing.
    """
    declared = [
        name for name in _PSR_DECLARING_FIELDS if getattr(spec.layers, name, None) is not None
    ]
    if not declared:
        return
    warnings.warn(
        f"{spec.world_id}: the spec declares PSR parameters ({', '.join(declared)}) but no PSR "
        "layer was supplied, so this bundle has no `illumination` component and its world_hash "
        "does not pin one. Compute one with `IlluminationModel(...).psr_mask(...)` inside a "
        "furnished `kernel_pool(...)` and pass `psr=`, or drop the `psr_*` fields from the spec so "
        "the declaration matches what is built.",
        UserWarning,
        stacklevel=3,
    )


def _check_illumination_matches_spec(spec: WorldSpec, psr: PsrResult) -> None:
    """Fail if the PSR mask was computed with parameters the WorldSpec declares differently.

    The WorldSpec is meant to *determine* the PSR mask (worlds.md §10 determinism gate; §5 /
    conventions.md §5 provenance): every mask-affecting parameter it records MUST match how ``psr``
    was actually computed, or a bundle could ship a mask its own declaration cannot reproduce
    (issue #36). Only declared (non-``None``) fields are checked — a spec that leaves an
    illumination parameter unspecified is unconstrained on it, and a fully-specified anchor spec is
    fully pinned. The window *start* is recorded in ``illumination/manifest.json`` but not
    re-derived here (that needs SPICE, which this layer deliberately does not import); the window
    *duration* and *step* are checkable from the epoch bounds and are enforced.
    """
    layers = spec.layers
    params = psr.illumination_manifest.get("params", {})
    mismatches: list[str] = []

    def _check(name: str, declared: object, actual: object) -> None:
        if declared is not None and declared != actual:
            mismatches.append(f"{name}: spec declares {declared!r}, mask used {actual!r}")

    _check("illumination_n_azimuth", layers.illumination_n_azimuth, params.get("n_azimuth"))
    _check(
        "illumination_max_radius_m", layers.illumination_max_radius_m, params.get("max_radius_m")
    )
    _check("illumination_abcorr", layers.illumination_abcorr, params.get("abcorr"))
    _check(
        "illumination_horizon_frame",
        layers.illumination_horizon_frame,
        params.get("horizon_frame"),
    )
    _check("psr_semantics", layers.psr_semantics, psr.semantics.value)
    _check("psr_step_hours", layers.psr_step_hours, psr.step_s / 3600.0)
    if layers.psr_days is not None:
        actual_days = (psr.window.end.tdb_seconds - psr.window.start.tdb_seconds) / 86_400.0
        if not math.isclose(layers.psr_days, actual_days, rel_tol=1e-9, abs_tol=1e-6):
            mismatches.append(
                f"psr_days: spec declares {layers.psr_days!r}, window spans {actual_days!r}"
            )

    if mismatches:
        raise ValueError(
            "PSR mask does not match the WorldSpec illumination declaration: "
            + "; ".join(mismatches)
        )


def _world_hash(spec: WorldSpec, component_hashes: dict[str, str]) -> str:
    """``sha256`` over the canonical spec and the resolved component hashes.

    The toolchain is recorded in the manifest but **not** hashed (astro-mine-worlds#46;
    :mod:`astro_mine.worlds._hashing`). It would be redundant: every component hash covers its own
    bytes — including each Zarr field store's ``store_hash`` — so a toolchain that writes different
    chunks, codecs or rasters moves ``world_hash`` through the component that changed. Hashing the
    version string on top of that bought nothing and cost reproducibility, because the hatch-vcs dev
    version moves with every commit.
    """
    payload = {
        "spec_hash": spec.spec_hash,
        "components": dict(sorted(component_hashes.items())),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
