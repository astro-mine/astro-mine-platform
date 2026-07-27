"""Build (and optionally publish) the anchor world bundle `shackleton-de-gerlache-v1`.

The canonical, repeatable recipe for the Phase-0 anchor world (RM-P1-WORLDS-15): it authors the
`shackleton-de-gerlache-v1` WorldSpec (there is no other authored spec), ingests the real LOLA DEM,
computes SPICE-backed illumination + a PSR mask over a sampled epoch window, attaches reduced-order
thermal curves, and assembles a content-addressed `build_world_bundle`. Prints the `world_hash`.

Data (see files/data/, fetched once):
  - a CRS-tagged **metres** GeoTIFF derived from LOLA `LDEM_875S_5M` (PDS LRO-L-LOLA-4-GDR-V1.0);
    the raw product is elevation in km on a lunar polar-stereographic sphere (R=1737.4 km), so it is
    downsampled to the working resolution and scaled km->m by `--convert` (or pass a ready GeoTIFF).
  - NAIF DE440 SPICE kernels via a meta-kernel (LSK/PCK/binary-PCK/FK/SPK).

The world_hash is reproducible only within a pinned toolchain (GDAL/PROJ/numpy) on one OS — see the
determinism notes; the benchmark is reproducible because consumers *pull* the published bundle by
digest, not rebuild it.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.spice import epoch_from_utc, kernel_pool
from astro_mine.worlds import terrain
from astro_mine.worlds.crs import LUNAR_SOUTH_POLAR_STEREOGRAPHIC
from astro_mine.worlds.illumination import HorizonFrame, IlluminationModel, PsrEpochSemantics
from astro_mine.worlds.regolith import build_regolith_field
from astro_mine.worlds.spec import LayerSpec, Region, SourceRef, WorldSpec, build_world_bundle
from astro_mine.worlds.thermal import diurnal_curve

WORLD_ID = "shackleton-de-gerlache-v1"
_SYNODIC_MONTH_S = 29.530588 * 86_400.0
_NATIVE_RES_M = 5.0  # LDEM_875S_5M ground sample distance


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def convert_dem(raw: Path, dst: Path, resolution_m: float) -> None:
    """Downsample the raw LOLA PDS DEM (km) to a metres GeoTIFF at ``resolution_m``.

    Nodata-aware average decimation (GDAL skips the -3.4e38 fill), then km->m. The source is already
    lunar polar-stereographic (R=1737400), so its CRS is carried through unchanged.
    """
    with rasterio.open(raw) as ds:
        factor = round(resolution_m / _NATIVE_RES_M)
        new_w, new_h = ds.width // factor, ds.height // factor
        arr = ds.read(1, out_shape=(new_h, new_w), resampling=Resampling.average)
        nodata, crs, t = ds.nodata, ds.crs, ds.transform
    mask = arr <= -1e30
    out = np.where(mask, np.float32(nodata), arr * 1000.0).astype(np.float32)
    new_t = Affine(resolution_m, 0.0, t.c, 0.0, -resolution_m, t.f)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        dst,
        "w",
        driver="GTiff",
        width=new_w,
        height=new_h,
        count=1,
        dtype="float32",
        crs=crs,
        transform=new_t,
        nodata=nodata,
        compress="deflate",
    ) as d:
        d.write(out, 1)


def build(args: argparse.Namespace) -> str:
    out = Path(args.out)
    if args.convert:
        dem_tif = out / "elevation_m.tif"
        convert_dem(Path(args.raw_dem), dem_tif, args.resolution_m)
    else:
        dem_tif = Path(args.dem)

    product = terrain.ingest_dem(dem_tif, out / "terrain", resolution_m=args.resolution_m)
    regolith = build_regolith_field(product, out / "regolith")

    # Author the WorldSpec first, recording every parameter that determines the PSR mask (issue
    # #36), then drive the illumination build *from the spec* so the declaration is the single
    # source of truth and the bundle rebuilds to its own world_hash (worlds.md §10 determinism).
    grid = product.manifest["grid"]
    half_x = grid["width"] * args.resolution_m / 2.0
    half_y = grid["height"] * args.resolution_m / 2.0
    spec = WorldSpec(
        world_id=WORLD_ID,
        version=args.version,
        crs=LUNAR_SOUTH_POLAR_STEREOGRAPHIC,
        region=Region(
            min_x_m=-half_x,
            min_y_m=-half_y,
            max_x_m=half_x,
            max_y_m=half_y,
            resolution_m=args.resolution_m,
        ),
        source_dem=SourceRef(
            id="LDEM_875S_5M",
            # Pin the exact input by the .img data digest (GDAL opens the sibling .lbl label).
            content_hash=_sha256(Path(args.raw_dem).with_suffix(".img")) if args.raw_dem else None,
            description=(
                "LOLA LDEM 87.5S 5m/px (PDS LRO-L-LOLA-4-GDR-V1.0), nodata-aware-downsampled to "
                f"{args.resolution_m:g} m and scaled km->m."
            ),
        ),
        layers=LayerSpec(
            regolith_prior="default_lunar",
            illumination_n_azimuth=args.n_azimuth,
            illumination_horizon_frame=args.horizon_frame,
            illumination_max_radius_m=args.max_radius_m,
            illumination_abcorr=args.abcorr,
            psr_semantics=args.psr_semantics,
            psr_start=args.psr_start,
            psr_days=args.psr_days,
            psr_step_hours=args.psr_step_hours,
            thermal_classes=("polar_lit", "crater_floor"),
        ),
        description=(
            "Shackleton-de Gerlache ridge, lunar south pole. LOLA 5 m DEM downsampled to "
            f"{args.resolution_m:g} m; SPICE DE440 illumination + PSR over "
            f"{args.psr_days:g} d; reduced-order thermal."
        ),
    )

    with kernel_pool(args.metakernel):
        model = IlluminationModel.from_spec(spec, product)
        start = epoch_from_utc(args.psr_start)
        end = Epoch(tdb_seconds=start.tdb_seconds + args.psr_days * 86_400.0, scale=TimeScale.TDB)
        window = EpochWindow(start=start, end=end)
        psr = model.psr_mask(
            window, args.psr_step_hours * 3600.0, semantics=PsrEpochSemantics(args.psr_semantics)
        )

    thermal = [diurnal_curve("polar_lit"), diurnal_curve("crater_floor")]

    bundle = build_world_bundle(
        spec,
        terrain=product,
        regolith=regolith,
        psr=psr,
        thermal=thermal,
        out_dir=out / "bundle",
    )
    ever_lit = float(psr.ever_lit_fraction)
    print(f"world_hash: {bundle.world_hash}")
    print(f"grid: {grid['width']}x{grid['height']} @ {args.resolution_m:g} m")
    print(
        f"PSR: {float(psr.mask.mean()) * 100:.2f}% of cells never sunlit "
        f"(ever-lit fraction {ever_lit:.4f}) over {psr.n_epochs} epochs"
    )
    print(f"bundle: {bundle.path}")
    return str(bundle.world_hash)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dem", help="LOLA PDS .lbl label (GDAL opens it; the .img is hashed)")
    p.add_argument("--dem", help="a ready CRS-tagged metres GeoTIFF (skip --convert)")
    p.add_argument("--convert", action="store_true", help="convert --raw-dem to a metres GeoTIFF")
    p.add_argument("--metakernel", required=True, help="SPICE meta-kernel (.tm)")
    p.add_argument("--out", required=True, help="output working directory")
    p.add_argument("--version", default="0.1.0", help="the WorldSpec version (SemVer)")
    p.add_argument("--resolution-m", type=float, default=120.0)
    p.add_argument("--n-azimuth", type=int, default=120)
    p.add_argument("--max-radius-m", type=float, default=30_000.0)
    p.add_argument(
        "--horizon-frame",
        default="grid",
        choices=[f.value for f in HorizonFrame],
        help="illumination horizon frame; GRID applies one region-centre Sun, TOPOCENTRIC per-cell",
    )
    p.add_argument("--abcorr", default="NONE", help="SPICE aberration correction for the Sun")
    p.add_argument("--psr-start", default="2025-06-21T00:00:00")
    p.add_argument("--psr-days", type=float, default=_SYNODIC_MONTH_S / 86_400.0)
    p.add_argument("--psr-step-hours", type=float, default=12.0)
    p.add_argument(
        "--psr-semantics", default="seasonal", choices=[s.value for s in PsrEpochSemantics]
    )
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
