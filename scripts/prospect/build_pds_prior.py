#!/usr/bin/env python3
"""Build (and optionally Hub-publish) the real PDS raster-ingest water-ice prior (PROSPECT-12).

Ingests the four public conditioning rasters onto the Shackleton prior grid, materializes the
content-addressed conditioning bundle, fits ``shackleton_water_ice_pds_v1``, and (with
``--registry``) publishes it as a signed, content-addressed Hub artifact. The multi-GB raster fetch
is a **one-time, documented, cached** step (``scripts/fetch_pds_conditioning.py``); this script
turns those cached rasters into the small conditioning bundle the offline recipe fits from, so the
local tier (``LUNAR-TR-004``) never touches GDAL or the network.

    # 1. fetch the public rasters (one-time)
    python scripts/fetch_pds_conditioning.py --dir data/pds
    # 2. build the conditioning bundle + fit the prior (+ publish to a local Hub registry)
    python scripts/build_pds_prior.py \
        --diviner data/pds/diviner/dgdr_tbol_avg_pols_20100107n_240_img.lbl \
        --m3       data/pds/m3/0416_OP2A_out_magnetotail_mosaic.img \
        --lend     data/pds/lend/lend_sanin2018_pss.txt \
        --psr      data/shackleton-build/bundle/illumination/psr_mask.tif \
        --out      data/pds/conditioning \
        --registry files/hub-registry

Then point the recipe at the bundle to re-fit offline:

    ASTRO_MINE_PROSPECT_CONDITIONING=data/pds/conditioning python -c \
        "from astro_mine.prospect.priors import load_prior; \
         print(load_prior('shackleton_water_ice_pds_v1').content_hash)"

Sources (verify against the live archives — PDS/Zenodo paths change over time):
- **Diviner** GDR L3 south-polar bolometric temperature (Tbol), PDS Geosciences Node — 240 m
  polar-stereographic ``int16`` (K = DN·0.02, missing -32768).
- **M³** Chandrayaan-1 surficial-hydration band-depth mosaic (Lu et al. 2024, GRL; Zenodo
  10608904) — ~140 m polar-stereographic ENVI ``float32`` (ignore value 0).
- **LEND** epithermal-neutron grid (Sanin et al. 2018, PSS 162 supplement) — 3° global ASCII
  (CSETN collimated counts/background/exposure); rasterized here to a global suppression index.
- **PSR** LOLA + SPICE-derived permanently-shadowed-region mask from the Worlds
  ``shackleton-de-gerlache-v1`` world bundle (RM-P1-WORLDS-14 illumination component).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astro_mine.prospect.priors import build_pds_prior, load_conditioning_bundle
from astro_mine.prospect.priors.catalog import SHACKLETON_CRS, SHACKLETON_PRIOR_GRID
from astro_mine.prospect.priors.ingest import (
    RasterInput,
    ingest_conditioning,
    materialize_conditioning_bundle,
)

# LEND Sanin-2018 ASCII: 2 scalar header lines (n_lon, n_lat), 2 column-header lines, then rows of
# `LODX LADX E_long W_long lat  STN3(cts,bkd,sec) SETN(...) CSETN(cts,bkd,sec) alt`. Data rows start
# with the integer LODX index; header/rule lines start with `LODX`/`----`, so filter by that.
_LEND_ELON, _LEND_LAT = 2, 4
_LEND_CSETN_CTS, _LEND_CSETN_BKD, _LEND_CSETN_SEC = 11, 12, 13
_LEND_RES_DEG = 3.0
_LEND_DRY_PCTL, _LEND_WET_PCTL = (
    90.0,
    10.0,
)  # count-rate percentiles anchoring the suppression index


def lend_ascii_to_suppression_geotiff(ascii_path: Path, out_path: Path) -> Path:
    """Rasterize the LEND Sanin-2018 ASCII grid to a global epithermal-suppression-index GeoTIFF.

    Computes the CSETN (collimated epithermal) background-subtracted count rate per 3° cell, then a
    normalized **suppression index ∈ [0, 1]** (hydrogen suppresses epithermal neutrons, so a lower
    count rate ⇒ more water-equivalent hydrogen): ``supp = clip((dry - rate)/(dry - wet), 0, 1)``
    with ``dry``/``wet`` the p90/p10 count-rate anchors. Written as an equirectangular lunar
    geographic GeoTIFF (``+proj=longlat +R=1737400``), NaN nodata — the ingest step reprojects it
    onto the polar prior grid.
    """
    import rasterio

    rows = [
        tokens
        for line in ascii_path.read_text(encoding="utf-8").splitlines()
        if (tokens := line.split()) and tokens[0].isdigit()
    ]
    n_lat, n_lon = 60, 120
    rate = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    for r in rows:
        sec = float(r[_LEND_CSETN_SEC])
        if sec <= 0.0:
            continue
        net = (float(r[_LEND_CSETN_CTS]) - float(r[_LEND_CSETN_BKD])) / sec
        col = round((float(r[_LEND_ELON]) - _LEND_RES_DEG / 2.0) / _LEND_RES_DEG) % n_lon
        row = round((90.0 - _LEND_RES_DEG / 2.0 - float(r[_LEND_LAT])) / _LEND_RES_DEG)
        if 0 <= row < n_lat:
            rate[row, col] = net
    finite = np.isfinite(rate)
    dry = float(np.percentile(rate[finite], _LEND_DRY_PCTL))
    wet = float(np.percentile(rate[finite], _LEND_WET_PCTL))
    supp = np.clip((dry - rate) / (dry - wet), 0.0, 1.0).astype(np.float32)
    supp = np.where(finite, supp, np.nan).astype(np.float32)

    transform = rasterio.transform.Affine(_LEND_RES_DEG, 0.0, 0.0, 0.0, -_LEND_RES_DEG, 90.0)
    crs = rasterio.crs.CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=n_lat,
        width=n_lon,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        dst.write(supp, 1)
    print(
        f"LEND: {int(finite.sum())}/{n_lat * n_lon} cells; dry={dry:.3f} wet={wet:.3f} "
        f"cts/s -> {out_path}",
        file=sys.stderr,
    )
    return out_path


def _summarize(name: str, arr: NDArray[np.float64]) -> None:
    fin = np.isfinite(arr)
    if not fin.any():
        print(f"  {name}: NO COVERAGE on the prior grid", file=sys.stderr)
        return
    print(
        f"  {name}: coverage={100 * fin.mean():.1f}%  "
        f"min/med/max={np.nanmin(arr):.3g}/{np.nanmedian(arr[fin]):.3g}/{np.nanmax(arr):.3g}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diviner", type=Path, required=True, help="Diviner Tbol .lbl/.img/.xml")
    parser.add_argument("--m3", type=Path, required=True, help="M³ band-depth mosaic (ENVI .img)")
    parser.add_argument("--lend", type=Path, required=True, help="LEND Sanin-2018 ASCII grid")
    parser.add_argument("--psr", type=Path, required=True, help="LOLA/SPICE PSR mask (GeoTIFF)")
    parser.add_argument("--out", type=Path, required=True, help="conditioning-bundle output dir")
    parser.add_argument("--registry", type=Path, help="Hub registry dir to publish the prior to")
    parser.add_argument("--private-key", type=Path, help="cosign ECDSA-P256 PEM (else generated)")
    parser.add_argument("--public-key-out", type=Path, help="write the generated public key here")
    args = parser.parse_args(argv)

    lend_tif = lend_ascii_to_suppression_geotiff(args.lend, args.out / "_lend_suppression.tif")

    # The four public conditioning rasters, each with its physical conversion + per-product cite.
    inputs = {
        "psr": RasterInput(
            path=args.psr,
            role="psr",
            units="fraction",
            citation="LOLA",
            resampling="average",
        ),
        "diviner_temperature": RasterInput(
            path=args.diviner,
            role="measured_temperature",
            units="K",
            citation="Diviner",
            scale=0.02,
            nodata=-32768.0,
            resampling="average",
        ),
        "lend_suppression": RasterInput(
            path=lend_tif,
            role="neutron_suppression",
            units="suppression_index",
            citation="LEND",
            resampling="bilinear",
        ),
        "m3_band_depth": RasterInput(
            path=args.m3,
            role="band_depth",
            units="band_depth",
            citation="M3",
            nodata=0.0,
            resampling="average",
        ),
    }
    print("ingesting conditioning rasters onto the Shackleton prior grid...", file=sys.stderr)
    layer_set = ingest_conditioning(inputs, grid=SHACKLETON_PRIOR_GRID, crs=SHACKLETON_CRS)
    for name, layer in sorted(layer_set.layers.items()):
        _summarize(name, layer.values)

    materialize_conditioning_bundle(layer_set, args.out)
    bundle = load_conditioning_bundle(args.out)
    prior = build_pds_prior(SHACKLETON_PRIOR_GRID, bundle)
    print(f"fitted {prior.provenance.recipe}: content_hash={prior.content_hash}", file=sys.stderr)
    print(
        f"  WEH mean range {float(prior.mean.min()):.4f}..{float(prior.mean.max()):.4f} "
        f"(unit {prior.metadata.unit})",
        file=sys.stderr,
    )
    for c in prior.provenance.citations:
        print(f"  cite {c.short_name}: source_hash={c.source_hash}", file=sys.stderr)

    if args.registry is not None:
        from astro_mine.prospect.publish import publish_prior

        key = args.private_key.read_bytes() if args.private_key else None
        published = publish_prior(prior, registry_path=args.registry, private_key_pem=key)
        print(f"published {published.reference} -> {published.digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
