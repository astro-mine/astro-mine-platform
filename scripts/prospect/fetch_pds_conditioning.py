#!/usr/bin/env python3
"""Fetch the public conditioning rasters for the real PDS raster-ingest prior (RM-P1-PROSPECT-12).

The conditioning rasters are **not** committed to the repo and **not** fetched in CI (which runs
offline against synthetic fixtures in ``tests/test_pds_ingest.py``). This script obtains the real
public products so you can build the actual ``shackleton_water_ice_pds_v1`` prior locally:

    python scripts/fetch_pds_conditioning.py --dir data/pds
    python scripts/build_pds_prior.py \
        --diviner data/pds/diviner/dgdr_tbol_avg_pols_20100107n_240_img.lbl \
        --m3       data/pds/m3/0416_OP2A_out_magnetotail_mosaic.img \
        --lend     data/pds/lend/lend_sanin2018_pss.txt \
        --psr      <worlds world bundle>/illumination/psr_mask.tif \
        --out      data/pds/conditioning --registry files/hub-registry

The PSR mask is produced upstream by Worlds (RM-P1-WORLDS-14 illumination component of the
``shackleton-de-gerlache-v1`` world bundle, itself built from the LOLA DEM + SPICE) — see
``astro-mine-worlds/scripts/build_shackleton_anchor.py``; it is not fetched here.

Sources (all account-free HTTP; **verify against the live archives** — PDS/Zenodo paths change):
- **Diviner** — NASA PDS Geosciences Node, LRO Diviner GDR L3 south-polar bolometric temperature
  (Tbol), 240 m polar-stereographic; ``.img`` + PDS3 ``.lbl`` + PDS4 ``.xml`` labels.
- **M³** — Zenodo record 10608904 (Lu et al. 2024, GRL): south-polar surficial-hydration band-depth
  mosaics (ENVI ``.img``/``.hdr``), ~140 m polar-stereographic; the ``Figure 3`` mosaics are used.
- **LEND** — Elsevier open supplement to Sanin et al. 2018 (Planet. Space Sci. 162): a 3° global
  ASCII grid of CSETN collimated epithermal-neutron counts/background/exposure.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# LRO Diviner GDR L3 south-polar Tbol (K = DN*0.02), 240 m polar-stereo. Confirm against PDS.
DIVINER_BASE = (
    "https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/"
    "data_derived_gdr_l3/2010/polar/img/"
)
DIVINER_PRODUCT = "dgdr_tbol_avg_pols_20100107n_240_img"

# Zenodo 10608904 (Lu et al. 2024, GRL): M³ south-polar hydration band-depth mosaics. Confirm.
M3_ZIP_URL = "https://zenodo.org/api/records/10608904/files/data.zip/content"
M3_MOSAICS = (
    "Figure 3/0416_OP2A_out_magnetotail_mosaic",
    "Figure 3/0209_OP1B_in_magnetotail_mosaic",
)

# Sanin et al. 2018 (PSS 162) open Elsevier supplement — 3° global LEND epithermal grid. Confirm.
LEND_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S0032063317300430-mmc1.txt"


def _fetch(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {out}", file=sys.stderr)
    urllib.request.urlretrieve(url, out)
    print(f"  wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data") / "pds", help="download root")
    args = parser.parse_args(argv)

    for ext in ("img", "lbl", "xml"):
        _fetch(
            DIVINER_BASE + f"{DIVINER_PRODUCT}.{ext}",
            args.dir / "diviner" / f"{DIVINER_PRODUCT}.{ext}",
        )

    _fetch(LEND_URL, args.dir / "lend" / "lend_sanin2018_pss.txt")

    m3_zip = args.dir / "m3" / "m3_lu2024_grl.zip"
    _fetch(M3_ZIP_URL, m3_zip)
    with zipfile.ZipFile(m3_zip) as zf:
        for stem in M3_MOSAICS:
            for ext in (".img", ".hdr"):
                member = stem + ext
                (args.dir / "m3" / Path(member).name).write_bytes(zf.read(member))
                print(f"  extracted {member}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
