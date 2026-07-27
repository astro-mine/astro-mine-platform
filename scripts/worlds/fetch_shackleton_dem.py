#!/usr/bin/env python3
"""Fetch the real LOLA south-polar DEM for the Shackleton-de Gerlache anchor world.

The multi-GB LOLA DEM is **not** committed to the repo and **not** fetched in CI (which runs
offline against the synthetic fixture in ``tests/conftest.py``). This script obtains the real
product so you can build the actual ``shackleton-de-gerlache-v1`` world locally:

    python scripts/fetch_shackleton_dem.py --dir data/dem

That downloads the PDS ``.img`` (raw elevation in **km**, on a lunar polar-stereographic sphere,
R=1737.4 km) and its ``.lbl`` label. The raw product is not metre-scaled and is far too large to
ingest whole (30336x30336 @ 5 m), so build the world with the recipe that downsamples + scales
km->m and runs the full pipeline:

    python scripts/build_shackleton_anchor.py --convert
        --raw-dem data/dem/ldem_875s_5m_float.lbl --metakernel data/spice/metakernel.tm
        --resolution-m 120 --out out/shackleton

Source: the LOLA (Lunar Orbiter Laser Altimeter) GDR archive at the NASA PDS Geosciences Node.
``LDEM_875S_5M`` covers 87.5-90S at 5 m/px — the Shackleton-de Gerlache ridge. Verify the current
product name/URL against the PDS archive before relying on it — archive paths change over time.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# The 87.5S 5 m/px LOLA polar DEM (PDS LRO-L-LOLA-4-GDR-V1.0). Confirm against the live archive.
PDS_LOLA_FLOAT_IMG = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/"
    "data/lola_gdr/polar/float_img/"
)
DEFAULT_PRODUCT = "ldem_875s_5m_float"  # fetched as .img (data) + .lbl (PDS label)


def fetch(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {out}", file=sys.stderr)
    urllib.request.urlretrieve(url, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=Path("data") / "dem", help="directory for the .img + .lbl"
    )
    parser.add_argument(
        "--product", default=DEFAULT_PRODUCT, help="LDEM product stem (verify against PDS)"
    )
    parser.add_argument(
        "--base-url", default=PDS_LOLA_FLOAT_IMG, help="override the PDS float_img/ directory URL"
    )
    args = parser.parse_args(argv)
    for ext in (".img", ".lbl"):
        fetch(args.base_url + args.product + ext, args.dir / (args.product + ext))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
