#!/usr/bin/env python3
"""Fetch real NAIF SPICE kernels and write a meta-kernel for the geometry backbone.

CI runs offline against the synthetic kernel set built in ``tests/conftest.py``; this
script is the documented way to obtain the *real* generic kernels so you can compute
actual lunar Sun/Earth geometry locally:

    python scripts/fetch_spice_kernels.py --dir data/spice

It downloads a leapseconds kernel (LSK), the planetary-constants kernel (PCK, lunar
orientation + the ``MOON_ME``/``MOON_PA`` body-fixed frames), the Moon frame kernel
(FK), and a planetary ephemeris (SPK, e.g. DE440) — then writes ``metakernel.tm``
listing them with a ``PATH_VALUES`` pointing at the download directory. Furnish it via
the same API the tests exercise:

    from astro_mine.spice import kernel_pool, sun_geometry, Site, epoch_from_utc
    with kernel_pool("data/spice/metakernel.tm"):
        site = Site.lunar_from_latlon(-89.9, 0.0)
        geom = sun_geometry(site, epoch_from_utc("2025-06-21T00:00:00"))

Source: the NAIF generic-kernels archive. Verify the current kernel names/URLs against
the archive before relying on them — NAIF revises the latest PCK/SPK over time.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

NAIF_GENERIC = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"

# (subdirectory under generic_kernels/, filename). Confirm against the live archive.
KERNELS = [
    ("lsk/", "naif0012.tls"),  # leapseconds
    ("pck/", "moon_pa_de440_200625.bpc"),  # lunar orientation (binary PCK)
    ("pck/", "pck00011.tpc"),  # planetary constants (radii, generic orientation)
    ("fk/satellites/", "moon_de440_250416.tf"),  # MOON_ME / MOON_PA frame definitions
    ("spk/planets/", "de440.bsp"),  # planetary + lunar ephemeris (large)
]

_METAKERNEL_NAME = "metakernel.tm"


def fetch(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"skip (exists) {out}", file=sys.stderr)
        return
    print(f"downloading {url} -> {out}", file=sys.stderr)
    urllib.request.urlretrieve(url, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


def write_metakernel(directory: Path, filenames: list[str]) -> Path:
    """Write a meta-kernel that furnishes ``filenames`` relative to ``directory``."""
    listing = "\n".join(f"                        '$KERNELS/{name}'" for name in filenames)
    text = (
        "\\begindata\n"
        "    PATH_VALUES     = ( '" + str(directory.resolve()) + "' )\n"
        "    PATH_SYMBOLS    = ( 'KERNELS' )\n"
        "    KERNELS_TO_LOAD = (\n" + listing + "\n                      )\n"
        "\\begintext\n"
    )
    path = directory / _METAKERNEL_NAME
    path.write_text(text, encoding="utf-8")
    print(f"wrote meta-kernel {path}", file=sys.stderr)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data") / "spice")
    parser.add_argument(
        "--base-url", default=NAIF_GENERIC, help="override the NAIF archive base URL"
    )
    args = parser.parse_args(argv)

    filenames: list[str] = []
    for subdir, name in KERNELS:
        fetch(args.base_url + subdir + name, args.dir / name)
        filenames.append(name)
    write_metakernel(args.dir, filenames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
