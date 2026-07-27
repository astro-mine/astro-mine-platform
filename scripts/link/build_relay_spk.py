#!/usr/bin/env python3
"""Write the anchor relay orbiter's SPK from its pinned orbital elements (RM-P0-LINK-04).

The anchor's relay is a **notional** spacecraft, so NAIF has no SPK for it. A real mission would
get one from flight dynamics (GMAT's ``EphemerisFile``, exactly as the RM-P0-LINK-05 oracle
regression consumes); this script is the offline stand-in, so the anchor ContactPlan is
reproducible from public kernels + pinned data alone, with no proprietary tool in the chain.

The propagation is **SPICE's own** two-body routine (``spiceypy.conics``, i.e. CSPICE
``conics_c``) applied to the pinned classical elements below, sampled onto a uniform grid and
written as a type-9 (Lagrange, unequal steps) SPK segment relative to the Moon in J2000. It lives
in ``scripts/``, not in the library: Link consumes ephemerides through the injected
``EphemerisProvider`` seam and never propagates an orbit itself (link.md §2.2; RFC-0002).

Deterministic: the same elements + the same grid produce byte-identical SPK bytes, so the file's
content hash - folded into the published plan's provenance - is stable across machines.

Usage (from the astro-mine-link project environment):

    uv run python scripts/build_relay_spk.py \
        --metakernel /path/to/metakernel.tm --out /path/to/relay_orbiter.bsp
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import spiceypy

from astro_mine.link.anchor import ANCHOR_EPOCH_WINDOW, ANCHOR_RELAY_TARGET

_MOON = 301

#: The Moon's gravitational parameter (km^3/s^2) — the DE440 value (NAIF ``gm_de440.tpc``,
#: ``BODY301_GM``). Pinned rather than read from the kernel pool on purpose: the generic PCK the
#: anchor furnishes carries radii but not GM, and a value that depended on *which* GM kernel was
#: loaded would make the relay ephemeris — and so the published plan digest — drift with the
#: caller's kernel set.
MOON_GM_KM3_S2 = 4902.800118

#: The pinned orbital elements of the anchor relay: a circular **polar** orbit at 500 km altitude
#: (period ~2.6 h, so a PSR agent sees several short passes and several blackouts each rotation
#: across the anchor's 30-day mission window - the intermittency the comms-denied benchmark needs).
#: Two-body only, so the plane holds inertially fixed over the mission - a deterministic *notional*
#: orbit, not a mission-planning ephemeris. Epoch is the anchor window's start; angles in degrees;
#: altitude in km above the mean lunar radius.
ALTITUDE_KM = 500.0
ECCENTRICITY = 0.0
INCLINATION_DEG = 90.0
RAAN_DEG = 0.0
ARG_PERIAPSIS_DEG = 0.0
MEAN_ANOMALY_DEG = 0.0

#: The sampling grid of the written SPK. 60 s matches the anchor's coarse rise/set step; type-9
#: Lagrange interpolation (degree 7) over a 60 s grid is far finer than the 20 s pass-time budget
#: the RM-P0-LINK-05 oracle regression holds Link to.
STEP_S = 60.0
DEGREE = 7
#: Pad the window on both sides so interpolation never runs off the end of the segment.
PAD_S = 3600.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metakernel", required=True, type=Path, help="SPICE meta-kernel (.tm).")
    parser.add_argument("--out", required=True, type=Path, help="SPK file to write (.bsp).")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    spiceypy.furnsh(str(args.metakernel))
    try:
        mu = MOON_GM_KM3_S2
        radius_km = float(spiceypy.bodvrd("MOON", "RADII", 3)[1][0])
        periapsis_km = radius_km + ALTITUDE_KM
        t0 = ANCHOR_EPOCH_WINDOW.start.tdb_seconds
        elements = [
            periapsis_km,
            ECCENTRICITY,
            math.radians(INCLINATION_DEG),
            math.radians(RAAN_DEG),
            math.radians(ARG_PERIAPSIS_DEG),
            math.radians(MEAN_ANOMALY_DEG),
            t0,
            mu,
        ]

        start = t0 - PAD_S
        end = ANCHOR_EPOCH_WINDOW.end.tdb_seconds + PAD_S
        epochs = [start + i * STEP_S for i in range(int((end - start) / STEP_S) + 1)]
        states = [list(spiceypy.conics(elements, et)) for et in epochs]

        handle = spiceypy.spkopn(str(out), "ASTRO-MINE ANCHOR RELAY", 0)
        try:
            spiceypy.spkw09(
                handle,
                int(ANCHOR_RELAY_TARGET),
                _MOON,
                "J2000",
                epochs[0],
                epochs[-1],
                "ANCHOR RELAY ORBITER",
                DEGREE,
                len(epochs),
                states,
                epochs,
            )
        finally:
            spiceypy.spkcls(handle)
    finally:
        spiceypy.kclear()

    print(f"wrote {out} ({out.stat().st_size} bytes, {len(epochs)} states)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
