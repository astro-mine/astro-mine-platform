"""Shared fixtures: a synthetic, self-contained SPICE kernel set.

CI runs offline, so tests exercise the *real* SPICE pipeline against synthesized inputs —
an LSK + a generated SPK + a text PCK + an FK aliasing ``MOON_ME`` — with
``abcorr="NONE"`` (the minimal SPK carries only the direct target->Moon segment, so
light-time/aberration corrections are out of reach here). The real kernels are fetched
via the ``scripts/`` helpers outside CI.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import spiceypy as sp

from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.spice import Site, clear_kernels

# Minimal leapsecond kernel — enough for str2et over the test epoch.
_LSK_TEXT = """\
\\begindata
DELTET/DELTA_T_A = 32.184
DELTET/K         = 1.657D-3
DELTET/EB        = 1.671D-2
DELTET/M         = ( 6.239996D0 1.99096871D-7 )
DELTET/DELTA_AT  = ( 10, @1972-JAN-1, 32, @2006-JAN-1, 33, @2009-JAN-1,
                     34, @2012-JUL-1, 35, @2015-JUL-1, 37, @2017-JAN-1 )
\\begintext
"""

# Text PCK — simple IAU-style lunar orientation constants (drives the IAU_MOON frame).
_PCK_TEXT = """\
\\begindata
BODY301_POLE_RA  = (  269.9949   0.0031   0. )
BODY301_POLE_DEC = (   66.5392   0.0130   0. )
BODY301_PM       = (   38.3213  13.17635815  -1.4D-12 )
BODY301_RADII    = ( 1737.4 1737.4 1737.4 )
\\begintext
"""

# Frame kernel — alias Core's MOON_ME to IAU_MOON so the Core frame name resolves.
_FK_TEXT = """\
\\begindata
FRAME_MOON_ME            = 1900301
FRAME_1900301_NAME       = 'MOON_ME'
FRAME_1900301_CLASS      = 4
FRAME_1900301_CLASS_ID   = 1900301
FRAME_1900301_CENTER     = 301
TKFRAME_1900301_SPEC     = 'MATRIX'
TKFRAME_1900301_RELATIVE = 'IAU_MOON'
TKFRAME_1900301_MATRIX   = ( 1 0 0  0 1 0  0 0 1 )
\\begintext
"""


@dataclass(frozen=True)
class SyntheticSpice:
    """A furnished synthetic kernel set and the epoch/site it is valid for."""

    directory: Path
    epoch: Epoch  # the window start
    window: EpochWindow  # the 24 h window the SPK covers
    site: Site  # a near-south-pole lunar site (off-pole so azimuth is defined)


def _synthetic_states(body: int, et0: float, ets: list[float]) -> list[list[float]]:
    """Simple circular-ish states (km, km/s) for Sun(10)/Earth(399) about the Moon(301)."""
    states: list[list[float]] = []
    for epoch in ets:
        ang = 2.0 * math.pi * (epoch - et0) / (27.3 * 86400.0)
        if body == 10:
            radius, rate = 1.5e8, ang
        else:
            radius, rate = 3.8e5, ang * 13.4
        x = radius * math.cos(rate)
        y = radius * math.sin(rate)
        z = 0.1 * radius if body == 10 else 0.05 * radius
        vx = -radius * math.sin(rate) * 1e-6
        vy = radius * math.cos(rate) * 1e-6
        states.append([x, y, z, vx, vy, 0.0])
    return states


@pytest.fixture
def synthetic_spice(tmp_path: Path) -> Iterator[SyntheticSpice]:
    """Generate, furnish, and (on teardown) clear a self-contained SPICE kernel set."""
    kernels = tmp_path / "kernels"
    kernels.mkdir(parents=True, exist_ok=True)
    lsk = kernels / "naif.tls"
    lsk.write_text(_LSK_TEXT, encoding="utf-8")
    (kernels / "moon.tpc").write_text(_PCK_TEXT, encoding="utf-8")
    (kernels / "moon_me.tf").write_text(_FK_TEXT, encoding="utf-8")

    sp.furnsh(str(lsk))
    et0 = float(sp.str2et("2025-06-21T00:00:00"))
    ets = [et0 + i * 3600.0 for i in range(25)]  # 24 h, hourly
    spk = kernels / "synth.bsp"
    handle = sp.spkopn(str(spk), "synthetic rfc-0002 test kernel", 0)
    for body in (10, 399):
        sp.spkw09(
            handle,
            body,
            301,
            "J2000",
            ets[0],
            ets[-1],
            "synthetic",
            3,
            len(ets),
            _synthetic_states(body, et0, ets),
            ets,
        )
    sp.spkcls(handle)
    sp.furnsh(str(spk))
    sp.furnsh(str(kernels / "moon.tpc"))
    sp.furnsh(str(kernels / "moon_me.tf"))

    fixture = SyntheticSpice(
        directory=kernels,
        epoch=Epoch(tdb_seconds=et0, scale=TimeScale.TDB),
        window=EpochWindow(
            start=Epoch(tdb_seconds=et0, scale=TimeScale.TDB),
            end=Epoch(tdb_seconds=ets[-1], scale=TimeScale.TDB),
        ),
        site=Site.lunar_from_latlon(-89.0, 0.0),
    )
    try:
        yield fixture
    finally:
        clear_kernels()
