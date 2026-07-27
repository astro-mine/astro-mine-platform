"""Shared fixtures: synthetic stand-ins for the real LOLA DEM and SPICE kernels.

CI runs offline, so tests exercise the *real* pipelines against synthesized inputs — a
GeoTIFF in a lunar geographic CRS for terrain ingest, and a self-contained SPICE kernel
set (LSK + a generated SPK + a text PCK + an FK aliasing ``MOON_ME``) for the geometry
backbone. The real products are fetched via the ``scripts/`` helpers outside CI.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.crs
import spiceypy as sp
from rasterio.transform import from_bounds

from astro_mine.core.units import Epoch, EpochWindow, FrameClass, ReferenceFrame, TimeScale
from astro_mine.spice import Site, clear_kernels
from astro_mine.worlds.crs import lunar_geographic_proj4

# A small south-polar extent (degrees), coarse enough to stay fast at a 2 km test grid.
_WIDTH, _HEIGHT = 40, 40
_WEST, _SOUTH, _EAST, _NORTH = -2.0, -87.0, 2.0, -86.0
_NODATA = -32768.0


def _elevation() -> np.ndarray:
    """A tilted plane plus a Gaussian peak — gives non-trivial slope/aspect/roughness."""
    yy, xx = np.mgrid[0:_HEIGHT, 0:_WIDTH].astype(np.float64)
    peak = 200.0 * np.exp(-(((xx - 20.0) ** 2 + (yy - 20.0) ** 2) / 60.0))
    return (1000.0 + 6.0 * xx + 4.0 * yy + peak).astype(np.float32)


def _write_dem(path: Path, *, with_crs: bool, with_void: bool) -> Path:
    elev = _elevation()
    if with_void:
        elev[8:13, 8:13] = _NODATA
    crs = rasterio.crs.CRS.from_proj4(lunar_geographic_proj4()) if with_crs else None
    transform = from_bounds(_WEST, _SOUTH, _EAST, _NORTH, _WIDTH, _HEIGHT)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=_WIDTH,
        height=_HEIGHT,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=_NODATA,
    ) as dst:
        dst.write(elev, 1)
    return path


@pytest.fixture
def synthetic_dem(tmp_path: Path) -> Path:
    """A lunar-geographic DEM with an explicit CRS and a void patch."""
    return _write_dem(tmp_path / "src_dem.tif", with_crs=True, with_void=True)


@pytest.fixture
def crsless_dem(tmp_path: Path) -> Path:
    """A DEM with no CRS — must be rejected at ingest."""
    return _write_dem(tmp_path / "no_crs.tif", with_crs=False, with_void=False)


# --- synthetic SPICE kernels (RM-P0-WORLDS-02) -----------------------------------

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
    handle = sp.spkopn(str(spk), "synthetic worlds-02 test kernel", 0)
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


# --- synthetic Mars SPICE kernels (RM-P1-WORLDS-11) ------------------------------

# Text PCK — IAU-style Mars orientation constants (drives the standard IAU_MARS frame).
_MARS_PCK_TEXT = """\
\\begindata
BODY499_POLE_RA  = (  317.68143  -0.1061      0. )
BODY499_POLE_DEC = (   52.88650  -0.0609      0. )
BODY499_PM       = (  176.630   350.89198226  0. )
BODY499_RADII    = ( 3396.19 3396.19 3376.20 )
\\begintext
"""


@pytest.fixture
def synthetic_mars_spice(tmp_path: Path) -> Iterator[SyntheticSpice]:
    """A self-contained Mars kernel set: LSK + BODY499 PCK (IAU_MARS) + Sun-about-Mars SPK."""
    kernels = tmp_path / "mars_kernels"
    kernels.mkdir(parents=True, exist_ok=True)
    lsk = kernels / "naif.tls"
    lsk.write_text(_LSK_TEXT, encoding="utf-8")
    (kernels / "mars.tpc").write_text(_MARS_PCK_TEXT, encoding="utf-8")

    sp.furnsh(str(lsk))
    et0 = float(sp.str2et("2025-06-21T00:00:00"))
    ets = [et0 + i * 3600.0 for i in range(25)]
    spk = kernels / "mars.bsp"
    handle = sp.spkopn(str(spk), "synthetic worlds-11 mars test kernel", 0)
    # Sun (10) about Mars (499): ~1.52 AU circular-ish, Mars year ~687 d.
    states = []
    for epoch in ets:
        ang = 2.0 * math.pi * (epoch - et0) / (687.0 * 86400.0)
        radius = 2.279e8  # km, ~1.52 AU
        states.append(
            [radius * math.cos(ang), radius * math.sin(ang), 0.08 * radius, 0.0, 0.0, 0.0]
        )
    sp.spkw09(handle, 10, 499, "J2000", ets[0], ets[-1], "synthetic", 3, len(ets), states, ets)
    sp.spkcls(handle)
    sp.furnsh(str(spk))
    sp.furnsh(str(kernels / "mars.tpc"))

    fixture = SyntheticSpice(
        directory=kernels,
        epoch=Epoch(tdb_seconds=et0, scale=TimeScale.TDB),
        window=EpochWindow(
            start=Epoch(tdb_seconds=et0, scale=TimeScale.TDB),
            end=Epoch(tdb_seconds=ets[-1], scale=TimeScale.TDB),
        ),
        site=Site(
            body="MARS",
            position_m=(
                3.3895e6 * math.cos(math.radians(10.0)),
                0.0,
                3.3895e6 * math.sin(math.radians(10.0)),
            ),
            frame=ReferenceFrame(name="IAU_MARS", frame_class=FrameClass.BODY_FIXED, center="MARS"),
        ),
    )
    try:
        yield fixture
    finally:
        clear_kernels()
