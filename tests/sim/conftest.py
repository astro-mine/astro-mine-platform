"""Shared fixtures — a synthetic, self-contained SPICE kernel set (RFC-0002).

CI runs offline, so the SPICE-backed frame bridge (RM-P0-SIM-04's cross-frame coupling boundary) is
exercised against the **real** ``astro-mine-spice`` pipeline driven by *synthesized* kernels rather
than a downloaded NAIF set — the same technique ``astro-mine-spice``'s own suite uses.

The bridge only ever asks for an **orientation** (``pxform`` J2000 ↔ MOON_ME), so the kernel set is
just two small text kernels: a PCK carrying the IAU lunar orientation constants (which drive the
``IAU_MOON`` frame) and an FK aliasing Core's ``MOON_ME`` onto it. No SPK is needed (no body
*positions* are queried), and no leapsecond kernel is needed (a Core ``Epoch`` is TDB seconds, which
is SPICE ET directly). The rotation this yields is genuine SPICE output, not a stand-in.

The pool is process-global, so the fixture clears it on teardown; a test that wants the
*unfurnished* behaviour (a bridge that must fail loudly) simply does not request the fixture.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import spiceypy as sp

from astro_mine.core.units import Epoch, TimeScale
from astro_mine.sim.coupling import SpiceFrameBridge

# Text PCK — the IAU lunar orientation constants that define the IAU_MOON body-fixed frame.
_PCK_TEXT = """\
\\begindata
BODY301_POLE_RA  = (  269.9949   0.0031   0. )
BODY301_POLE_DEC = (   66.5392   0.0130   0. )
BODY301_PM       = (   38.3213  13.17635815  -1.4D-12 )
BODY301_RADII    = ( 1737.4 1737.4 1737.4 )
\\begintext
"""

# Frame kernel — alias Core's MOON_ME onto IAU_MOON so the Core frame name resolves in SPICE.
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

#: An epoch well inside the (analytic, unbounded) IAU orientation model — mid-2025, TDB.
SPICE_EPOCH = Epoch(tdb_seconds=8.0e8, scale=TimeScale.TDB)


@pytest.fixture
def spice_kernels(tmp_path: Path) -> Iterator[Path]:
    """Furnish the synthetic lunar-orientation kernel set; clear the pool on teardown."""
    kernels = tmp_path / "kernels"
    kernels.mkdir(parents=True, exist_ok=True)
    (kernels / "moon.tpc").write_text(_PCK_TEXT, encoding="utf-8")
    (kernels / "moon_me.tf").write_text(_FK_TEXT, encoding="utf-8")
    sp.furnsh(str(kernels / "moon.tpc"))
    sp.furnsh(str(kernels / "moon_me.tf"))
    try:
        yield kernels
    finally:
        sp.kclear()  # the SPICE pool is process-global — never leak it into the next test


@pytest.fixture
def spice_bridge(spice_kernels: Path) -> SpiceFrameBridge:
    """The default SPICE frame bridge, over a furnished pool (so it resolves rather than raises)."""
    return SpiceFrameBridge()
