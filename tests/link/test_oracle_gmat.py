"""RM-P0-LINK-05 -- the live external-oracle pass-time regression against NASA GMAT.

The flight-grade half of the contact-window validation (link.md sec 10): Link's
:func:`~astro_mine.link.windows.dsn_contact_windows` is regressed against a GMAT
``ContactLocator`` run of the *same* station + spacecraft within an explicit rise/set budget.
GMAT runs **live** -- this test loads the committed ``contact_locator.script``, runs GMAT via
``gmat-run``, reads both the ContactLocator report (the oracle) and the exported SPK (the shared
ephemeris), then recomputes the windows from that SPK + GMAT's own DE/LSK/PCK kernels and
cross-checks the pass times.

It is a required gate in CI (where ``setup-gmat`` provisions GMAT + exports ``GMAT_ROOT`` and the
``gmat`` dependency group provides ``gmat-run``), and **skips** wherever GMAT or ``gmat-run`` is
unavailable -- a Python 3.13 dev checkout, or a contributor without a GMAT install. The
dependency-free comparator itself (``cross_check_pass_times``) is always exercised by
``test_cache.py``; this mirrors the Sim orbital regression (RM-P0-SIM-10).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from astro_mine import spice
from astro_mine.core.units import EARTH, EpochWindow, FrameClass, ReferenceFrame
from astro_mine.link.cache import assert_within_budget, cross_check_pass_times
from astro_mine.link.windows import GroundStation, SpiceTopocentric, dsn_contact_windows

pytest.importorskip(
    "gmat_run", reason="gmat-run not installed (needs Python <3.13 + a GMAT install)"
)

_SCRIPT = Path(__file__).parent / "data" / "contact_locator.script"

# Fixed by the committed script: the station, the propagated spacecraft's NAIF id, the epoch
# window, and the elevation mask. Link reproduces the same scenario from the exported SPK.
_STATION = ("DSS", 35.3, 243.1)  # name, planetocentric lat_deg, lon_deg
_MIN_ELEVATION_DEG = 10.0
_TARGET = "-10001"  # Sat.NAIFId in the script / exported SPK
_WINDOW_UTC = ("01 Jun 2004 00:05:00.000", "01 Jun 2004 23:55:00.000")

# The worst observed rise/set delta is ~7 s, bounded by the Phase-0 spherical-Earth station +
# analytic IAU_EARTH frame (light-time/aberration are OFF on both sides). 20 s carries ~2.7x
# margin while still tripping any real regression: a wrong frame drifts minutes, a wrong kernel
# or body id changes the pass count, and a km<->m unit slip is astronomically off.
_BUDGET_S = 20.0

_CONTACT_RE = re.compile(
    r"(\d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2}\.\d+)\s+(\d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2}\.\d+)"
)


def _gmat_data_kernels() -> list[Path]:
    """GMAT's bundled DE/PCK/LSK kernels — the ones its own run used. Skips if unavailable."""
    root = os.environ.get("GMAT_ROOT")
    if not root:
        pytest.skip("GMAT_ROOT is not set; the live oracle needs a provisioned GMAT install")
    data = Path(root) / "data"
    kernels = [
        data / "time" / "SPICELeapSecondKernel.tls",
        data / "planetary_coeff" / "SPICEPlanetaryConstantsKernel.tpc",
        data / "planetary_ephem" / "spk" / "DE405AllPlanets.bsp",
    ]
    missing = [str(k) for k in kernels if not k.is_file()]
    if missing:
        pytest.skip(f"expected GMAT R2026a kernels not found: {missing}")
    return kernels


@pytest.fixture(scope="module")
def gmat_contacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[tuple[str, str]], Path]:
    """Run GMAT once for the module: returns (rise/set UTC pairs, exported SPK path).

    Skips (rather than fails) when no GMAT install is discoverable, so the test is required only
    where GMAT is actually provisioned (CI via ``setup-gmat``)."""
    from gmat_run import Mission
    from gmat_run.errors import GmatNotFoundError

    try:
        mission = Mission.load(_SCRIPT)
    except GmatNotFoundError as exc:  # gmat-run installed, but no GMAT install on this machine
        pytest.skip(f"GMAT install not found: {exc}")

    work_dir = tmp_path_factory.mktemp("gmat")
    mission.run(working_dir=work_dir)  # relative script outputs land in work_dir
    report = work_dir / "contacts.txt"
    spk = work_dir / "sat.bsp"
    if not report.is_file() or not spk.is_file():
        pytest.skip(f"GMAT produced no contact report / SPK in {work_dir}")

    passes = _CONTACT_RE.findall(report.read_text())
    return passes, spk


def test_dsn_windows_match_gmat_contact_locator(
    gmat_contacts: tuple[list[tuple[str, str]], Path],
) -> None:
    utc_passes, spk = gmat_contacts
    assert len(utc_passes) >= 3  # the reference LEO day yields ~6 station passes

    with spice.kernel_pool(*_gmat_data_kernels(), spk):
        gmat_passes = [
            (spice.epoch_from_utc(rise).tdb_seconds, spice.epoch_from_utc(setk).tdb_seconds)
            for rise, setk in utc_passes
        ]
        window = EpochWindow(
            start=spice.epoch_from_utc(_WINDOW_UTC[0]),
            end=spice.epoch_from_utc(_WINDOW_UTC[1]),
        )
        frame = ReferenceFrame(name="IAU_EARTH", frame_class=FrameClass.BODY_FIXED, center=EARTH)
        station = GroundStation.from_latlon(
            _STATION[0], _STATION[1], _STATION[2], min_elevation_deg=_MIN_ELEVATION_DEG, frame=frame
        )
        # abcorr='NONE' matches GMAT's UseLightTimeDelay/UseStellarAberration = false; a fine step
        # + bisection refinement puts Link's rise/set precision well under the oracle budget.
        link_passes = dsn_contact_windows(
            station,
            _TARGET,
            window,
            5.0,
            topocentric=SpiceTopocentric(abcorr="NONE"),
            refine_s=0.01,
        )

    report = cross_check_pass_times(
        link_passes, gmat_passes, tolerance_s=_BUDGET_S, name="dsn-vs-gmat"
    )
    assert_within_budget(report)
