"""SPICE epochs, frames, and Sun/Earth geometry (RFC-0002).

Driven against the ``synthetic_spice`` fixture (a self-contained LSK + generated SPK +
text PCK + FK kernel set), with ``abcorr="NONE"`` — the minimal SPK carries only the
direct target->Moon segment, so light-time/aberration corrections (which need the
Moon-relative-to-SSB chain) are out of reach here. Production callers keep the ``"LT+S"``
default.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astro_mine.core.units import (
    EARTH,
    INERTIAL_J2000,
    MOON,
    MOON_BODY_FIXED,
    SUN,
    Epoch,
    FrameClass,
    ReferenceFrame,
    TimeScale,
)
from astro_mine.spice import (
    Site,
    SpiceGeometryError,
    SpiceKernelError,
    body_geometry,
    body_position,
    earth_geometry,
    epoch_from_utc,
    epoch_range,
    et,
    frame_transform,
    sun_geometry,
)

_GEOMETRIC = "NONE"


def test_et_is_tdb_seconds() -> None:
    epoch = Epoch(tdb_seconds=1.234e8, scale=TimeScale.TDB)
    assert et(epoch) == 1.234e8


def test_epoch_from_utc(synthetic_spice) -> None:
    epoch = epoch_from_utc("2025-06-21T00:00:00")
    assert epoch.scale is TimeScale.TDB
    # str2et of the same calendar instant the fixture was built from.
    assert epoch.tdb_seconds == pytest.approx(synthetic_spice.epoch.tdb_seconds)


def test_epoch_range_is_half_open(synthetic_spice) -> None:
    window = synthetic_spice.window
    step = 6.0 * 3600.0
    epochs = list(epoch_range(window, step))
    assert len(epochs) == 4  # 0, 6, 12, 18 h — 24 h end excluded (half-open)
    assert all(e.scale is TimeScale.TDB for e in epochs)
    assert all(window.start.tdb_seconds <= e.tdb_seconds < window.end.tdb_seconds for e in epochs)
    assert epochs[0].tdb_seconds == window.start.tdb_seconds


def test_epoch_range_rejects_nonpositive_step(synthetic_spice) -> None:
    with pytest.raises(SpiceGeometryError, match="step_s must be positive"):
        list(epoch_range(synthetic_spice.window, 0.0))


def test_body_position_is_metres(synthetic_spice) -> None:
    pos = body_position(SUN, MOON, synthetic_spice.epoch, frame=INERTIAL_J2000, abcorr=_GEOMETRIC)
    assert len(pos) == 3
    # Synthetic Sun sits ~1.5e8 km from the Moon → ~1.5e11 m (SI at the boundary).
    assert np.linalg.norm(pos) == pytest.approx(1.5e8 * 1000.0, rel=0.2)


def test_frame_transform_is_a_rotation(synthetic_spice) -> None:
    rot = frame_transform(INERTIAL_J2000, MOON_BODY_FIXED, synthetic_spice.epoch)
    assert rot.shape == (3, 3)
    # Orthonormal, proper rotation.
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)


def test_sun_geometry_fields(synthetic_spice) -> None:
    geom = sun_geometry(synthetic_spice.site, synthetic_spice.epoch, abcorr=_GEOMETRIC)
    assert geom.target == SUN
    assert geom.range_m > 0.0
    assert -90.0 <= geom.elevation_deg <= 90.0
    assert 0.0 <= geom.azimuth_deg < 360.0
    # direction is a unit vector consistent with position / range.
    assert np.linalg.norm(geom.direction) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(
        np.array(geom.position_m) / geom.range_m, np.array(geom.direction), atol=1e-9
    )


def test_earth_geometry_fields(synthetic_spice) -> None:
    geom = earth_geometry(synthetic_spice.site, synthetic_spice.epoch, abcorr=_GEOMETRIC)
    assert geom.target == EARTH
    assert geom.range_m > 0.0
    assert 0.0 <= geom.azimuth_deg < 360.0


def test_polar_site_azimuth_is_degenerate(synthetic_spice) -> None:
    # A site exactly on the spin axis has no defined east/north → azimuth pinned to 0.
    pole_site = Site.lunar_from_latlon(-90.0, 0.0)
    geom = sun_geometry(pole_site, synthetic_spice.epoch, abcorr=_GEOMETRIC)
    assert geom.azimuth_deg == 0.0


def test_body_geometry_matches_position(synthetic_spice) -> None:
    site = synthetic_spice.site
    epoch = synthetic_spice.epoch
    geom = body_geometry(SUN, site, epoch, abcorr=_GEOMETRIC)
    target = np.array(body_position(SUN, site.body, epoch, frame=site.frame, abcorr=_GEOMETRIC))
    expected = target - np.array(site.position_m)
    np.testing.assert_allclose(np.array(geom.position_m), expected, rtol=1e-9)


def test_body_position_unknown_body_raises_geometry_error(synthetic_spice) -> None:
    # An untranslatable body name is a resolution failure, not a coverage gap.
    with pytest.raises(SpiceGeometryError, match="NOTABODY"):
        body_position("NOTABODY", MOON, synthetic_spice.epoch, abcorr=_GEOMETRIC)


def test_body_position_outside_coverage_raises_kernel_error(synthetic_spice) -> None:
    # An epoch past the synthetic SPK's 24 h span → SPICE(SPKINSUFFDATA) → kernel error.
    far = Epoch(tdb_seconds=synthetic_spice.window.end.tdb_seconds + 1.0e9, scale=TimeScale.TDB)
    with pytest.raises(SpiceKernelError):
        body_position(SUN, MOON, far, frame=INERTIAL_J2000, abcorr=_GEOMETRIC)


def test_frame_transform_unknown_frame_raises_geometry_error(synthetic_spice) -> None:
    bogus = ReferenceFrame(name="NOSUCHFRAME", frame_class=FrameClass.BODY_FIXED, center=MOON)
    with pytest.raises(SpiceGeometryError, match="NOSUCHFRAME"):
        frame_transform(bogus, INERTIAL_J2000, synthetic_spice.epoch)


def test_epoch_from_utc_bad_string_raises_geometry_error(synthetic_spice) -> None:
    with pytest.raises(SpiceGeometryError, match="not-a-real-epoch"):
        epoch_from_utc("not-a-real-epoch")


def test_body_reference_radii_are_shared() -> None:
    # RFC-0002: body reference radii live in astro_mine.spice; consumers re-import them.
    from astro_mine.spice import EARTH_RADIUS_M, MOON_RADIUS_M

    assert EARTH_RADIUS_M == 6_371_000.0
    assert MOON_RADIUS_M == 1_737_400.0


def test_site_lunar_from_latlon_on_sphere() -> None:
    site = Site.lunar_from_latlon(-89.0, 12.0)
    assert site.body == MOON
    assert site.frame == MOON_BODY_FIXED
    from astro_mine.spice import MOON_RADIUS_M

    assert np.linalg.norm(site.position_m) == pytest.approx(MOON_RADIUS_M)
    # -89° latitude → just shy of the south pole (negative z).
    assert site.position_m[2] == pytest.approx(MOON_RADIUS_M * math.sin(math.radians(-89.0)))
