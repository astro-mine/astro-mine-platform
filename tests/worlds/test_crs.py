"""Lunar CRS (RM-P0-WORLDS-01)."""

from __future__ import annotations

import pytest

from astro_mine.core.units import MOON, PlanetaryCRS
from astro_mine.worlds.crs import (
    LUNAR_SOUTH_POLAR_STEREOGRAPHIC,
    MOON_RADIUS_M,
    to_lonlat,
    to_rasterio_crs,
)


def test_canonical_crs_is_an_explicit_lunar_polar_stereographic() -> None:
    crs = LUNAR_SOUTH_POLAR_STEREOGRAPHIC
    assert crs.body == MOON
    assert crs.reference_radius_m == MOON_RADIUS_M
    assert crs.projection is not None
    assert "stere" in crs.projection and "+R=1737400" in crs.projection


def test_to_rasterio_crs_projected() -> None:
    rio_crs = to_rasterio_crs(LUNAR_SOUTH_POLAR_STEREOGRAPHIC)
    assert rio_crs.is_projected
    # The Moon sphere radius survives the round-trip into the rasterio/PROJ CRS.
    assert "1737400" in rio_crs.to_proj4()


def test_to_rasterio_crs_geographic_when_no_projection() -> None:
    geographic = PlanetaryCRS(
        body=MOON, body_fixed_frame="MOON_ME", reference_radius_m=MOON_RADIUS_M
    )
    rio_crs = to_rasterio_crs(geographic)
    assert rio_crs.is_geographic


def test_to_lonlat_inverts_the_polar_stereographic_projection() -> None:
    """The south pole is the grid origin; +y is the central meridian (RM-P1-WORLDS-16)."""
    longitude_deg, latitude_deg = to_lonlat(LUNAR_SOUTH_POLAR_STEREOGRAPHIC, 0.0, 0.0)
    assert latitude_deg == pytest.approx(-90.0, abs=1e-9)

    longitude_deg, latitude_deg = to_lonlat(LUNAR_SOUTH_POLAR_STEREOGRAPHIC, 0.0, 30_000.0)
    assert longitude_deg == pytest.approx(0.0, abs=1e-9)
    assert latitude_deg == pytest.approx(-89.0107, abs=1e-4)

    # +x is 90 degrees east of the central meridian.
    longitude_deg, _ = to_lonlat(LUNAR_SOUTH_POLAR_STEREOGRAPHIC, 30_000.0, 0.0)
    assert longitude_deg == pytest.approx(90.0, abs=1e-9)


def test_to_lonlat_passes_a_geographic_crs_straight_through() -> None:
    """An unprojected CRS is already lon/lat degrees; there is nothing to invert."""
    geographic = PlanetaryCRS(
        body=MOON, body_fixed_frame="MOON_ME", reference_radius_m=MOON_RADIUS_M
    )
    assert to_lonlat(geographic, 12.5, -80.25) == (12.5, -80.25)
