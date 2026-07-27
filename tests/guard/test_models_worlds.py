"""WorldsTerrain — deriving illumination / slope / temperature signals (RM-P1-GUARD-04)."""

from __future__ import annotations

import math

import pytest

from astro_mine.core.units import ReferenceFrame
from astro_mine.core.units.enums import FrameClass
from astro_mine.core.world.model import IlluminationState
from astro_mine.guard.models import WorldsTerrain, slope_deg_from_normal
from tests.guard.models_fixtures import MOON_ME, FakeWorldProvider


def test_slope_from_normal_flat_and_tilted() -> None:
    assert slope_deg_from_normal((0.0, 0.0, 1.0), (0.0, 0.0, -1.62)) == 0.0
    tilt = slope_deg_from_normal(
        (0.0, math.sin(math.radians(30.0)), math.cos(math.radians(30.0))), (0.0, 0.0, -1.0)
    )
    assert tilt == pytest.approx(30.0)


def test_slope_from_normal_degenerate_is_nan() -> None:
    assert math.isnan(slope_deg_from_normal((0.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
    assert math.isnan(slope_deg_from_normal((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)))


def test_charging_window_keys_off_illumination() -> None:
    lit = WorldsTerrain(FakeWorldProvider(illumination=IlluminationState.LIT))
    penumbra = WorldsTerrain(FakeWorldProvider(illumination=IlluminationState.PENUMBRA))
    shadow = WorldsTerrain(FakeWorldProvider(illumination=IlluminationState.SHADOW))
    assert lit.charging_window_active((0.0, 0.0, 0.0)) == 1.0
    assert penumbra.charging_window_active((0.0, 0.0, 0.0)) == 1.0  # any non-shadow charges
    assert shadow.charging_window_active((0.0, 0.0, 0.0)) == 0.0  # PSR ⇒ no window


def test_slope_and_temperature_from_provider() -> None:
    t = WorldsTerrain(
        FakeWorldProvider(
            surface_normal=(0.0, math.sin(math.radians(15.0)), math.cos(math.radians(15.0))),
            gravity=(0.0, 0.0, -1.62),
            temperature_k=95.0,
        )
    )
    assert t.slope_deg((10.0, 0.0, 0.0)) == pytest.approx(15.0)
    assert t.surface_temperature_k((10.0, 0.0, 0.0)) == 95.0
    assert t.frame_name == "MOON_ME"


def test_rejects_non_body_fixed_frame() -> None:
    inertial = ReferenceFrame(name="J2000", frame_class=FrameClass.INERTIAL, center="MOON")
    with pytest.raises(ValueError, match="body-fixed"):
        WorldsTerrain(FakeWorldProvider(frame=inertial))


def test_rejects_unexpected_center() -> None:
    earth = ReferenceFrame(name="ITRF93", frame_class=FrameClass.BODY_FIXED, center="EARTH")
    with pytest.raises(ValueError, match="centred on"):
        WorldsTerrain(FakeWorldProvider(frame=earth), expected_center="MOON")


def test_accepts_matching_center() -> None:
    t = WorldsTerrain(FakeWorldProvider(frame=MOON_ME), expected_center="MOON")
    assert t.frame_name == "MOON_ME"


def test_keep_out_box_is_built_in_the_provider_frame() -> None:
    t = WorldsTerrain(FakeWorldProvider(frame=MOON_ME))
    vol = t.keep_out_box((500.0, -1200.0, 0.0), (200.0, 200.0, 40.0))
    assert vol.frame == "MOON_ME"
    assert (vol.center_m.x, vol.center_m.y, vol.center_m.z) == (500.0, -1200.0, 0.0)
    assert (vol.dimensions_m.x, vol.dimensions_m.y, vol.dimensions_m.z) == (200.0, 200.0, 40.0)
