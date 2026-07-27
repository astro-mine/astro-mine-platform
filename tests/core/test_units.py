"""Tests for ``astro_mine.core.units`` — frame/CRS/epoch primitives and the fail-loud
ingest guards (RM-P0-CORE-06)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.core import units
from astro_mine.core.units import (
    DIMENSIONLESS_UNITS,
    KNOWN_UNITS,
    SI_UNITS,
    Epoch,
    EpochWindow,
    FrameClass,
    PlanetaryCRS,
    ReferenceFrame,
    SiUnit,
    TimeScale,
    UnitsError,
    UnitsValidationError,
    is_si_unit,
    require_crs,
    require_frame,
    require_si_unit,
)

_LUNAR_R_M = 1_737_400.0


# --- vocabularies -----------------------------------------------------------------


def test_time_scale_admits_only_tdb_et() -> None:
    assert {s.value for s in TimeScale} == {"tdb", "et"}


def test_frame_class_values() -> None:
    assert {f.value for f in FrameClass} == {"body_fixed", "inertial", "topocentric"}


def test_si_unit_catalog() -> None:
    assert {"m", "kg", "s", "K", "Pa", "rad"} <= SI_UNITS
    assert frozenset(u.value for u in SiUnit) == SI_UNITS
    assert {"dimensionless", "mass_fraction", "volume_fraction"} == DIMENSIONLESS_UNITS
    assert KNOWN_UNITS == SI_UNITS | DIMENSIONLESS_UNITS
    # SI symbols and dimensionless markers are disjoint vocabularies.
    assert SI_UNITS.isdisjoint(DIMENSIONLESS_UNITS)


# --- ReferenceFrame ---------------------------------------------------------------


def test_reference_frame_valid() -> None:
    f = ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center="MOON")
    assert (f.name, f.frame_class, f.center) == ("MOON_ME", FrameClass.BODY_FIXED, "MOON")


def test_reference_frame_center_optional() -> None:
    assert ReferenceFrame(name="J2000", frame_class=FrameClass.INERTIAL).center is None


@pytest.mark.parametrize("bad", ["", " ", "MOON ME", " MOON_ME", "MOON_ME ", "\tX"])
def test_reference_frame_rejects_bad_name(bad: str) -> None:
    with pytest.raises(ValidationError):
        ReferenceFrame(name=bad, frame_class=FrameClass.BODY_FIXED)


def test_reference_frame_rejects_bad_center() -> None:
    with pytest.raises(ValidationError):
        ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center="bad center")


def test_reference_frame_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        ReferenceFrame(name="X", frame_class=FrameClass.INERTIAL, bogus=1)  # type: ignore[call-arg]


# --- PlanetaryCRS -----------------------------------------------------------------


def test_planetary_crs_geographic() -> None:
    crs = PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=_LUNAR_R_M)
    assert crs.projection is None and crs.datum is None


def test_planetary_crs_projected() -> None:
    crs = PlanetaryCRS(
        body="MOON",
        body_fixed_frame="MOON_ME",
        reference_radius_m=_LUNAR_R_M,
        projection="+proj=stere +lat_0=-90 +R=1737400",
        datum="MOON_ME",
    )
    assert crs.projection is not None and crs.projection.startswith("+proj=stere")


def test_planetary_crs_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValidationError):
        PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=0.0)


def test_planetary_crs_rejects_missing_frame() -> None:
    with pytest.raises(ValidationError):
        PlanetaryCRS(body="MOON", reference_radius_m=_LUNAR_R_M)  # type: ignore[call-arg]


def test_planetary_crs_rejects_blank_body() -> None:
    with pytest.raises(ValidationError):
        PlanetaryCRS(body="  ", body_fixed_frame="MOON_ME", reference_radius_m=_LUNAR_R_M)


# --- Epoch / EpochWindow ----------------------------------------------------------


def test_epoch_valid() -> None:
    assert Epoch(tdb_seconds=123.0, scale=TimeScale.TDB).tdb_seconds == 123.0


def test_epoch_et_scale() -> None:
    assert Epoch(tdb_seconds=0.0, scale=TimeScale.ET).scale is TimeScale.ET


def test_epoch_requires_scale() -> None:
    with pytest.raises(ValidationError):
        Epoch(tdb_seconds=0.0)  # type: ignore[call-arg]


def test_epoch_rejects_non_tdb_scale() -> None:
    with pytest.raises(ValidationError):
        Epoch(tdb_seconds=0.0, scale="utc")  # type: ignore[arg-type]


def test_epoch_window_valid() -> None:
    w = EpochWindow(
        start=Epoch(tdb_seconds=0.0, scale=TimeScale.TDB),
        end=Epoch(tdb_seconds=10.0, scale=TimeScale.TDB),
    )
    assert w.end.tdb_seconds == 10.0


@pytest.mark.parametrize("end_s", [10.0, 5.0])
def test_epoch_window_rejects_non_increasing(end_s: float) -> None:
    start = Epoch(tdb_seconds=10.0, scale=TimeScale.TDB)
    with pytest.raises(ValidationError):
        EpochWindow(start=start, end=Epoch(tdb_seconds=end_s, scale=TimeScale.TDB))


# --- canonical constants ----------------------------------------------------------


def test_canonical_constants() -> None:
    assert units.MOON_BODY_FIXED.name == "MOON_ME"
    assert units.MOON_BODY_FIXED.frame_class is FrameClass.BODY_FIXED
    assert units.MOON_BODY_FIXED.center == "MOON"
    assert units.INERTIAL_J2000.frame_class is FrameClass.INERTIAL
    assert units.INERTIAL_J2000.center is None
    assert units.J2000_EPOCH.tdb_seconds == 0.0
    assert (units.MOON, units.SUN, units.EARTH) == ("MOON", "SUN", "EARTH")


# --- unit guards ------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["m", "kg", "Pa", "rad", "mass_fraction", "dimensionless"])
def test_require_si_unit_accepts(unit: str) -> None:
    assert require_si_unit(unit) == unit
    assert is_si_unit(unit)


@pytest.mark.parametrize("unit", ["kg/m3", "furlong", "deg", ""])
def test_require_si_unit_rejects(unit: str) -> None:
    assert not is_si_unit(unit)
    with pytest.raises(UnitsValidationError):
        require_si_unit(unit)


# --- frame / CRS guards -----------------------------------------------------------


def test_require_frame_passthrough() -> None:
    f = ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED)
    assert require_frame(f) is f


def test_require_frame_from_dict() -> None:
    f = require_frame({"name": "J2000", "frame_class": "inertial"})
    assert isinstance(f, ReferenceFrame) and f.name == "J2000"


def test_require_frame_rejects_none() -> None:
    with pytest.raises(UnitsValidationError, match="no implicit Earth/WGS84"):
        require_frame(None)


def test_require_frame_rejects_invalid_dict() -> None:
    with pytest.raises(UnitsValidationError):
        require_frame({"name": "  ", "frame_class": "inertial"})


def test_require_frame_rejects_wrong_type() -> None:
    with pytest.raises(UnitsValidationError):
        require_frame(42)


def test_require_crs_passthrough() -> None:
    crs = PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=_LUNAR_R_M)
    assert require_crs(crs) is crs


def test_require_crs_from_dict() -> None:
    crs = require_crs(
        {"body": "MOON", "body_fixed_frame": "MOON_ME", "reference_radius_m": _LUNAR_R_M}
    )
    assert isinstance(crs, PlanetaryCRS)


def test_require_crs_rejects_none() -> None:
    with pytest.raises(UnitsValidationError, match="explicit planetary CRS"):
        require_crs(None)


def test_require_crs_rejects_invalid_dict() -> None:
    with pytest.raises(UnitsValidationError):
        require_crs({"body": "MOON"})  # missing required fields


def test_require_crs_rejects_wrong_type() -> None:
    with pytest.raises(UnitsValidationError):
        require_crs(3.14)


# --- misc -------------------------------------------------------------------------


def test_exception_hierarchy() -> None:
    assert issubclass(UnitsValidationError, UnitsError)


def test_models_round_trip_deterministically() -> None:
    crs = PlanetaryCRS(
        body="MOON",
        body_fixed_frame="MOON_ME",
        reference_radius_m=_LUNAR_R_M,
        projection="+proj=stere",
    )
    dumped = crs.model_dump(mode="json")
    assert PlanetaryCRS.model_validate(dumped) == crs
    assert crs.model_dump(mode="json") == dumped  # stable across repeated dumps

    w = EpochWindow(start=units.J2000_EPOCH, end=Epoch(tdb_seconds=1.0, scale=TimeScale.TDB))
    assert EpochWindow.model_validate(w.model_dump(mode="json")) == w
