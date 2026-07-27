"""The units guards enforce the six RFC-0007 rules, driven by the shared vectors.

RM-P1-CORE-08. This runs the language-neutral conformance vectors
(``units/schema/conformance.json``) against the Python **reference** implementation
(``units.validate``); every other binding runs the same vectors in its own CI. Two things
JSON cannot express — a non-finite reference radius (rule 4) and the ET ≡ TDB equivalence
(rule 3) — are exercised in-language below, alongside the RFC-0007 acceptance criteria.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import astro_mine.core.units as units_pkg
from astro_mine.core.units import validate
from astro_mine.core.units.enums import TimeScale
from astro_mine.core.units.model import Epoch, PlanetaryCRS
from astro_mine.core.units.validate import UnitsValidationError

_CONFORMANCE_PATH = Path(units_pkg.__file__).resolve().parent / "schema" / "conformance.json"

_GUARDS: dict[str, Callable[[Any], Any]] = {
    "reference_frame": validate.require_frame,
    "epoch": validate.require_epoch,
    "epoch_window": validate.require_epoch_window,
    "planetary_crs": validate.require_crs,
    "si_unit": validate.require_si_unit,
}


def _load() -> dict[str, Any]:
    return json.loads(_CONFORMANCE_PATH.read_text(encoding="utf-8"))


def _vectors() -> list[tuple[str, str, bool, Any]]:
    out: list[tuple[str, str, bool, Any]] = []
    for kind, cases in _load().items():
        if kind.startswith("$"):  # skip $comment
            continue
        for case in cases:
            out.append((kind, case["name"], case["valid"], case["value"]))
    return out


_VECTORS = _vectors()


def test_conformance_covers_every_guard() -> None:
    kinds = {kind for kind, *_ in _VECTORS}
    assert kinds == set(_GUARDS), "a guard is missing conformance vectors (or vice versa)"


@pytest.mark.parametrize(
    ("kind", "name", "valid", "value"),
    _VECTORS,
    ids=[f"{k}-{n}" for k, n, _, _ in _VECTORS],
)
def test_conformance_vector(kind: str, name: str, valid: bool, value: Any) -> None:
    guard = _GUARDS[kind]
    if valid:
        guard(value)  # must return, not raise
    else:
        with pytest.raises(UnitsValidationError):
            guard(value)


# --- rules JSON cannot express, and the RFC-0007 acceptance criteria ----------------


def test_reference_radius_must_be_finite() -> None:
    # Rule 4: +inf passes Pydantic's gt=0 but require_crs rejects it. JSON has no inf.
    crs_inf = PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=math.inf)
    with pytest.raises(UnitsValidationError):
        validate.require_crs(crs_inf)  # typed path
    with pytest.raises(UnitsValidationError):
        validate.require_crs(
            {"body": "MOON", "body_fixed_frame": "MOON_ME", "reference_radius_m": math.inf}
        )  # dict path


def test_earth_crs_body_datum_consistency() -> None:
    # Rule 6 acceptance criteria: reject an Earth marker off EARTH, accept it on EARTH.
    with pytest.raises(UnitsValidationError):
        validate.require_crs(
            PlanetaryCRS(
                body="MOON",
                body_fixed_frame="MOON_ME",
                reference_radius_m=1737400.0,
                projection="+proj=longlat +datum=WGS84",
            )
        )
    crs = validate.require_crs(
        PlanetaryCRS(
            body="EARTH",
            body_fixed_frame="ITRF93",
            reference_radius_m=6378137.0,
            projection="+proj=longlat +datum=WGS84",
        )
    )
    assert crs.body == "EARTH"


def test_et_scale_is_equivalent_to_tdb() -> None:
    # Rule 3: an ET epoch passes every guard and every consumer path that accepts TDB.
    et = validate.require_epoch(Epoch(tdb_seconds=42.0, scale=TimeScale.ET))
    assert et.scale is TimeScale.ET
    assert validate.scales_equivalent(TimeScale.ET, TimeScale.TDB)
    assert validate.scales_equivalent(TimeScale.TDB, TimeScale.ET)

    # A naive `scale == TDB` gate would reject an ET epoch; scales_equivalent must not.
    def accepts_tdb(e: Epoch) -> bool:
        return validate.scales_equivalent(e.scale, TimeScale.TDB)

    assert accepts_tdb(Epoch(tdb_seconds=0.0, scale=TimeScale.ET))
    assert accepts_tdb(Epoch(tdb_seconds=0.0, scale=TimeScale.TDB))


def test_conformance_file_is_shaped_and_shippable() -> None:
    data = _load()
    assert set(data) - {"$comment"} == set(_GUARDS)
    # ships in the schema bundle — asserted structurally in tests/test_schema_bundle.py.
    assert _CONFORMANCE_PATH.is_file()
