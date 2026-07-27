"""The parameter-resolution engine (RM-P1-FLEET-10)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from astro_mine.core.sadf import to_wire
from astro_mine.fleet.params import Family, ParamError, ParamSpec


def _tiny_asset(variant: str, version: str, p: Mapping[str, float]) -> dict[str, Any]:
    return {
        "identity": {"id": f"test.{variant}", "name": "T", "version": version, "kind": "rover"},
        "core_interface_versions": {"sadf": "0.1.0"},
        "root_frame": "base",
        "frames": [{"name": "base"}],
        "bodies": [
            {
                "name": "hull",
                "frame": "base",
                "mass_kg": p["mass_kg"],
                "center_of_mass_m": {"x": 0.0, "y": 0.0, "z": 0.0},
                "inertia_kg_m2": {"ixx": p["mass_kg"], "iyy": p["mass_kg"], "izz": p["mass_kg"]},
            }
        ],
    }


_FAMILY = Family(
    name="tiny",
    kind="rover",
    summary="a tiny test family",
    params=(
        ParamSpec("mass_kg", 1.0, 100.0, 10.0, "kg", "mass"),
        ParamSpec("wheels", 3.0, 8.0, 4.0, "count", "wheel count", integer=True),
    ),
    build_asset=_tiny_asset,
)


def test_paramspec_rejects_inverted_range() -> None:
    with pytest.raises(ParamError, match="exceeds maximum"):
        ParamSpec("m", 100.0, 1.0, 5.0, "kg", "bad")


def test_paramspec_rejects_default_outside_range() -> None:
    with pytest.raises(ParamError, match="outside"):
        ParamSpec("m", 0.0, 10.0, 20.0, "kg", "bad default")


def test_coerce_accepts_a_value_in_range() -> None:
    assert _FAMILY.spec("mass_kg").coerce(50) == 50.0


def test_coerce_rejects_out_of_range() -> None:
    with pytest.raises(ParamError, match="outside its validated range"):
        _FAMILY.spec("mass_kg").coerce(500.0)


def test_coerce_snaps_and_validates_integers() -> None:
    assert _FAMILY.spec("wheels").coerce(6.0) == 6.0
    with pytest.raises(ParamError, match="must be an integer"):
        _FAMILY.spec("wheels").coerce(4.5)


def test_coerce_rejects_non_numeric() -> None:
    with pytest.raises(ParamError, match="is not a number"):
        _FAMILY.spec("mass_kg").coerce("heavy")  # type: ignore[arg-type]


def test_bind_applies_defaults_then_overrides() -> None:
    assert _FAMILY.bind() == {"mass_kg": 10.0, "wheels": 4.0}
    assert _FAMILY.bind({"mass_kg": 25.0}) == {"mass_kg": 25.0, "wheels": 4.0}


def test_bind_rejects_unknown_parameter() -> None:
    with pytest.raises(ParamError, match="unknown parameter 'length'"):
        _FAMILY.bind({"length": 3.0})


def test_resolve_returns_validated_document() -> None:
    doc = _FAMILY.resolve({"mass_kg": 30.0}, variant="mid", version="1.2.3")
    assert doc.asset.identity.id == "test.mid"
    assert doc.asset.identity.version == "1.2.3"
    assert doc.asset.bodies[0].mass_kg == 30.0


def test_resolution_is_deterministic() -> None:
    assert to_wire(_FAMILY.resolve({"mass_kg": 42.0})) == to_wire(
        _FAMILY.resolve({"mass_kg": 42.0})
    )
