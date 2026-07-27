"""Broadened parametric families — orbital/surface/manipulation/logistics/isru (FLEET-10)."""

from __future__ import annotations

import pytest

from astro_mine.core.sadf import to_wire, validate_sadf
from astro_mine.core.sadf.enums import SensorKind
from astro_mine.fleet.capabilities import as_tags
from astro_mine.fleet.lint import lint_asset
from astro_mine.fleet.params import ParamError
from astro_mine.fleet.templates import FAMILIES, families, get_family, resolve_family

_NAMES = sorted(FAMILIES)


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("pick", ["minimum", "default", "maximum"])
def test_family_resolves_valid_and_plausible_across_its_range(name: str, pick: str) -> None:
    fam = FAMILIES[name]
    overrides = {spec.name: getattr(spec, pick) for spec in fam.params}
    doc = fam.resolve(overrides, variant=pick)
    validate_sadf(doc)  # schema + semantic gate (Core)
    assert lint_asset(doc.asset) == []  # physically plausible at this parameter corner
    # every applied capability tag is in Core's closed vocabulary (autonomy + export-control)
    assert as_tags(doc.asset.capabilities)


@pytest.mark.parametrize("name", _NAMES)
def test_family_resolution_is_deterministic(name: str) -> None:
    assert to_wire(resolve_family(name)) == to_wire(resolve_family(name))


def test_families_menu_is_sorted() -> None:
    assert families() == _NAMES
    assert set(_NAMES) == {
        "isru-plant",
        "logistics-hauler",
        "manipulation-excavator",
        "orbital-relay",
        "surface-rover",
    }


def test_get_unknown_family_lists_the_menu() -> None:
    with pytest.raises(ParamError, match="unknown family 'flying-saucer'"):
        get_family("flying-saucer")


def test_surface_rover_range_extends_beyond_the_anchor() -> None:
    # The anchor prospecting rover is ~180 kg; the family spans 10-500 kg (fleet.md §2.3).
    fam = get_family("surface-rover")
    light = fam.resolve({"chassis_mass_kg": 10.0})
    heavy = fam.resolve({"chassis_mass_kg": 500.0})
    assert light.asset.bodies[0].mass_kg == 10.0
    assert heavy.asset.bodies[0].mass_kg == 500.0
    # derived quantities track mass -> distinct canonical bytes
    assert to_wire(light) != to_wire(heavy)


def test_resolve_family_rejects_out_of_range_binding() -> None:
    with pytest.raises(ParamError, match="outside its validated range"):
        resolve_family("surface-rover", {"chassis_mass_kg": 5.0})


def test_isru_family_applies_the_resource_storage_gauge() -> None:
    # The broadened ISRU family applies the Core RESOURCE_STORAGE SensorKind (RFC-0003).
    doc = resolve_family("isru-plant")
    assert SensorKind.RESOURCE_STORAGE in {sensor.kind for sensor in doc.asset.sensors}


def test_variant_and_version_flow_into_identity() -> None:
    doc = resolve_family("orbital-relay", variant="heavy", version="2.0.0")
    assert doc.asset.identity.id.endswith(".heavy")
    assert doc.asset.identity.version == "2.0.0"


def test_manipulation_family_declares_joint_between_declared_bodies() -> None:
    # An articulated family must be referentially closed (Core closure check).
    doc = resolve_family("manipulation-excavator")
    body_names = {b.name for b in doc.asset.bodies}
    for joint in doc.asset.joints:
        assert joint.parent_body in body_names and joint.child_body in body_names
