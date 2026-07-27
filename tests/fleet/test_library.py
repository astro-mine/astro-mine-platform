"""The anchor reference asset library (RM-P0-FLEET-04).

Every shipped reference asset must load and validate through Core, pass Fleet's
physical-plausibility lint, declare a low-fi ``massmodel`` plus at least one higher
fidelity tier under one identity, and package to a stable content hash so Bench can pin
it (scenario §6; LUNAR-FR-003; fleet.md §12). These tests are the realizability gate for
the roster ahead of the Sim instantiation smoke test (RM-P0-FLEET-07).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.sadf import SadfDocument, validate_sadf
from astro_mine.core.sadf.enums import (
    CapabilityTag,
    ContactElementKind,
    FidelityTier,
    SensorKind,
)
from astro_mine.core.units import is_si_unit
from astro_mine.fleet import fidelity, library, lint, packaging

NAMES = library.available()

#: The anchor "robot menu" the roster must cover (issue #4 scope; ``identity.kind`` labels).
EXPECTED_KINDS = {"orbiter", "lander", "rover", "excavator", "hauler", "isru_plant"}


def test_available_lists_full_roster() -> None:
    assert sorted(library.REFERENCE_ASSETS) == NAMES
    assert len(NAMES) == 6


def test_unknown_reference_raises() -> None:
    with pytest.raises(ValueError, match="unknown reference asset"):
        library.load_reference("nonexistent")


def test_roster_covers_anchor_menu() -> None:
    kinds = {library.load_reference(name).asset.identity.kind for name in NAMES}
    assert kinds == EXPECTED_KINDS


#: Each reference asset's current published revision. Pinned per-asset rather than asserted as a
#: blanket "everything is 0.1.0": registry digests are immutable, so revising an asset means
#: *publishing a new version*, and a test that forbids any asset from ever having been revised would
#: have to be edited to allow the very thing it exists to track. Bump an entry here when its asset
#: is republished, and say why.
REFERENCE_VERSIONS = {
    "relay_orbiter": "0.1.0",
    "lander": "0.1.0",
    "prospecting_rover": "0.1.0",
    # 0.2.0 — declares its digging blade as a `tool` contact element (fleet#37). See
    # `test_excavator_declares_a_digging_tool` below.
    "excavator": "0.2.0",
    "hauler": "0.1.0",
    # 0.2.0 — declares the `water_gauge` storage sensor (fleet#40). Without it the plant filled a
    # tank nothing could read: Bench scores `water_mass` off that gauge, so a full plant was
    # indistinguishable from an idle one. See `test_the_plant_declares_a_scoreable_water_gauge`.
    "isru_plant": "0.2.0",
}


def test_reference_versions_cover_the_roster() -> None:
    assert sorted(REFERENCE_VERSIONS) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_reference_loads_and_validates(name: str) -> None:
    doc = library.load_reference(name)
    assert isinstance(doc, SadfDocument)
    assert doc.sadf_version == "0.1"
    # Re-validating a freshly loaded doc must be a no-op, not an error.
    validate_sadf(doc)
    assert doc.asset.core_interface_versions == {"sadf": "0.1.0"}
    assert doc.asset.identity.version == REFERENCE_VERSIONS[name]


def test_excavator_declares_a_digging_tool() -> None:
    """The excavator's blade is a `tool` contact element — the thing that makes it an excavator.

    A `tool` element is the *only* declaration a physics engine reads to decide that an asset's
    contact with the ground is a cutting interaction rather than a rolling one, and so the only
    thing that routes it to a granular (DEM / learned-surrogate) contact model instead of a
    wheel-soil one. Until 0.2.0 the excavator declared only a wheel, so the granular tier was
    unreachable from any published asset in this library and the whole DEM/surrogate ladder was
    dead content (fleet#37).

    The two dimensions asserted here are load-bearing downstream, not decoration: a granular engine
    sizes its **particle bed** from the blade's cutting width (`dimensions_m.x`) and its **tool**
    from the blade height (`dimensions_m.z`). An engine that finds no tool element silently falls
    back to its own defaults — which is exactly how a missing blade stays invisible.
    """
    asset = library.load_reference("excavator").asset
    contact = {element.kind: element for element in asset.mobility.contact}
    assert ContactElementKind.TOOL in contact, "the excavator has no digging blade"
    # It is still a rover as well as a digger: it has to drive to the dig site.
    assert ContactElementKind.WHEEL in contact

    tool = contact[ContactElementKind.TOOL]
    assert tool.dimensions_m is not None, "a blade with no dimensions sizes no particle bed"
    assert tool.dimensions_m.x > 0.0  # cutting width
    assert tool.dimensions_m.z > 0.0  # blade height

    # The bucket must be able to lift what it can hold: one pan of lunar regolith at the nominal
    # 1500 kg/m^3 bulk density has to fit inside the declared payload capacity, or the asset
    # describes a machine that overloads itself every scoop.
    pan_volume_m3 = tool.dimensions_m.x * tool.dimensions_m.y * tool.dimensions_m.z
    assert asset.payload is not None
    assert pan_volume_m3 * 1500.0 <= asset.payload.capacity_kg

    # A blade concentrates force where a wheel spreads it; if the tool's pressure ceiling were the
    # lower of the two the asset would be declaring a bucket it must not dig with.
    wheel = contact[ContactElementKind.WHEEL]
    assert tool.max_ground_pressure_pa is not None
    assert wheel.max_ground_pressure_pa is not None
    assert tool.max_ground_pressure_pa > wheel.max_ground_pressure_pa


@pytest.mark.parametrize("name", NAMES)
def test_reference_is_lint_clean(name: str) -> None:
    findings = lint.lint_asset(library.load_reference(name).asset)
    assert findings == [], f"{name} has plausibility findings: {findings}"


@pytest.mark.parametrize("name", NAMES)
def test_reference_kinematic_graph_is_referentially_closed(name: str) -> None:
    """Every joint/actuator/frame cross-reference must resolve to a declared entity.

    Core's SADF loader does not (yet) validate kinematic-graph referential integrity, so a
    dangling ``child_body`` loads and lints clean yet cannot be realized at the
    ``articulated`` tier (issue #15). This guards the shipped roster against that class of
    latent defect until Core grows the check.
    """
    asset = library.load_reference(name).asset
    bodies = {b.name for b in asset.bodies}
    joints = {j.name for j in asset.joints}
    frames = {f.name for f in asset.frames}
    for j in asset.joints:
        assert j.parent_body in bodies, (
            f"{name}: joint {j.name!r} parent_body {j.parent_body!r} not declared"
        )
        assert j.child_body in bodies, (
            f"{name}: joint {j.name!r} child_body {j.child_body!r} not declared"
        )
    for act in asset.actuators:
        assert act.target_joint in joints, (
            f"{name}: actuator {act.name!r} target_joint {act.target_joint!r} not declared"
        )
    for f in asset.frames:
        if f.parent is not None:
            assert f.parent in frames, f"{name}: frame {f.name!r} parent {f.parent!r} not declared"


@pytest.mark.parametrize("name", NAMES)
def test_reference_has_massmodel_plus_one_tier(name: str) -> None:
    asset = library.load_reference(name).asset
    fidelity.validate_profiles(asset)  # unique tiers, no surrogate
    tiers = fidelity.tiers(asset)
    assert FidelityTier.MASSMODEL in tiers, f"{name} lacks a low-fi massmodel"
    assert len(tiers) >= 2, f"{name} needs >= 1 higher-fidelity profile"
    assert FidelityTier.SURROGATE not in tiers  # deferred to P1


@pytest.mark.parametrize("name", NAMES)
def test_reference_declares_no_gated_capabilities(name: str) -> None:
    caps = set(library.load_reference(name).asset.capabilities)
    assert CapabilityTag.OPERATIONAL_TARGETING not in caps
    assert CapabilityTag.GROUND_TRUTH_ACCESS not in caps


@pytest.mark.parametrize("name", NAMES)
def test_reference_packages_to_stable_digest(name: str, tmp_path: Path) -> None:
    doc = library.load_reference(name)
    first = packaging.package_asset(doc, tmp_path / "a")
    second = packaging.package_asset(doc, tmp_path / "b")
    assert first.digest == second.digest, "packaging must be deterministic for Bench pinning"
    assert first.digest.startswith("sha256:")


def test_reference_digests_are_distinct(tmp_path: Path) -> None:
    # package_asset writes under <out>/sha256/<digest>/, so a shared base dir is fine.
    digests = {
        name: packaging.package_asset(library.load_reference(name), tmp_path).digest
        for name in NAMES
    }
    assert len(set(digests.values())) == len(NAMES), digests


def test_prospecting_rover_has_required_sensing_suite() -> None:
    asset = library.load_reference("prospecting_rover").asset
    sensor_kinds = {s.kind for s in asset.sensors}
    assert {
        SensorKind.NEUTRON_SPECTROMETER,
        SensorKind.NIR_SPECTROMETER,
        SensorKind.GPR,
        SensorKind.DRILL_ASSAY,
    } <= sensor_kinds
    caps = set(asset.capabilities)
    assert {
        CapabilityTag.PROSPECTING_NEUTRON,
        CapabilityTag.PROSPECTING_NIR,
        CapabilityTag.PROSPECTING_GPR,
        CapabilityTag.PROSPECTING_DRILL_ASSAY,
        CapabilityTag.MOBILITY_WHEELED,
    } <= caps


def test_relay_orbiter_is_a_comms_relay() -> None:
    asset = library.load_reference("relay_orbiter").asset
    assert CapabilityTag.COMMS_RELAY in asset.capabilities
    assert CapabilityTag.MOBILITY_ORBITER in asset.capabilities
    assert any(c.relay for c in asset.comms), "relay orbiter must declare a relay radio"


def test_isru_plant_extracts_water_but_does_not_electrolyse() -> None:
    asset = library.load_reference("isru_plant").asset
    caps = set(asset.capabilities)
    assert {
        CapabilityTag.ISRU_THERMAL_EXTRACTION,
        CapabilityTag.ISRU_PURIFICATION,
        CapabilityTag.ISRU_STORAGE,
    } <= caps
    # Baseline value chain ends at stored water (scenario §15) — electrolysis is deferred.
    assert CapabilityTag.ISRU_ELECTROLYSIS not in caps
    assert asset.payload is not None and asset.payload.isru is not None
    assert asset.payload.isru.throughput_kg_hr == 10.0


def test_the_plant_declares_a_scoreable_water_gauge() -> None:
    """The anchor's plant must report its tank, in a unit the platform matches (fleet#40).

    Two things have to be true for `water_mass` to see a kilogram of water: the gauge has to exist,
    and its `si_unit` has to be the token Bench filters on. The plant declared neither — it had no
    `resource_storage` sensor at all — while the parametric ISRU family declared one with
    `si_unit="mass_kg"`, which is not a known unit and is therefore silently skipped downstream.
    """
    doc = library.load_reference("isru_plant")
    gauges = [s for s in doc.asset.sensors if s.kind is SensorKind.RESOURCE_STORAGE]
    assert len(gauges) == 1, "the plant declares no storage gauge, so its tank is unreadable"
    resource = gauges[0].resource
    assert resource is not None, "a storage gauge must say what it holds"
    assert resource.species == "water"
    assert resource.si_unit == "kg"
    assert is_si_unit(resource.si_unit), "the declared unit is not one the platform knows"
