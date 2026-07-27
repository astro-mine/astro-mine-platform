"""The anchor scenario in the zoo (RM-P0-BENCH-02).

Asserts the anchor "Lunar Polar Water-Ice Prospecting v1" loads, resolves end-to-end and
deterministically, pins the expected content/seeds/metrics, that every content pin is a real Hub
digest distinct from the provisional derivation of its ``pins.json`` descriptor (no silent drift),
and that the held-out seeds are embargoed outside the packaged tree yet bound by the spec's
``heldout_commit``.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

import pytest

from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.scenario._hash import content_hash
from astro_mine.bench.zoo import (
    ANCHOR_SCENARIO_ID,
    ResolvedScenario,
    list_scenarios,
    load_scenario,
    resolve_anchor,
    resolve_scenario,
)
from astro_mine.bench.zoo._provisional import provisional_pin_hash
from astro_mine.core import SCHEMA_DIGEST

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELDOUT = _REPO_ROOT / "embargo" / ANCHOR_SCENARIO_ID / "heldout_seeds.json"

_PINS: dict[str, dict[str, str]] = json.loads(
    files("astro_mine.bench.zoo")
    .joinpath(ANCHOR_SCENARIO_ID.replace("-", "_"), "pins.json")
    .read_text(encoding="utf-8")
)

EXPECTED_FLEET_IDS = (
    "astro-mine.fleet.relay-orbiter",
    "astro-mine.fleet.lander",
    "astro-mine.fleet.prospecting-rover",
    "astro-mine.fleet.excavator",
    "astro-mine.fleet.hauler",
    "astro-mine.fleet.isru-plant",
)
EXPECTED_LINK_ID = "astro-mine.link.lunar-polar-relay-dsn"
EXPECTED_METRICS = (
    "water_mass",
    "energy_per_kg",
    "information_gain",
    "psr_area_characterized",
    "nights_survived",
    "comms_robustness",
    "discovery_latency",
)
EXPECTED_CORE_INTERFACES = frozenset(
    {
        "env",
        "messages",
        "policy",
        "sadf",
        "objective",
        "resource_field",
        "world_provider",
        "registry",
    }
)


@pytest.fixture(scope="module")
def anchor() -> ScenarioSpec:
    return load_scenario(ANCHOR_SCENARIO_ID)


def test_anchor_is_registered_in_zoo() -> None:
    assert ANCHOR_SCENARIO_ID in list_scenarios()


def test_anchor_loads_with_identity(anchor: ScenarioSpec) -> None:
    assert anchor.scenario_id == ANCHOR_SCENARIO_ID
    assert anchor.name == "Lunar Polar Water-Ice Prospecting v1"


def test_core_interface_pins(anchor: ScenarioSpec) -> None:
    assert set(anchor.core_interface) == EXPECTED_CORE_INTERFACES
    assert set(anchor.core_interface.values()) == {"0.1.0"}


def test_anchor_pins_the_core_schema_digest(anchor: ScenarioSpec) -> None:
    # The interface versions above are frozen at 0.1.0 through Phase 3, so they are constant across
    # every Core revision and cannot tell two schema sets apart. The schema digest can — it is the
    # contract identity the anchor reproduces against (VERSIONING.md §4.1; CX-REPRO).
    assert anchor.core_schema_digest == SCHEMA_DIGEST
    # And it is enforced, not decorative: resolving records the digest the run validated under.
    assert resolve_anchor().core_schema_digest == SCHEMA_DIGEST


def test_content_pins(anchor: ScenarioSpec) -> None:
    assert anchor.content.world.id == "shackleton-de-gerlache-v1"
    assert tuple(f.id for f in anchor.content.fleet) == EXPECTED_FLEET_IDS
    assert tuple(p.id for p in anchor.content.prospect) == ("shackleton_water_ice_v1",)
    assert anchor.content.link is not None
    assert anchor.content.link.id == EXPECTED_LINK_ID
    assert len(anchor.content_refs()) == 9  # world + 6 fleet + 1 prospect + 1 link


# The whole anchor is published to Hub — world (RM-P1-WORLDS-15), fleet (RM-P1-FLEET-10), prospect
# (RM-P1-PROSPECT-13), link (the anchor ContactPlan) — and Sim resolves it into a runnable Scenario
# (RM-P1-SIM-01). Every pin is a real Hub artifact digest, distinct from the provisional-pin
# derivation of its descriptor.
_REAL_PIN_IDS = frozenset(EXPECTED_FLEET_IDS) | {
    "shackleton_water_ice_v1",
    "shackleton-de-gerlache-v1",
    EXPECTED_LINK_ID,
}


def test_pins_cover_the_content_refs(anchor: ScenarioSpec) -> None:
    assert set(_PINS) == {ref.id for ref in anchor.content_refs()}
    assert set(_PINS) == _REAL_PIN_IDS


def test_all_pins_are_real_hub_digests(anchor: ScenarioSpec) -> None:
    assert {ref.id for ref in anchor.content_refs()} == _REAL_PIN_IDS
    for ref in anchor.content_refs():
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", ref.content_hash), ref.id
        # a real published digest, not the provisional-pin derivation of its descriptor
        assert ref.content_hash != provisional_pin_hash(**_PINS[ref.id]), ref.id


def test_metric_set(anchor: ScenarioSpec) -> None:
    assert tuple(m.name for m in anchor.metrics) == EXPECTED_METRICS
    assert all(m.version == "0.1.0" for m in anchor.metrics)


def test_episode_spans_a_lunar_cycle(anchor: ScenarioSpec) -> None:
    assert anchor.episode.max_sim_seconds is not None
    assert anchor.episode.max_sim_seconds >= 29.53 * 86_400  # ≥ one synodic month
    assert anchor.episode.horizon_steps > 0


def test_termination_conditions(anchor: ScenarioSpec) -> None:
    assert anchor.termination.conditions == (
        "critical_asset_loss",
        "power_floor_violation",
        "thermal_ceiling_violation",
    )


def test_public_seeds_and_heldout_commitment(anchor: ScenarioSpec) -> None:
    assert anchor.seeds.public == (1001, 1002, 1003, 1004, 1005)
    assert anchor.seeds.heldout_commit is not None


def test_heldout_seeds_are_embargoed_outside_the_package() -> None:
    assert _HELDOUT.is_file()
    rel = _HELDOUT.relative_to(_REPO_ROOT)
    assert rel.parts[0] == "embargo"  # sealed outside the wheel (ships only src/astro_mine)
    assert "src" not in rel.parts


def test_heldout_commitment_binds_the_sealed_seeds(anchor: ScenarioSpec) -> None:
    payload = json.loads(_HELDOUT.read_text(encoding="utf-8"))
    assert content_hash(payload) == anchor.seeds.heldout_commit
    assert not (set(payload["seeds"]) & set(anchor.seeds.public))  # disjoint from public seeds


def test_resolve_anchor_is_deterministic() -> None:
    first = resolve_anchor()
    second = resolve_anchor()
    assert isinstance(first, ResolvedScenario)
    assert first.scenario_id == ANCHOR_SCENARIO_ID
    assert first.scenario_hash == second.scenario_hash
    assert set(first.content_hashes) == set(_PINS)  # every pinned content id resolved


def test_resolve_scenario_reexport_matches(anchor: ScenarioSpec) -> None:
    assert resolve_scenario(anchor).scenario_hash == resolve_anchor().scenario_hash


def test_unknown_scenario_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        load_scenario("no-such-scenario")
