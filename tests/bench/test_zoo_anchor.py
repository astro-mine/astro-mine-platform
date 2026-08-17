"""The anchor scenario in the zoo (RM-P0-BENCH-02).

Asserts the anchor "Lunar Polar Water-Ice Prospecting v1" loads, resolves end-to-end and
deterministically, pins the expected content/seeds/metrics, that every content pin is a real Hub
digest distinct from the provisional derivation of its ``pins.json`` descriptor (no silent drift),
and that the held-out seeds are absent from this tree entirely while remaining bound by the spec's
``heldout_commit``.

**The seeds moved out of this repository** (astro-mine-platform#37). They were committed here in
plaintext on the standing assumption that the repository was private, and the public flip retires
that assumption for every commit rather than just ``HEAD`` — so rotating in place would have
republished the problem one commit later. They now live in the private ``astro-mine/embargo``,
reached through ``$ASTRO_MINE_BENCH_EMBARGO_ROOT``.

That splits one assertion into two of different character, which is the honest shape rather than a
concession: **absence** is unconditional and needs no secret, while the **commitment** can only be
checked where the store is reachable. The second is guarded so that an unreachable store is loud and
a *wrong* store still fails.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

import pytest

from astro_mine.bench.leaderboard import resolve_embargo_root
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
#: The sealed set is no longer in this repository at all (astro-mine-platform#37) — it lives in the
#: private ``astro-mine/embargo``, reached through ``$ASTRO_MINE_BENCH_EMBARGO_ROOT``. Resolving it
#: through the library's own :func:`resolve_embargo_root` rather than rebuilding the path here means
#: this test exercises the seam an evaluator uses, not a parallel one that could drift from it.
_HELDOUT = resolve_embargo_root() / ANCHOR_SCENARIO_ID / "heldout_seeds.json"

#: Set when the store is reachable. Unset is *not* a pass — see
#: :func:`test_heldout_commitment_binds_the_sealed_seeds` for why that distinction is enforced.
_STORE_AVAILABLE = _HELDOUT.is_file()

_PINS: dict[str, dict[str, str]] = json.loads(
    files("astro_mine.bench.zoo")
    .joinpath(ANCHOR_SCENARIO_ID.replace("-", "_"), "pins.json")
    .read_text(encoding="utf-8")
)

EXPECTED_FLEET_IDS = (
    "relay-orbiter",
    "lander",
    "prospecting-rover",
    "excavator",
    "hauler",
    "isru-plant",
)
EXPECTED_LINK_ID = "lunar-polar-relay-dsn"
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
    assert anchor.content.world.id == "shackleton-de-gerlache"
    assert tuple(f.id for f in anchor.content.fleet) == EXPECTED_FLEET_IDS
    assert tuple(p.id for p in anchor.content.prospect) == ("shackleton-water-ice",)
    assert anchor.content.link is not None
    assert anchor.content.link.id == EXPECTED_LINK_ID
    assert len(anchor.content_refs()) == 9  # world + 6 fleet + 1 prospect + 1 link


# The whole anchor is published to Hub — world (RM-P1-WORLDS-15), fleet (RM-P1-FLEET-10), prospect
# (RM-P1-PROSPECT-13), link (the anchor ContactPlan) — and Sim resolves it into a runnable Scenario
# (RM-P1-SIM-01). Every pin is a real Hub artifact digest, distinct from the provisional-pin
# derivation of its descriptor.
_REAL_PIN_IDS = frozenset(EXPECTED_FLEET_IDS) | {
    "shackleton-water-ice",
    "shackleton-de-gerlache",
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


def test_no_heldout_seeds_anywhere_in_the_working_tree() -> None:
    """**Always runs.** The property the public flip actually needs, and it is unconditional.

    This replaces an assertion that the sealed set *was* present at ``embargo/<id>/`` and merely
    outside ``src/``. That was the right check while the repository was private and the seeds were
    committed on purpose; it is the wrong one now, because the flip publishes every commit rather
    than ``HEAD``, so "outside the wheel" stopped being far enough away.

    Absence is also the half that can be checked without the secret, which is what makes it
    unconditional where the commitment test below cannot be.
    """
    stragglers = sorted(
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _REPO_ROOT.rglob("heldout_seeds.json")
        if ".git" not in path.parts
    )
    assert stragglers == [], (
        f"held-out seed sets found in the working tree: {stragglers}. These must live only in the "
        f"private astro-mine/embargo store, reached via $ASTRO_MINE_BENCH_EMBARGO_ROOT "
        f"(astro-mine-platform#37)."
    )


@pytest.mark.skipif(
    not _STORE_AVAILABLE,
    reason=(
        "the held-out seed store is not reachable: set $ASTRO_MINE_BENCH_EMBARGO_ROOT to a "
        "checkout of the private astro-mine/embargo repository. THIS IS NOT A PASS — the "
        "anchor's commitment went unverified on this run (astro-mine-platform#37)."
    ),
)
def test_heldout_commitment_binds_the_sealed_seeds(anchor: ScenarioSpec) -> None:
    """Verifies the commitment when the store is reachable, and is honest when it is not.

    The seeds left the tree, so this check cannot be unconditional the way it was — and #37 names
    the failure mode to avoid: *a test that silently skips*. Two things keep that from happening.
    The skip reason says in as many words that nothing was verified, rather than reading as a
    benign environmental skip. And a store that is *present but wrong* fails rather than skips —
    the guard below is `is_file()` at collection time, so a malformed or stale set gets here and
    trips the assertion.
    """
    payload = json.loads(_HELDOUT.read_text(encoding="utf-8"))
    assert content_hash(payload) == anchor.seeds.heldout_commit, (
        "the sealed set does not match the spec's heldout_commit — the store is stale, or the "
        "scenario was re-versioned without re-sealing"
    )
    assert not (set(payload["seeds"]) & set(anchor.seeds.public))  # disjoint from public seeds


def test_the_rotated_commitment_is_not_the_retired_one() -> None:
    """The rotation is asserted, not just performed.

    The set retired on 2026-08-16 stays readable in this repository's history forever once it is
    public, so the one thing that must never regress is the anchor pointing back at it. Pinning the
    dead commitment by value is the only check that survives the seeds themselves being gone.
    """
    retired = "sha256:fee93327b5943041865348cc47b4b9db5bde955a9cd8c307ebeba18569ab5640"
    spec = json.loads(
        files("astro_mine.bench.zoo")
        .joinpath(ANCHOR_SCENARIO_ID.replace("-", "_"), "scenario.json")
        .read_text(encoding="utf-8")
    )
    assert spec["seeds"]["heldout_commit"] != retired, (
        "the anchor still commits to the seed set retired in astro-mine-platform#37, which is "
        "published in this repository's git history"
    )


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
