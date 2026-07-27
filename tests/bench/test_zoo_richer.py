"""Richer scenario zoo — new immutable, content-addressed tasks (RM-P1-BENCH-12; bench.md §3, §8).

The acceptance criterion: **a new ScenarioSpec added to the zoo evaluates on the existing harness
with no code change, and historical leaderboards remain valid (immutable specs).** Two new tasks —
a prospecting *sprint* and a full-chain *endurance* run — are authored as documents that reuse the
anchor's published content by hash; the catalog discovers them without a loader edit, they score on
the baseline harness, and the anchor's ``spec_hash`` is pinned so an in-place edit fails CI.
"""

from __future__ import annotations

import pytest

from astro_mine.bench.baseline import BaselinePolicy, run
from astro_mine.bench.zoo import (
    ANCHOR_SCENARIO_ID,
    ZooIntegrityError,
    check_scenario_immutable,
    list_scenarios,
    load_scenario,
    resolve_scenario,
    verify_zoo,
)

SPRINT_ID = "lunar-polar-ice-prospecting-sprint-v1"
ENDURANCE_ID = "lunar-polar-ice-endurance-v1"
#: The surrogate-fidelity task (bench#31). Like the others it reuses the anchor's world and prior by
#: hash — but it pins the excavator at **0.2.0**, the first revision to declare a `tool` contact
#: element, without which no asset in the library reaches the granular contact ladder at all
#: (astro-mine-fleet#37). Its headline result is a *speedup*, not a scorecard; see its RESULTS.md.
FIDELITY_ID = "lunar-polar-ice-excavation-fidelity-v1"
NEW_IDS = (SPRINT_ID, ENDURANCE_ID, FIDELITY_ID)

# Pinned historical spec hashes, keyed by the anchor's spec_version. A published spec is immutable:
# editing one in place silently invalidates every leaderboard pinned to it (bench.md §8, §5). Keying
# by spec_version enforces the *rule* rather than freezing one literal — an in-place edit that
# forgets to bump spec_version still trips the guard, and a deliberate re-pin must declare a new
# version.
#
# 0.1.1 -> 0.2.0: pinned the Link ContactPlan digest (#28) and the Core schema digest (#39). No
# leaderboard exists for 0.1.1 — the repos are private and nothing has been published to an external
# leaderboard — so no historical result is invalidated by the re-pin.
ANCHOR_SPEC_HASHES = {
    "0.1.1": "sha256:195c24b943abe0c9aeaae7137542d306ae4d6e6a8fda243aa2e3dc6cbf5d95cc",
    "0.2.0": "sha256:23da01297df782c636e259057214f7344565a9c516fd0afc535b7a5cf8bb06f0",
    # 0.3.0 re-pins the world to 0.4.0 (which ships its horizon map — astro-mine-worlds#46) and the
    # contact plan to 0.2.0 (whose nodes carry the Fleet SADF asset ids — astro-mine-link#30, and
    # without which `comms_robustness` was structurally unscorable), and declares the episode's
    # `start_epoch` — without which Sim ran the anchor at its J2000 default, thirty years before the
    # plan's window, so every contact was inactive and `comms_robustness` scored a confident 0.0.
    # All three are content changes, so the spec is a new immutable version rather than an edit of
    # 0.2.0 (bench.md §8).
    "0.3.0": "sha256:eaab1106055f8f4947a05c2da791b201a3ed3a2528f5e8c97d1f1617be1fda30",
    # 0.4.0 re-pins the contact plan to 0.3.0, whose epoch window is extended from 24 h to the full
    # 30-day mission the episode runs (astro-mine-link#34) — so `comms_robustness` scores over the
    # whole episode instead of counting the ~29 days a 24 h plan did not model as denied. A content
    # change, so a new immutable spec_version rather than an edit of 0.3.0 (bench.md §8).
    "0.4.0": "sha256:6c0df3f5921b2521961fe8da114a2c6053dff8c9a6c86a40cf4c58c7f109a33f",
    # 0.5.0 re-pins the six fleet assets and the belief prior to the digests current source produces
    # (published to ghcr.io/astro-mine): the excavator gains a `tool` contact element (0.2.0,
    # astro-mine-fleet#38), the belief prior's GMRF SPDE operator is corrected so alpha=2 yields a
    # valid Matern nu=1 field (1.0.0, astro-mine-prospect#39), and re-pinning Core to v0.3.0
    # (RFC-0009) re-stamped every producer manifest — so 0.4.0's fleet/prospect digests no longer
    # reproduce from source. The world (0.4.0) and the Core schema digest are unchanged and keep
    # their pins; the link ContactPlan is rebuilt from unchanged inputs. Content changes, so a new
    # immutable spec_version rather than an edit of 0.4.0 (bench.md §8).
    "0.5.0": "sha256:a1f4c1c8ff861934c20ed14983171ee5bbd3608f7dc9eae8f8e66a9aeaff407b",
    # --- the content address changed basis at 0.6.0 ------------------------------------------
    # Every entry at or below 0.5.0 was computed over a *full* model dump, which serialized
    # defaulted fields (`"budgets": {"wall_clock_seconds": null, ...}`). Under that basis,
    # appending any optional field to ScenarioSpec re-identified every scenario in the zoo,
    # including historical ones — the exact recomputation bench.md §8 says the zoo's
    # add-only discipline exists to avoid. From 0.6.0 the canonical form excludes defaults
    # (`ScenarioSpec.canonical_json`), so a spec that does not exercise an optional block
    # hashes as though the block did not exist and future additive growth is hash-stable.
    #
    # The entries above are therefore **historical records, not reproducible from current
    # code** — recomputing 0.5.0 under today's basis yields
    # sha256:645279277323096990959ba499b764215cd11ef6f83c337b25742d706bf036d6, not the value
    # recorded above. They are kept because that is what those versions actually addressed to
    # while they were the live spec, and rewriting them would erase the record rather than
    # correct it. Nothing recomputes them: the immutability test checks only the version the
    # zoo currently ships. This basis change is a one-time cost, taken deliberately while the
    # leaderboard is still pre-public and no published result is bound to these digests.
    #
    # 0.6.0 pins placement and the belief/discovery scoring parameters — the two things the
    # spec could not express, so siting was a jitter derived from the content digest
    # (astro-mine-bench#63) and the belief metrics' thresholds were unreachable from the task
    # (astro-mine-sim#66). No content pin moved: the world, fleet, prospect and link digests
    # are identical to 0.5.0's. A schema change, so a new immutable spec_version rather than
    # an edit of 0.5.0 (bench.md §8).
    "0.6.0": "sha256:d7b39f61b46a502a3c5748f0fc5481e77703254892175520ecd3a447690f5cde",
    # 0.7.0 drops the four sites' `elevation_m`, so each snaps to the pinned DEM instead. 0.6.0
    # had copied astro-mine-link's anchor elevations along with its lat/lon, but those figures are
    # descriptive of Link's own geometry and were never sampled from this world bundle. Measured
    # against the pinned DEM they are off by -4436 m (prospecting-rover), -3276 m (excavator),
    # +1617 m (hauler) and +301 m (isru-plant) — so two of the four assets would have started
    # kilometres *underneath* the terrain, where `line_of_sight` is occluded by the ground they
    # are buried in. Exactly the defect astro-mine-bench#31 fixed for the ring layout,
    # reintroduced by way of a different source. Snapping is also strictly more reproducible: the
    # terrain is pinned by hash, so a site cannot drift from the world the run actually uses.
    # A spec change, so a new immutable spec_version rather than an edit of 0.6.0 (bench.md §8).
    "0.7.0": "sha256:7c23ba063fd82c321d353a99afc3c533fc9deecbb957f28c2e518b7549f8de9e",
    # 0.8.0 re-pins the ISRU plant to 0.2.0, the first revision to declare a `water_gauge`
    # (`resource_storage`, species `water`, unit `kg`). Without it the plant filled a tank nothing
    # could read — `water_mass` matches a reading's species and unit, so a plant holding 16.7 kg
    # scored an empty sum indistinguishable from a swarm that produced nothing
    # (astro-mine-fleet#40). The last of four separate reasons this metric read zero on every
    # Sim-backed run: gauge dispatch (astro-mine-sim#61), uncoupled extraction (astro-mine-sim#64),
    # no declared gauge, and a template unit the platform does not know. A content change, so a new
    # immutable spec_version rather than an edit of 0.7.0 (bench.md §8).
    "0.8.0": "sha256:e811049a392e06c7b4291a3d4b01d029ef75a8f1bd39979f20e787888751ca71",
}


def test_new_scenarios_are_discovered_with_no_loader_change() -> None:
    catalog = set(list_scenarios())
    assert {ANCHOR_SCENARIO_ID, *NEW_IDS} <= catalog


@pytest.mark.parametrize("scenario_id", NEW_IDS)
def test_new_scenario_resolves_to_a_distinct_task(scenario_id: str) -> None:
    spec = load_scenario(scenario_id)
    resolved = resolve_scenario(spec)
    anchor = resolve_scenario(load_scenario(ANCHOR_SCENARIO_ID))
    # A distinct task: same content by hash, but a different spec hash and scenario identity.
    assert resolved.scenario_hash != anchor.scenario_hash
    assert spec.spec_hash != load_scenario(ANCHOR_SCENARIO_ID).spec_hash


@pytest.mark.parametrize("scenario_id", NEW_IDS)
def test_new_scenario_evaluates_on_the_existing_harness(scenario_id: str) -> None:
    spec = load_scenario(scenario_id)
    card = run(spec, BaselinePolicy(), seeds=spec.seeds.public)
    assert card.scenario_id == scenario_id
    # Scores exactly the pinned metric set, in order — the primary metric ranks the leaderboard.
    assert [m.metric for m in card.metrics] == [ref.name for ref in spec.metrics]
    assert card.metrics[0].seeds == spec.seeds.public


def test_new_scenarios_reuse_the_anchor_world_by_hash() -> None:
    anchor_world = load_scenario(ANCHOR_SCENARIO_ID).content.world.content_hash
    for scenario_id in NEW_IDS:
        assert load_scenario(scenario_id).content.world.content_hash == anchor_world


def test_zoo_is_immutable_and_content_addressed() -> None:
    hashes = verify_zoo()
    # Every catalog entry passes the immutability/content-addressing check.
    assert set(hashes) == set(list_scenarios())
    # Historical validity: the anchor's spec hash is exactly what its spec_version pins. An in-place
    # edit that forgets to bump spec_version trips this, as does a bump that forgets to re-pin.
    anchor_version = load_scenario(ANCHOR_SCENARIO_ID).spec_version
    assert hashes[ANCHOR_SCENARIO_ID] == ANCHOR_SPEC_HASHES[anchor_version]
    # Distinct task identities across the (grown) catalog.
    assert len(set(hashes.values())) == len(hashes)


def test_integrity_check_rejects_an_unpinned_content_ref() -> None:
    spec = load_scenario(SPRINT_ID)
    tampered = spec.model_copy(
        update={
            "content": spec.content.model_copy(
                update={
                    "world": spec.content.world.model_copy(update={"content_hash": "not-a-digest"})
                }
            )
        }
    )
    # Pydantic normalizes a bare hex, but a non-hex string cannot be a sha256 pin — caught here.
    with pytest.raises((ZooIntegrityError, ValueError)):
        check_scenario_immutable(tampered)


def test_a_scenario_that_pins_a_contact_plan_declares_when_it_runs() -> None:
    """A pinned ContactPlan is a plan over a *window of time* (bench#48).

    Whether it applies at all depends on when the episode runs, and nothing used to say. Bench's
    EpisodeSpec carried no start epoch, so Sim fell back to its own default (J2000, TDB 0.0) while
    the anchor's plan covers 24 h at TDB 946_728_000 (2030-01-01) — thirty years apart. Every
    contact interval was inactive at every tick, `earth_contact` was false forever, and
    `comms_robustness` scored a confident 0.0 instead of failing.

    So the rule this pins: **if a scenario pins a `link` ref, it must declare `episode.start_epoch`,
    and that epoch must fall inside the plan's window.** A scenario that pins no plan may leave it
    unset — the runner picks the epoch, which is the pre-existing behaviour and is correct when
    nothing time-dependent is pinned.
    """
    for scenario_id in list_scenarios():
        spec = load_scenario(scenario_id)
        if spec.content.link is None:
            continue
        start = spec.episode.start_epoch
        assert start is not None, (
            f"{scenario_id} pins a ContactPlan but declares no episode.start_epoch, so nothing "
            "fixes whether the plan's contact windows are even reachable"
        )
        assert start.scale.value == "tdb"
        # The anchor's plan window: the 30-day mission from 2030-01-01T00:00:00 TDB (link's
        # ANCHOR_EPOCH_WINDOW, extended to the episode horizon in plan 0.3.0 — astro-mine-link#34).
        assert 946_728_000.0 <= start.tdb_seconds < 946_728_000.0 + 2_592_000.0, (
            f"{scenario_id}'s start_epoch {start.tdb_seconds} is outside the pinned plan's window"
        )


def test_a_scenario_without_a_contact_plan_need_not_declare_an_epoch() -> None:
    """The other half of the contract: the epoch is required by the *plan*, not by every task."""
    for scenario_id in (SPRINT_ID, "lunar-polar-ice-endurance-v1"):
        spec = load_scenario(scenario_id)
        assert spec.content.link is None
        assert spec.episode.start_epoch is None


# --- 0.6.0: the anchor pins its siting and its scoring parameters ---------------------------


def test_the_anchor_pins_where_the_swarm_stands() -> None:
    # Before 0.6.0 siting was a `sha256(scenario_hash:index)` jitter on a fixed ring, so a re-pin
    # moved the swarm and every position-dependent metric with it (astro-mine-bench#63).
    spec = load_scenario(ANCHOR_SCENARIO_ID)
    assert spec.placement is not None
    placed = {site.asset for site in spec.placement.sites}
    # The four *surface* assets. The relay-orbiter is on orbit and the lander is a delivery
    # vehicle, so neither is sited here — a scenario may place only what it cares about.
    assert placed == {
        "astro-mine.fleet.prospecting-rover",
        "astro-mine.fleet.excavator",
        "astro-mine.fleet.hauler",
        "astro-mine.fleet.isru-plant",
    }
    # Every site is a real polar site on the pinned world, not a placeholder at the origin.
    for site in spec.placement.sites:
        assert site.lat_deg < -89.5, f"{site.asset} is not near the south pole"
        # Elevation is deliberately *not* pinned — it comes from the pinned DEM. 0.6.0 copied
        # astro-mine-link's figures, which are descriptive of Link's own geometry and were never
        # sampled from this world: they sit up to 4.4 km off its terrain and would have started
        # the rover and the excavator kilometres underground, occluded by the ground they were
        # buried in. The terrain is pinned by hash, so sampling it cannot drift from the world
        # the run uses.
        assert site.elevation_m is None


def test_the_anchor_sites_are_spread_across_the_region_not_clustered() -> None:
    # The lat/lon are astro-mine-link's anchor sites, which is the point: the contact plan this
    # scenario pins was computed against exactly this siting, so comms geometry and physics now
    # describe the same swarm. Bench cannot import Link to assert that directly (one-way
    # dependency, conventions.md §1.1), so the structural property is asserted instead — the swarm
    # is spread across the region rather than clustered on one 25 m ring at the pole.
    spec = load_scenario(ANCHOR_SCENARIO_ID)
    assert spec.placement is not None
    longitudes = {site.lon_deg for site in spec.placement.sites}
    assert len(longitudes) == 4, "the sites must not share a meridian"
    latitudes = sorted(site.lat_deg for site in spec.placement.sites)
    # The rover prospects deepest into the polar cap; the plant sits furthest out, on the ridge.
    # Kilometres apart on the surface, not centimetres.
    assert latitudes[-1] - latitudes[0] > 0.15, "the sites are not meaningfully separated"


def test_the_anchor_pins_the_belief_and_discovery_scoring_parameters() -> None:
    # Each of these had a runner-side default that produced a confident wrong number rather than
    # an abstention (astro-mine-sim#66).
    spec = load_scenario(ANCHOR_SCENARIO_ID)
    assert spec.scoring is not None
    # 250 m is the pinned prior's own grid pitch (a 60x60 km box at 240x240 cells), so the PSR
    # area is reported in real square metres rather than in cell counts.
    assert spec.scoring.cell_area_m2 == 250.0 * 250.0
    # Strictly positive, or the metrics cannot distinguish "characterized" from "impossible".
    assert spec.scoring.characterized_variance_threshold is not None
    assert spec.scoring.characterized_variance_threshold > 0.0
    assert spec.scoring.discovery_threshold is not None
    assert spec.scoring.discovery_threshold > 0.0
    # The PSR window covers the polar cap the two shadowed sites sit in.
    assert spec.scoring.psr_region is not None
    lat_min, lat_max = spec.scoring.psr_region.lat_deg
    assert lat_min == -90.0 and lat_max < -89.5


def test_the_anchor_re_pins_only_the_plant_at_0_8_0() -> None:
    # 0.6.0 and 0.7.0 were *spec* changes that moved no content. 0.8.0 moves exactly one pin — the
    # ISRU plant, to the revision that declares its water gauge (astro-mine-fleet#40) — and nothing
    # else. A re-pin that quietly dragged the world or the contact plan with it would change what
    # the task runs against for reasons nobody reviewed.
    spec = load_scenario(ANCHOR_SCENARIO_ID)
    assert spec.spec_version == "0.8.0"
    plant = next(r for r in spec.content.fleet if r.id == "astro-mine.fleet.isru-plant")
    assert plant.content_hash == (
        "sha256:3b13364714c41693c4523c0bed6303a77d3d6348a3b8a549561d2addb57fa7eb"
    )
    # Every other fleet pin is 0.5.0's, verbatim.
    others = {r.id: r.content_hash for r in spec.content.fleet if r.id != plant.id}
    assert others == {
        "astro-mine.fleet.relay-orbiter": (
            "sha256:47d74f0e851826718a5064d101983ea187d40d9d8ac51488c023755c29141480"
        ),
        "astro-mine.fleet.lander": (
            "sha256:f0270170c2cac90953392b5cd70d79a2d7abf3799454d29f46fa6a4cea4bdc90"
        ),
        "astro-mine.fleet.prospecting-rover": (
            "sha256:74db543ff304a2fe4cef9c421873a3375243019acd0314f0ef9aae6e6fb04629"
        ),
        "astro-mine.fleet.excavator": (
            "sha256:d576d7844625b25baec6496a45ad0a18a4da945c2cb0a5ebfa5c957d62fd5d35"
        ),
        "astro-mine.fleet.hauler": (
            "sha256:5e0c5b6d075d81c06507f283f390b74afd92953ec106d67656e00c31f7c63aab"
        ),
    }
    assert spec.content.world.content_hash == (
        "sha256:fd14f5f6e618a9c01c36028e642e6efdbee449a5e96bac832047654913a4fd2c"
    )
    assert spec.content.link is not None
    assert spec.content.link.content_hash == (
        "sha256:0a0eb64e4a0a25fe9f767209f1e8baca4fb4df4c4b1313c0b5f78a5bbf27f9ed"
    )
    assert spec.core_schema_digest == (
        "sha256:2ebc6353bda4ecd0ed14b39ef04747b84a8fa79f8a094146f74ee027cbf07980"
    )
