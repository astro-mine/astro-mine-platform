"""ScenarioSpec schema: validation, content-hash references, and the spec hash.

Covers the acceptance criterion's first half — *changing any input changes the spec hash* —
plus the frozen/extra-forbid house contract and hash-reference normalization.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from astro_mine.bench.scenario import (
    BudgetSpec,
    ContentPins,
    ContentRef,
    LatLonRegion,
    PlacementSpec,
    ScenarioSpec,
    ScoringSpec,
    SeedSet,
    SitePlacement,
    TerminationSpec,
)
from astro_mine.bench.scenario._hash import normalize_sha256
from tests.bench._factories import make_scenario_spec, sha256_of


def test_valid_spec_constructs_and_exposes_hash() -> None:
    spec = make_scenario_spec()
    assert spec.scenario_id == "lunar-polar-ice-prospecting-v1"
    assert spec.spec_hash.startswith("sha256:")
    assert len(spec.spec_hash) == len("sha256:") + 64


def test_spec_is_frozen() -> None:
    spec = make_scenario_spec()
    with pytest.raises(ValidationError):
        spec.scenario_id = "mutated"


def test_spec_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        make_scenario_spec(unexpected_field="nope")


def test_spec_hash_is_deterministic() -> None:
    assert make_scenario_spec().spec_hash == make_scenario_spec().spec_hash


def test_content_hash_normalizes_bare_hex_and_prefixed() -> None:
    bare = "A" * 64
    ref = ContentRef(id="x", content_hash=bare)
    assert ref.content_hash == "sha256:" + "a" * 64
    assert (
        ContentRef(id="x", content_hash="sha256:" + "b" * 64).content_hash == "sha256:" + "b" * 64
    )


def test_content_hash_rejects_malformed() -> None:
    with pytest.raises(ValidationError):
        ContentRef(id="x", content_hash="not-a-hash")


def test_normalize_sha256_rejects_malformed_directly() -> None:
    with pytest.raises(ValueError, match="not a sha256"):
        normalize_sha256("sha256:tooshort")


def test_seedset_normalizes_heldout_commit() -> None:
    seeds = SeedSet(public=(1,), heldout_commit="F" * 64)
    assert seeds.heldout_commit == "sha256:" + "f" * 64
    assert SeedSet(public=(1,)).heldout_commit is None


def test_core_interface_must_be_nonempty() -> None:
    with pytest.raises(ValidationError):
        make_scenario_spec(core_interface={})


def test_core_interface_rejects_empty_key() -> None:
    with pytest.raises(ValidationError):
        make_scenario_spec(core_interface={"": "0.1.0"})


def test_core_interface_rejects_non_semver() -> None:
    with pytest.raises(ValidationError):
        make_scenario_spec(core_interface={"env": "one-point-oh"})


def test_metric_version_must_be_semver() -> None:
    from astro_mine.bench.scenario import MetricRef

    with pytest.raises(ValidationError):
        MetricRef(name="water_mass", version="bad")


def test_spec_version_must_be_semver() -> None:
    with pytest.raises(ValidationError):
        make_scenario_spec(spec_version="nope")


def test_fleet_requires_at_least_one_asset() -> None:
    with pytest.raises(ValidationError):
        make_scenario_spec(
            content=ContentPins(
                world=ContentRef(id="w", content_hash=sha256_of("a")),
                fleet=(),
            )
        )


def test_content_refs_includes_link_when_present() -> None:
    spec = make_scenario_spec(
        content=ContentPins(
            world=ContentRef(id="w", content_hash=sha256_of("a")),
            fleet=(ContentRef(id="rover", content_hash=sha256_of("b")),),
            prospect=(ContentRef(id="prior", content_hash=sha256_of("c")),),
            link=ContentRef(id="contact-plan", content_hash=sha256_of("d")),
        )
    )
    ids = [ref.id for ref in spec.content_refs()]
    assert ids == ["w", "rover", "prior", "contact-plan"]


def test_content_refs_omits_link_by_default() -> None:
    spec = make_scenario_spec()
    assert all(ref.id != "contact-plan" for ref in spec.content_refs())
    assert len(spec.content_refs()) == 3  # world + 1 fleet + 1 prospect


def test_optional_blocks_have_working_defaults() -> None:
    spec = make_scenario_spec()
    assert spec.termination == TerminationSpec()
    assert spec.budgets == BudgetSpec()
    assert spec.budgets.wall_clock_seconds is None


def test_budget_and_termination_accept_values() -> None:
    spec = make_scenario_spec(
        termination=TerminationSpec(conditions=("all_psr_characterized",), max_steps=9_000),
        budgets=BudgetSpec(wall_clock_seconds=3600.0, sim_steps=1_000_000, compute_units=8.0),
    )
    assert spec.termination.conditions == ("all_psr_characterized",)
    assert spec.budgets.sim_steps == 1_000_000


def test_json_round_trip_preserves_hash() -> None:
    spec = make_scenario_spec()
    reloaded = ScenarioSpec.model_validate(spec.model_dump(mode="json"))
    assert reloaded.spec_hash == spec.spec_hash


def test_json_schema_is_emitted() -> None:
    schema = ScenarioSpec.json_schema()
    assert schema["type"] == "object"
    assert "core_interface" in schema["properties"]


def test_canonical_json_is_key_sorted_and_compact() -> None:
    text = make_scenario_spec().canonical_json
    assert text.startswith("{") and ", " not in text  # compact separators
    assert '"core_interface"' in text


def test_any_input_change_changes_spec_hash() -> None:
    base = make_scenario_spec().spec_hash
    from astro_mine.bench.scenario import EpisodeSpec, MetricRef

    variants = [
        make_scenario_spec(scenario_id="other-scenario"),
        make_scenario_spec(name="Other Name"),
        make_scenario_spec(core_interface={"env": "0.1.0"}),
        make_scenario_spec(seeds=SeedSet(public=(9,))),
        make_scenario_spec(seeds=SeedSet(public=(1, 2, 3), heldout_commit=sha256_of("e"))),
        make_scenario_spec(episode=EpisodeSpec(horizon_steps=42)),
        make_scenario_spec(metrics=(MetricRef(name="energy_per_kg"),)),
        make_scenario_spec(budgets=BudgetSpec(sim_steps=5)),
        make_scenario_spec(
            content=ContentPins(
                world=ContentRef(id="shackleton-v1", content_hash=sha256_of("f")),
                fleet=(
                    ContentRef(
                        id="prospecting-rover", content_hash=sha256_of("b")
                    ),
                ),
                prospect=(ContentRef(id="ice-prior-v1", content_hash=sha256_of("c")),),
            )
        ),
    ]
    hashes = {v.spec_hash for v in variants}
    assert base not in hashes
    assert len(hashes) == len(variants)  # each change is distinct


# --- the content address excludes defaults (astro-mine-bench#63) ----------------------------


def test_an_omitted_optional_block_stays_out_of_the_content_address() -> None:
    # The property the exclude-defaults basis buys: a spec that does not exercise an optional
    # block serializes as though the block did not exist. This is what lets ScenarioSpec grow
    # additively without re-identifying every historical scenario (bench.md §8).
    dumped = json.loads(make_scenario_spec().canonical_json)
    assert "placement" not in dumped
    assert "scoring" not in dumped
    assert "budgets" not in dumped  # defaulted sub-model, likewise absent


def test_setting_a_block_to_its_own_default_is_not_a_new_task() -> None:
    # Explicitly authoring the default must not fork the content address — otherwise "omitted"
    # and "stated but unchanged" would be two different tasks with identical semantics.
    omitted = make_scenario_spec()
    stated = make_scenario_spec(placement=None, scoring=None, budgets=BudgetSpec())
    assert stated.spec_hash == omitted.spec_hash


def test_exercising_an_optional_block_does_change_the_content_address() -> None:
    # The other half: a scenario that actually pins placement is a different task.
    placed = make_scenario_spec(placement=_placement("prospecting-rover"))
    assert placed.spec_hash != make_scenario_spec().spec_hash


# --- placement (astro-mine-bench#63) --------------------------------------------------------


def _placement(asset: str, **overrides: object) -> PlacementSpec:
    site = {"asset": asset, "lat_deg": -89.9, "lon_deg": 0.0, "elevation_m": -3800.0}
    site.update(overrides)
    return PlacementSpec(sites=(SitePlacement(**site),))  # type: ignore[arg-type]


def test_placement_must_reference_an_asset_the_scenario_pins() -> None:
    # A site naming an unpinned asset is inert rather than wrong at runtime — the runner simply
    # never matches it and silently falls back to its own layout. Caught here instead.
    with pytest.raises(ValidationError, match="does not pin"):
        make_scenario_spec(placement=_placement("excavator"))


def test_placement_rejects_a_repeated_asset() -> None:
    rover = "prospecting-rover"
    with pytest.raises(ValidationError, match="unique per asset"):
        PlacementSpec(
            sites=(
                SitePlacement(asset=rover, lat_deg=-89.9, lon_deg=0.0),
                SitePlacement(asset=rover, lat_deg=-89.8, lon_deg=10.0),
            )
        )


def test_longitude_folds_into_one_revolution() -> None:
    # -45 and 315 are the same meridian, so they must be the same task.
    west = make_scenario_spec(
        placement=_placement("prospecting-rover", lon_deg=-45.0)
    )
    east = make_scenario_spec(
        placement=_placement("prospecting-rover", lon_deg=315.0)
    )
    assert west.placement is not None and west.placement.sites[0].lon_deg == 315.0
    assert west.spec_hash == east.spec_hash


def test_placement_rejects_an_off_body_latitude() -> None:
    with pytest.raises(ValidationError):
        SitePlacement(asset="a", lat_deg=-91.0, lon_deg=0.0)


def test_elevation_may_be_omitted_and_then_comes_from_the_pinned_terrain() -> None:
    site = SitePlacement(asset="a", lat_deg=-89.9, lon_deg=0.0)
    assert site.elevation_m is None


# --- scoring parameters (astro-mine-sim#66) -------------------------------------------------


def test_an_unsatisfiable_variance_threshold_is_rejected() -> None:
    # 0.0 is the hostile runner-side default this block exists to displace: no posterior
    # variance can satisfy it, so psr_area_characterized reports a confident 0.0 m² instead of
    # abstaining. A scenario must not be able to pin it.
    with pytest.raises(ValidationError):
        ScoringSpec(characterized_variance_threshold=0.0)
    with pytest.raises(ValidationError):
        ScoringSpec(discovery_threshold=0.0)
    with pytest.raises(ValidationError):
        ScoringSpec(cell_area_m2=0.0)


def test_unpinned_scoring_parameters_are_none_not_zero() -> None:
    # "the scenario does not pin this" must stay distinguishable from "the scenario pins zero".
    scoring = ScoringSpec()
    assert scoring.characterized_variance_threshold is None
    assert scoring.discovery_threshold is None
    assert scoring.cell_area_m2 is None
    assert scoring.psr_region is None


def test_a_psr_region_needs_increasing_bounds() -> None:
    with pytest.raises(ValidationError, match="increasing bounds"):
        LatLonRegion(lat_deg=(-89.0, -90.0), lon_deg=(0.0, 360.0))
    with pytest.raises(ValidationError, match="increasing bounds"):
        LatLonRegion(lat_deg=(-90.0, -89.0), lon_deg=(360.0, 0.0))


def test_a_psr_region_may_cross_the_prime_meridian() -> None:
    # Authored as an unwrapped window so min < max stays true and the region is unambiguous.
    region = LatLonRegion(lat_deg=(-90.0, -89.0), lon_deg=(350.0, 370.0))
    assert region.lon_deg == (350.0, 370.0)


def test_a_psr_region_cannot_span_more_than_a_revolution() -> None:
    with pytest.raises(ValidationError, match="full revolution"):
        LatLonRegion(lat_deg=(-90.0, -89.0), lon_deg=(0.0, 400.0))
