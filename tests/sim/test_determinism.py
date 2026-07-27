"""RM-P0-SIM-10 — the determinism gates: seeded reproducibility + golden traces.

Proves determinism is enforced, not hoped for (sim.md §2.4, §10): same inputs + same seed reproduce
byte-for-byte; a pinned golden hash catches unintended dynamics drift; and a deliberately
non-deterministic run (an unseeded resource field) trips the gate — CI fails on non-reproducibility.
"""

from __future__ import annotations

import random

import pytest

from astro_mine.core.sadf.enums import SensorKind
from astro_mine.core.sadf.model import ObservationModel, ResourceTarget, Sensor
from astro_mine.sim.runtime import AgentSpec, Scenario
from astro_mine.sim.sensors import ReferenceResourceField
from astro_mine.sim.validation import (
    DeterminismError,
    assert_matches_golden,
    assert_reproducible,
    golden_hash,
)

# A BIT_EXACT kinematic reference scenario and its pinned content hash. The kinematic engine's
# jitter is a hashlib-seeded stdlib `random` stream (byte-for-byte across builds), so this golden is
# portable; an unintended change to the dynamics, the provenance, or an engine version breaks it.
# (Re-pinned for RM-P1-SIM-02: the additive optional ``AgentSpec.isru`` field appears in the
# scenario digest, an intended provenance change — the rollout dynamics are unchanged.)
_GOLDEN_SCENARIO = Scenario(
    name="determinism-golden",
    agents=(
        AgentSpec(agent_id="rover-a", velocity_mps=(1.0, 0.5, 0.0), battery_soc_j=500.0),
        AgentSpec(
            agent_id="rover-b",
            initial_position_m=(10.0, 0.0, 0.0),
            velocity_mps=(-0.5, 0.0, 0.0),
            battery_soc_j=500.0,
        ),
    ),
    seed=2026,
    horizon_steps=12,
)
# Re-pinned for RM-P1-SIM-03: FidelityPolicy gained the additive `error_budget` field, so the
# scenario (which carries the policy) serializes with `error_budget: null` and content-addresses
# anew. An intentional, reviewed spec-schema change — not a determinism regression.
#
# Re-pinned again for #64: `AgentSpec` gained the additive optional `cargo_capacity_kg`, which
# names how much regolith an asset can carry along the value chain. It serializes as
# `cargo_capacity_kg: null` for these two rovers, so the scenario content-addresses anew. The
# rollout dynamics are untouched — the third schema-growth re-pin of this golden, and the third
# time the physics did not move.
_GOLDEN_HASH = "a6df09c93c0e2d0ac9e03b15217c2f9a9e192c5ab6f70a8e979e1a82630594b7"


class _DriftingField(ReferenceResourceField):
    """A resource field whose value drifts on every read — a deliberately non-deterministic input.

    It ignores the seeded sensor stream and draws from the *global* RNG, so two same-seed runs see
    different sensor readings — exactly the kind of non-determinism the gate must catch."""

    def sample(self, position, *, n=1, seed=None, epoch=None):  # type: ignore[no-untyped-def]
        return tuple(random.random() for _ in range(n))  # deliberately unseeded for the teeth test


def _sensing_scenario() -> Scenario:
    sensor = Sensor(
        name="neutron",
        kind=SensorKind.NEUTRON_SPECTROMETER,
        frame="body",
        observation_model=ObservationModel(noise_sigma=0.01),
        resource=ResourceTarget(species="water_equivalent_hydrogen", si_unit="mass_fraction"),
    )
    return Scenario(
        name="sensing",
        agents=(AgentSpec(agent_id="prospector", battery_soc_j=500.0, sensors=(sensor,)),),
        seed=1,
        horizon_steps=4,
    )


# --- reproducibility --------------------------------------------------------------------------


def test_same_seed_runs_are_reproducible() -> None:
    digest = assert_reproducible(_GOLDEN_SCENARIO, runs=3)
    assert digest == golden_hash(_GOLDEN_SCENARIO)


def test_assert_reproducible_needs_at_least_two_runs() -> None:
    with pytest.raises(ValueError, match="at least 2 runs"):
        assert_reproducible(_GOLDEN_SCENARIO, runs=1)


def test_golden_hash_is_seed_sensitive() -> None:
    assert golden_hash(_GOLDEN_SCENARIO, seed=1) != golden_hash(_GOLDEN_SCENARIO, seed=2)


# --- golden trace -----------------------------------------------------------------------------


def test_reference_scenario_matches_its_pinned_golden_hash() -> None:
    assert assert_matches_golden(_GOLDEN_SCENARIO, _GOLDEN_HASH) == _GOLDEN_HASH


def test_a_drifted_result_fails_the_golden_gate() -> None:
    with pytest.raises(DeterminismError, match="drifted from its golden hash"):
        assert_matches_golden(_GOLDEN_SCENARIO, "0" * 64)


# --- the teeth: a non-deterministic run trips the gate ----------------------------------------


def test_non_deterministic_field_trips_the_reproducibility_gate() -> None:
    with pytest.raises(DeterminismError, match="non-deterministic"):
        assert_reproducible(_sensing_scenario(), resource_field=_DriftingField())
