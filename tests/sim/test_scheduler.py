"""RM-P0-SIM-05 — the first-cut rule-based multi-fidelity scheduler.

Covers the acceptance criteria: the scheduler selects fidelity tiers by rule and records the
error implied by each substitution, and a run can be pinned to a tier for reproducibility. The
error-budget-driven scheduler (numeric, oracle-validated budgets) is Phase 1 and out of scope.
"""

from __future__ import annotations

import pytest

from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier
from astro_mine.core.sadf.model import FidelityProfile
from astro_mine.sim.runtime import (
    AgentSpec,
    GranularDynamics,
    MobilityDynamics,
    Scenario,
    run_episode,
)
from astro_mine.sim.scheduler import (
    FidelityPolicy,
    FidelityRule,
    FidelitySelection,
    Scheduler,
    select_fidelity,
)


def _profile(tier: FidelityTier, determinism: DeterminismClass) -> FidelityProfile:
    return FidelityProfile(tier=tier, determinism_class=determinism)


#: A three-rung ladder (the shape Fleet's multi-fidelity profiles supply, RM-P0-FLEET-05).
LADDER = (
    _profile(FidelityTier.MASSMODEL, DeterminismClass.TOLERANCE),
    _profile(FidelityTier.KINEMATIC, DeterminismClass.TOLERANCE),
    _profile(FidelityTier.ARTICULATED, DeterminismClass.BIT_EXACT),
)


# --- rule-based selection + implied-error tracking -------------------------------


def test_coarsest_rule_picks_the_cheapest_tier_and_tracks_the_implied_error() -> None:
    selection = select_fidelity("rover", LADDER, FidelityPolicy(rule=FidelityRule.COARSEST))
    assert selection.tier is FidelityTier.MASSMODEL
    assert selection.reference_tier is FidelityTier.ARTICULATED  # finest available
    assert selection.implied_error_rungs == 2  # massmodel is two rungs below articulated
    assert selection.determinism_class is DeterminismClass.TOLERANCE  # the massmodel rung's
    assert selection.pinned is False


def test_finest_rule_picks_the_reference_tier_with_zero_implied_error() -> None:
    selection = select_fidelity("rover", LADDER, FidelityPolicy(rule=FidelityRule.FINEST))
    assert selection.tier is FidelityTier.ARTICULATED
    assert selection.reference_tier is FidelityTier.ARTICULATED
    assert selection.implied_error_rungs == 0
    assert selection.determinism_class is DeterminismClass.BIT_EXACT


def test_a_single_tier_ladder_has_no_implied_error() -> None:
    one = (_profile(FidelityTier.MASSMODEL, DeterminismClass.BIT_EXACT),)
    selection = select_fidelity("relay", one, FidelityPolicy())
    assert selection.tier is selection.reference_tier is FidelityTier.MASSMODEL
    assert selection.implied_error_rungs == 0


# --- pinning a run to a tier (reproducibility) -----------------------------------


def test_run_wide_pin_overrides_the_rule_and_marks_the_selection_pinned() -> None:
    policy = FidelityPolicy(rule=FidelityRule.COARSEST, pinned_tier=FidelityTier.KINEMATIC)
    selection = select_fidelity("rover", LADDER, policy)
    assert selection.tier is FidelityTier.KINEMATIC and selection.pinned is True
    assert selection.implied_error_rungs == 1  # kinematic is one rung below articulated


def test_per_agent_pin_overrides_the_run_wide_pin() -> None:
    policy = FidelityPolicy(
        pinned_tier=FidelityTier.MASSMODEL,
        agent_pins={"rover": FidelityTier.ARTICULATED},
    )
    assert select_fidelity("rover", LADDER, policy).tier is FidelityTier.ARTICULATED
    assert select_fidelity("other", LADDER, policy).tier is FidelityTier.MASSMODEL


def test_pinning_a_tier_the_agent_does_not_offer_fails_loudly() -> None:
    policy = FidelityPolicy(pinned_tier=FidelityTier.SURROGATE)  # not in LADDER
    with pytest.raises(ValueError, match="cannot pin agent 'rover' to fidelity tier 'surrogate'"):
        select_fidelity("rover", LADDER, policy)


def test_selecting_from_an_empty_ladder_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no available fidelity profiles"):
        select_fidelity("rover", (), FidelityPolicy())


# --- the available ladder: scenario profiles, else the engine's declared tier ----


def test_default_ladder_comes_from_the_agents_regime_engine() -> None:
    # No Fleet profiles declared -> the single tier each engine declares (no drift from the
    # engine descriptors).
    scenario = Scenario(
        name="defaults",
        agents=(
            AgentSpec(
                agent_id="rover",
                dynamics=MobilityDynamics(mass_kg=200.0, max_speed_mps=0.5, max_traction_n=400.0),
            ),
            AgentSpec(agent_id="digger", dynamics=GranularDynamics(max_dig_rate_m3_s=0.01)),
        ),
    )
    selections = Scheduler().resolve(scenario)
    # mobility engine declares KINEMATIC/TOLERANCE; granular declares MASSMODEL/BIT_EXACT.
    assert selections["rover"].tier is FidelityTier.KINEMATIC
    assert selections["rover"].determinism_class is DeterminismClass.TOLERANCE
    assert selections["digger"].tier is FidelityTier.MASSMODEL
    assert selections["digger"].determinism_class is DeterminismClass.BIT_EXACT
    assert all(s.implied_error_rungs == 0 for s in selections.values())  # single-tier ladders


def test_declared_fidelity_profiles_override_the_engine_default() -> None:
    scenario = Scenario(
        name="fleet-profiles",
        agents=(
            AgentSpec(
                agent_id="rover",
                dynamics=MobilityDynamics(mass_kg=200.0, max_speed_mps=0.5, max_traction_n=400.0),
                fidelity_profiles=LADDER,
            ),
        ),
        fidelity=FidelityPolicy(rule=FidelityRule.FINEST),
    )
    selection = Scheduler(scenario.fidelity).resolve(scenario)["rover"]
    assert selection.tier is FidelityTier.ARTICULATED  # finest of the declared ladder


# --- scenario integration + serialization ----------------------------------------


def test_fidelity_policy_is_a_scenario_field_and_defaults_to_coarsest() -> None:
    scenario = Scenario.from_mapping(
        {
            "name": "x",
            "agents": [{"agent_id": "a"}],
            "fidelity": {"rule": "finest"},
        }
    )
    assert scenario.fidelity.rule is FidelityRule.FINEST
    assert Scheduler(scenario.fidelity).policy is scenario.fidelity  # the scheduler exposes it
    assert Scenario(name="d", agents=(AgentSpec(agent_id="a"),)).fidelity.rule is (
        FidelityRule.COARSEST
    )


# --- the selections are recorded in the run provenance (the RM-P0-SIM-09 hook) ----


def test_run_episode_records_fidelity_selections_in_provenance() -> None:
    scenario = Scenario(
        name="provenance",
        agents=(
            AgentSpec(
                agent_id="rover",
                battery_soc_j=100.0,
                dynamics=MobilityDynamics(mass_kg=200.0, max_speed_mps=0.5, max_traction_n=400.0),
                fidelity_profiles=LADDER,
            ),
        ),
        fidelity=FidelityPolicy(pinned_tier=FidelityTier.KINEMATIC),
        horizon_steps=2,
    )
    provenance = run_episode(scenario).provenance
    recorded = provenance["fidelity"]["rover"]
    assert recorded == {
        "tier": "kinematic",
        "reference_tier": "articulated",
        "determinism_class": "tolerance",
        "implied_error_rungs": 1,
        "pinned": True,
    }
    # the selection is seed-independent, so it does not perturb the determinism guarantee:
    # two runs still produce byte-identical traces.
    assert run_episode(scenario).content_hash == run_episode(scenario).content_hash


def test_fidelity_selection_is_immutable() -> None:
    selection = select_fidelity("rover", LADDER, FidelityPolicy())
    assert isinstance(selection, FidelitySelection)
    with pytest.raises(AttributeError):
        selection.tier = FidelityTier.SURROGATE  # type: ignore[misc]


# --- error-budget-driven selection (RM-P1-SIM-03) --------------------------------

#: A budgeted ladder: a learned SURROGATE tier over the ARTICULATED (DEM ground-truth) reference.
BUDGET_LADDER = (
    _profile(FidelityTier.ARTICULATED, DeterminismClass.TOLERANCE),  # the DEM reference
    _profile(FidelityTier.SURROGATE, DeterminismClass.TOLERANCE),
)
#: The surrogate tier's declared per-channel budget (its manifest recommended_error_budget).
SURROGATE_BUDGET = {"pos_x": 0.002, "pos_z": 0.001, "vel_x": 0.12, "vel_z": 0.08}


def test_error_budget_admits_the_surrogate_within_tolerance() -> None:
    policy = FidelityPolicy(error_budget={"pos_x": 0.01, "vel_x": 0.5})  # looser than the budget
    selection = select_fidelity(
        "digger", BUDGET_LADDER, policy, tier_budgets={FidelityTier.SURROGATE: SURROGATE_BUDGET}
    )
    assert selection.tier is FidelityTier.SURROGATE
    assert selection.within_budget is True
    assert selection.admitted_budget == SURROGATE_BUDGET
    outcomes = selection.error_budget_outcomes()
    assert {o.metric for o in outcomes} == {"pos_x", "vel_x"}
    assert all(o.within_budget for o in outcomes)
    assert all(o.tier == "surrogate" for o in outcomes)


def test_error_budget_falls_back_to_the_reference_when_the_surrogate_exceeds_tolerance() -> None:
    # A tolerance tighter than the surrogate's declared pos_x budget -> escalate to ground truth.
    policy = FidelityPolicy(error_budget={"pos_x": 0.0005, "vel_x": 0.5})
    selection = select_fidelity(
        "digger", BUDGET_LADDER, policy, tier_budgets={FidelityTier.SURROGATE: SURROGATE_BUDGET}
    )
    assert selection.tier is FidelityTier.ARTICULATED  # the DEM reference
    assert selection.within_budget is False
    # The rejected surrogate's budget is still reported so the outcome shows why it fell back.
    assert selection.admitted_budget == SURROGATE_BUDGET
    assert all(o.within_budget is False for o in selection.error_budget_outcomes())


def test_error_budget_rejects_a_channel_the_surrogate_never_measured() -> None:
    policy = FidelityPolicy(error_budget={"reaction_force_n": 1.0})  # not in the surrogate budget
    selection = select_fidelity(
        "digger", BUDGET_LADDER, policy, tier_budgets={FidelityTier.SURROGATE: SURROGATE_BUDGET}
    )
    assert selection.tier is FidelityTier.ARTICULATED and selection.within_budget is False


def test_error_budget_policy_without_a_budgeted_tier_uses_the_rule() -> None:
    # A non-excavation agent (no surrogate tier) is unaffected by a global error-budget policy.
    policy = FidelityPolicy(rule=FidelityRule.COARSEST, error_budget={"pos_x": 0.01})
    selection = select_fidelity("rover", LADDER, policy)  # LADDER has no SURROGATE tier
    assert selection.tier is FidelityTier.MASSMODEL  # the rule still applies
    assert selection.tolerance is None  # no budget verdict recorded


def test_a_pin_overrides_the_error_budget() -> None:
    policy = FidelityPolicy(
        error_budget={"pos_x": 0.0005}, agent_pins={"digger": FidelityTier.SURROGATE}
    )
    selection = select_fidelity(
        "digger", BUDGET_LADDER, policy, tier_budgets={FidelityTier.SURROGATE: SURROGATE_BUDGET}
    )
    assert selection.tier is FidelityTier.SURROGATE and selection.pinned is True


def test_error_budget_verdict_rides_in_the_provenance() -> None:
    policy = FidelityPolicy(error_budget={"pos_x": 0.01})
    selection = select_fidelity(
        "digger", BUDGET_LADDER, policy, tier_budgets={FidelityTier.SURROGATE: SURROGATE_BUDGET}
    )
    record = selection.as_provenance()
    assert record["within_budget"] is True
    assert record["tolerance"] == {"pos_x": 0.01}
    assert record["admitted_budget"] == SURROGATE_BUDGET
