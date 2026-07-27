"""The learned-surrogate granular fidelity tier (RM-P1-SIM-03; sim.md §5, §10, §11).

Acceptance: with a calibrated Surrogate tier available, the scheduler substitutes it only within
budget and falls back to the DEM ground truth when it exceeds tolerance; the tier arrives via ONNX
+ a Core contract (Sim imports only Core), verified fail-closed; drift/OOD trips re-validation and a
mid-episode escalation; error-budget outcomes round-trip to Parquet. The tier is exercised against a
**frozen bundle fixture** built offline by ``scripts/gen_surrogate_fixture.py`` — Sim never imports
``astro_mine.surrogate``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astro_mine.core.registry import PluginManifest, UnsignedManifest
from astro_mine.hub.supply_chain import SignatureError, generate_keypair, make_verifier
from astro_mine.sim.engines.dem._engine import DemGranularEngine, build_dem_engine
from astro_mine.sim.engines.surrogate import (
    AdaptiveGranularEngine,
    SurrogateIntegrityError,
    build_scheduled_granular_engine,
    build_surrogate_granular_engine,
    load_surrogate_tier,
)
from astro_mine.sim.recording.error_budget import (
    read_error_budget_report,
    write_error_budget_report,
)
from astro_mine.sim.runtime import AgentSpec, DemGranularDynamics, RngStreams, Scenario
from astro_mine.sim.scheduler import FidelityPolicy

_FIXTURE = Path(__file__).parent / "fixtures" / "surrogate"


def _manifest() -> PluginManifest:
    return PluginManifest.model_validate_json((_FIXTURE / "manifest.json").read_text())


def _bundle() -> bytes:
    return (_FIXTURE / "excavation_surrogate.onnxbundle").read_bytes()


def _verifier():
    return make_verifier(trusted_public_key_pem=(_FIXTURE / "signer_public_key.pem").read_bytes())


@pytest.fixture
def tier():
    return load_surrogate_tier(_bundle(), _manifest(), verifier=_verifier())


def _scenario(tool_speed_mps: float) -> Scenario:
    dynamics = DemGranularDynamics(
        n_particles=30,
        settle_substeps=120,
        regolith_density_kg_m3=1500.0,
        friction_coeff=0.55,
        restitution=0.3,
        tool_speed_mps=tool_speed_mps,
    )
    return Scenario(
        name="dig",
        horizon_steps=1,
        dt_s=0.05,
        agents=(AgentSpec(agent_id="digger", dynamics=dynamics),),
    )


# --- fail-closed load (the tier arrives via ONNX + a signed Core manifest) --------


def test_tier_loads_and_reads_the_admission_budget_from_the_manifest(tier) -> None:
    assert tier.name == "excavation-gns"
    assert set(tier.recommended_error_budget) == {"pos_x", "pos_z", "vel_x", "vel_z"}
    assert tier.input_config_order == ["density", "friction", "restitution", "tool_speed"]


def test_tampered_bundle_fails_closed() -> None:
    with pytest.raises(SurrogateIntegrityError):
        load_surrogate_tier(_bundle() + b"tamper", _manifest(), verifier=_verifier())


def test_untrusted_key_fails_closed() -> None:
    _, other_public = generate_keypair()
    with pytest.raises(SignatureError):
        load_surrogate_tier(
            _bundle(), _manifest(), verifier=make_verifier(trusted_public_key_pem=other_public)
        )


def test_unsigned_manifest_fails_closed() -> None:
    unsigned = _manifest().model_copy(update={"signature": None})
    with pytest.raises(UnsignedManifest):
        load_surrogate_tier(_bundle(), unsigned, verifier=_verifier())


# --- the runtime surrogate-error gate ---------------------------------------------


def test_in_domain_run_stays_on_the_surrogate_and_records_outcomes(tier) -> None:
    loose = {channel: value * 1e6 for channel, value in tier.recommended_error_budget.items()}
    engine = build_surrogate_granular_engine(
        _scenario(0.065), RngStreams(7), tier, tolerance=loose, revalidate_every=2
    )
    for _ in range(9):
        engine.advance(0.05)
    assert engine.modes()["digger"] == "surrogate"
    outcomes = engine.error_budget_outcomes
    assert len(outcomes) == 16  # reval every 2 steps over 9 -> 4 re-validations x 4 channels
    assert all(o.within_budget and o.tier == "surrogate" for o in outcomes)


def test_out_of_domain_query_escalates_to_the_dem_ground_truth(tier) -> None:
    # tool_speed 0.04 is below the surrogate's trust region [0.05, 0.08] -> OOD on the first query.
    engine = build_surrogate_granular_engine(
        _scenario(0.04), RngStreams(7), tier, tolerance=tier.recommended_error_budget
    )
    engine.advance(0.05)
    assert engine.modes()["digger"] == "dem"
    assert engine.error_budget_outcomes[0].within_budget is False
    # once escalated, subsequent ticks run on DEM ground truth (no new surrogate outcomes).
    engine.advance(0.05)
    assert len(engine.error_budget_outcomes) == 1


def test_digging_delegates_actions_draws_battery_and_retires(tier) -> None:
    from tests.sim.test_engines_dem import _excavate

    loose = {channel: value * 1e6 for channel, value in tier.recommended_error_budget.items()}
    engine = build_surrogate_granular_engine(
        _scenario(0.065), RngStreams(7), tier, tolerance=loose, revalidate_every=2
    )
    engine.apply_actions(_excavate(None))  # delegated to the wrapped DEM engine -> digging
    before = engine.export_coupling_state().samples[0].battery_soc_j
    for _ in range(8):
        engine.advance(0.05)
    after = engine.export_coupling_state().samples[0].battery_soc_j
    assert after <= before  # battery drawn on the surrogate path (reduced proxy) while digging
    assert engine.modes()["digger"] == "surrogate"
    engine.retire(["digger"])
    assert "digger" not in engine.modes()


def test_revalidation_breach_escalates_mid_episode(tier) -> None:
    tight = {channel: 1e-9 for channel in tier.recommended_error_budget}
    engine = build_surrogate_granular_engine(
        _scenario(0.065), RngStreams(7), tier, tolerance=tight, revalidate_every=1
    )
    engine.advance(0.05)
    assert engine.modes()["digger"] == "dem"  # DEM re-validation found deviation over tolerance


def test_two_runs_are_deterministic(tier) -> None:
    loose = {channel: value * 1e6 for channel, value in tier.recommended_error_budget.items()}

    def run() -> tuple[float, float, float]:
        engine = build_surrogate_granular_engine(
            _scenario(0.065), RngStreams(7), tier, tolerance=loose, revalidate_every=2
        )
        for _ in range(6):
            engine.advance(0.05)
        translation = engine.export_coupling_state().samples[0].pose.translation_m
        return (translation.x, translation.y, translation.z)

    assert run() == run()


# --- the scheduler admission drives the engine binding ----------------------------


def test_admission_binds_the_adaptive_engine_within_budget(tier) -> None:
    loose = {channel: value * 2 for channel, value in tier.recommended_error_budget.items()}
    engine = build_scheduled_granular_engine(
        _scenario(0.065), RngStreams(7), tier, policy=FidelityPolicy(error_budget=loose)
    )
    assert isinstance(engine, AdaptiveGranularEngine)


def test_admission_falls_back_to_dem_when_over_budget(tier) -> None:
    tight = {channel: value * 0.01 for channel, value in tier.recommended_error_budget.items()}
    engine = build_scheduled_granular_engine(
        _scenario(0.065), RngStreams(7), tier, policy=FidelityPolicy(error_budget=tight)
    )
    assert isinstance(engine, DemGranularEngine)


# --- the error-budget report round-trips to Parquet -------------------------------


def test_engine_outcomes_round_trip_through_parquet(tier, tmp_path) -> None:
    loose = {channel: value * 1e6 for channel, value in tier.recommended_error_budget.items()}
    engine = build_surrogate_granular_engine(
        _scenario(0.065), RngStreams(7), tier, tolerance=loose, revalidate_every=2
    )
    for _ in range(6):
        engine.advance(0.05)
    path = tmp_path / "error_budget.parquet"
    write_error_budget_report(engine.error_budget_outcomes, path)
    assert read_error_budget_report(path) == engine.error_budget_outcomes


def test_a_passing_revalidation_re_anchors_the_bed_to_dem(tier) -> None:
    """Option B (astro-mine-surrogate#23): a passing check keeps the DEM-advanced bed.

    A step surrogate feeds its own output back as input, so left alone it drifts without bound and
    the state it is graded on runs away from anything its budget covers. On a due re-validation the
    engine now advances the *real* bed by DEM and keeps that bed whether or not the check passes —
    capping the drift at `revalidate_every` steps. This asserts the bed after a passing check is the
    DEM bed, by comparing against a pure-DEM engine stepped the same way: they must agree, because a
    surrogate re-anchored every step is just DEM.
    """
    loose = {channel: value * 1e6 for channel, value in tier.recommended_error_budget.items()}
    adaptive = build_surrogate_granular_engine(
        _scenario(0.065), RngStreams(7), tier, tolerance=loose, revalidate_every=1
    )
    reference = build_dem_engine(_scenario(0.065), RngStreams(7))
    for _ in range(4):
        adaptive.advance(0.05)
        reference.advance(0.05)

    assert adaptive.modes()["digger"] == "surrogate"  # loose tolerance → never escalated
    a_pos, a_vel = adaptive._dem._states["digger"].bed.pos, adaptive._dem._states["digger"].bed.vel
    r_pos, r_vel = reference._states["digger"].bed.pos, reference._states["digger"].bed.vel
    # Re-anchored every step, the surrogate engine's bed IS the DEM bed — drift is fully bounded.
    assert np.allclose(a_pos, r_pos) and np.allclose(a_vel, r_vel)


def test_a_cadence_coarser_than_the_declared_horizon_is_refused(tier) -> None:
    """The 'or refuses' half of the contract (astro-mine-surrogate#23).

    The fixture tier declares a 2-step budget horizon. Grading it every 4 steps would check a bound
    the producer never made — a 2-step budget applied after 3 steps of drift — so the engine refuses
    to build rather than silently mismeasure.
    """
    assert tier.budget_horizon_steps == 2
    with pytest.raises(ValueError, match="exceeds the tier's declared budget_horizon_steps"):
        build_surrogate_granular_engine(
            _scenario(0.065),
            RngStreams(7),
            tier,
            tolerance=tier.recommended_error_budget,
            revalidate_every=4,
        )


def test_the_default_cadence_honours_the_declared_horizon(tier) -> None:
    """Unset, the cadence defaults to the tier's declared horizon — a caller need not know it."""
    engine = build_surrogate_granular_engine(
        _scenario(0.065), RngStreams(7), tier, tolerance=tier.recommended_error_budget
    )
    assert engine._revalidate_every == tier.budget_horizon_steps == 2
