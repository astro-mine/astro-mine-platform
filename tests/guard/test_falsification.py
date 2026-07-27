"""Adversarial falsification: zero hard-constraint violations under attack (RM-P1-GUARD-05).

The central validation gate for an assurance component (guard.md §10; issue #5): a seeded,
reproducible search over policy actions and disturbances tries to drive the anchor rover into a
keep-out region or to certify an unsafe action, and the shield must prevent every one — **zero
violations**. The deliberately unshielded control run proves the search is real (it finds
violations), and the two-runs-identical check is the determinism gate (conventions §11).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from astro_mine.guard.audit.sink import CollectingSink
from astro_mine.guard.falsify import (
    DEFAULT_DT,
    DEFAULT_U_MAX,
    AdversaryPolicy,
    SeededAdversary,
    WorstCaseAdversary,
    anchor_initial_state,
    control_rollout,
    control_violations,
    shielded_rollout,
    shielded_violations,
)
from astro_mine.guard.models import compile_anchor
from astro_mine.guard.spec.ir import CompiledSafetyModel
from astro_mine.guard.wrap import CoreConfig, PolicyShield

# A coarse sample period keeps the two ~14-day survival monitors' ring buffers tiny, so a per-agent
# SafetyCore builds instantly and the seeded rollouts run in well under a second.
COARSE_PERIOD_S = 120_960.0
HORIZON = 120


@pytest.fixture(scope="module")
def compiled() -> CompiledSafetyModel:
    return compile_anchor(sample_period_s=COARSE_PERIOD_S)


def _shield(
    compiled: CompiledSafetyModel, adversary: object
) -> tuple[PolicyShield, CollectingSink]:
    sink = CollectingSink()
    shield = PolicyShield(
        AdversaryPolicy(adversary, spatial_dim=3),  # type: ignore[arg-type]
        compiled,
        sink=sink,
        core_config=CoreConfig(),
    )
    return shield, sink


# --- zero violations under adversarial test --------------------------------------------------


@pytest.mark.parametrize("seed", range(16))
def test_seeded_adversary_finds_zero_violations(compiled: CompiledSafetyModel, seed: int) -> None:
    adversary = SeededAdversary(seed)
    shield, sink = _shield(compiled, adversary)
    steps = shielded_rollout(
        shield, adversary, initial=anchor_initial_state(), horizon=HORIZON, sink=sink
    )
    violations = shielded_violations(steps, compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
    assert violations == [], f"seed {seed}: {violations[:3]}"


def test_worst_case_adversary_finds_zero_violations(compiled: CompiledSafetyModel) -> None:
    # Full-thrust-toward-keep-out + monotonic energy/thermal drain: the shield must correct the
    # spatial approach AND fall back (never fail open) as the survival floors are crossed.
    adversary = WorstCaseAdversary(compiled)
    shield, sink = _shield(compiled, adversary)
    steps = shielded_rollout(
        shield, adversary, initial=anchor_initial_state(), horizon=200, sink=sink
    )
    assert shielded_violations(steps, compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT) == []
    # the drain genuinely forced verified fallbacks (otherwise the test would be vacuous)
    backups = [s for s in steps if s.verdict is not None and s.verdict.layer == "backup"]
    assert backups, "the resource-drain attack never triggered a backup — test is vacuous"


@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_property_based_no_violation_for_any_seed(compiled: CompiledSafetyModel, seed: int) -> None:
    # Property: for *any* seeded adversarial action/disturbance/signal sequence, the shield yields
    # zero hard-constraint violations (the Hypothesis analogue of the Rust core's proptest gate).
    adversary = SeededAdversary(seed)
    shield, sink = _shield(compiled, adversary)
    steps = shielded_rollout(
        shield, adversary, initial=anchor_initial_state(), horizon=60, sink=sink
    )
    assert shielded_violations(steps, compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT) == []


# --- the unshielded control proves the search is not vacuous ----------------------------------


def test_unshielded_control_finds_violations(compiled: CompiledSafetyModel) -> None:
    # Remove the shield: the raw worst-case proposals drive the rover into the lander-zone keep-out.
    adversary = WorstCaseAdversary(compiled)
    steps = control_rollout(adversary, spatial_dim=3, initial=anchor_initial_state(), horizon=200)
    violations = control_violations(steps, compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
    assert violations, "the unshielded search found no violations — the harness is vacuous"
    assert any(v.kind == "keep_out" for v in violations)


def test_shield_strictly_dominates_the_control(compiled: CompiledSafetyModel) -> None:
    # Same worst-case attack, shielded vs not: shielded is clean, control breaches. Directly
    # exhibits the shield preventing what the raw policy would have done.
    adversary = WorstCaseAdversary(compiled)
    shield, sink = _shield(compiled, adversary)
    shielded = shielded_rollout(
        shield, adversary, initial=anchor_initial_state(), horizon=150, sink=sink
    )
    control = control_rollout(
        WorstCaseAdversary(compiled), spatial_dim=3, initial=anchor_initial_state(), horizon=150
    )
    assert shielded_violations(shielded, compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT) == []
    assert control_violations(control, compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)


# --- seeded / reproducible (the determinism gate) ---------------------------------------------


def test_rollout_is_deterministic(compiled: CompiledSafetyModel) -> None:
    def run() -> list[tuple[float, ...]]:
        adversary = SeededAdversary(1234)
        shield, sink = _shield(compiled, adversary)
        steps = shielded_rollout(
            shield, adversary, initial=anchor_initial_state(), horizon=80, sink=sink
        )
        return [s.certified_action for s in steps]

    first, second = run(), run()
    assert first == second  # identical trajectories across two runs (byte-for-byte on floats)
    assert all(all(math.isfinite(x) for x in a) for a in first)
