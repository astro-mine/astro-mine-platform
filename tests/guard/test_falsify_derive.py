"""Deriving a falsification start and attack from an arbitrary spec (issue #35).

`falsify` could only ever falsify the anchor, so the authoring loop the guide teaches P2 —
``validate → compile → falsify → sign`` — stopped one step short of the step that is supposed to
justify trusting what you wrote. What was anchor-shaped was never the *plant* (a synthetic double
integrator) and never the *spatial* attack (already aimed at whatever keep-out geometry the model
carries) — it was the **state**: a hardcoded start position and a hardcoded dict of the anchor's six
signal keys, which any other spec would miss or `KeyError` on.

These tests pin the derivation against three shapes of spec: the anchor, a spec with no keep-out
geometry at all (what `astro-mine new safety` scaffolds), and adversarial/degenerate bounds.
"""

from __future__ import annotations

import pytest

from astro_mine.guard.falsify import (
    DEFAULT_DT,
    DEFAULT_U_MAX,
    SeededAdversary,
    WorstCaseAdversary,
    control_rollout,
    control_violations,
    keepout_barrier,
    scalar_violations,
)
from astro_mine.guard.falsify.derive import (
    FalsifyDeriveError,
    SignalEnvelope,
    initial_state,
    safe_position,
    safe_signals,
    signal_envelopes,
)
from astro_mine.guard.models import compile_anchor
from astro_mine.guard.spec.enums import PredicateOp
from astro_mine.guard.spec.ir import CompiledSafetyModel, PredicateAtom, ScalarBound

COARSE_PERIOD_S = 120_960.0


@pytest.fixture(scope="module")
def compiled() -> CompiledSafetyModel:
    return compile_anchor(sample_period_s=COARSE_PERIOD_S)


# --- envelopes -------------------------------------------------------------------------------


def test_envelopes_read_the_anchors_own_bounds(compiled: CompiledSafetyModel) -> None:
    envelopes = signal_envelopes(compiled)
    # Every signal in the predicate table gets an envelope, including monitor-only ones.
    assert set(envelopes) == set(compiled.predicate_table.signals)
    assert envelopes["battery_soc_j"] == SignalEnvelope("battery_soc_j", floor=180_000.0)
    assert envelopes["traverse_speed_mps"] == SignalEnvelope("traverse_speed_mps", ceiling=0.1)
    # chassis_temp_k is the two-sided case: a floor AND a ceiling.
    assert envelopes["chassis_temp_k"] == SignalEnvelope("chassis_temp_k", 120.0, 320.0)
    # charging_window_active is read only by a temporal monitor — no scalar bound, no envelope.
    assert envelopes["charging_window_active"] == SignalEnvelope("charging_window_active")


def test_the_tightest_bound_on_each_side_wins() -> None:
    """Two floors on one signal: satisfying the tightest satisfies the looser."""
    base = compile_anchor(sample_period_s=COARSE_PERIOD_S)
    table = base.predicate_table
    index = table.signals.index("battery_soc_j")
    atoms = [
        *table.atoms,
        PredicateAtom(op=PredicateOp.GE, signal_index=index, threshold=250_000.0),
    ]
    bounds = [
        *base.scalar_bounds,
        ScalarBound(
            constraint_id="c_extra_floor", on_uncertain="safe_state", atom_index=len(atoms) - 1
        ),
    ]
    widened = base.model_copy(
        update={
            "predicate_table": table.model_copy(update={"atoms": atoms}),
            "scalar_bounds": bounds,
        }
    )
    assert signal_envelopes(widened)["battery_soc_j"].floor == 250_000.0


def test_an_empty_interval_is_reported_as_unfalsifiable() -> None:
    """A floor above its own ceiling means no state satisfies the spec — say so, don't search."""
    base = compile_anchor(sample_period_s=COARSE_PERIOD_S)
    table = base.predicate_table
    index = table.signals.index("chassis_temp_k")
    atoms = [*table.atoms, PredicateAtom(op=PredicateOp.GE, signal_index=index, threshold=400.0)]
    bounds = [
        *base.scalar_bounds,
        ScalarBound(
            constraint_id="c_impossible", on_uncertain="safe_state", atom_index=len(atoms) - 1
        ),
    ]
    broken = base.model_copy(
        update={
            "predicate_table": table.model_copy(update={"atoms": atoms}),
            "scalar_bounds": bounds,
        }
    )
    with pytest.raises(FalsifyDeriveError, match="empty interval"):
        signal_envelopes(broken)


@pytest.mark.parametrize(
    ("envelope", "expected"),
    [
        (SignalEnvelope("s", 100.0, 300.0), 200.0),  # two-sided -> midpoint
        (SignalEnvelope("s", floor=200.0), 300.0),  # floor only -> floor + 50%
        (SignalEnvelope("s", ceiling=0.1), 0.05),  # ceiling only -> ceiling - 50%
        (SignalEnvelope("s", floor=0.0), 1.0),  # a zero bound has no fraction; fall back
        (SignalEnvelope("s"), 1.0),  # unbounded -> nominal
    ],
)
def test_safe_value_is_scale_free(envelope: SignalEnvelope, expected: float) -> None:
    """One rule has to serve joules and metres-per-second alike, so it is relative, not absolute."""
    assert envelope.safe_value() == pytest.approx(expected)


def test_toward_violation_follows_the_envelopes_own_direction() -> None:
    """The same call must drain a floor and inflate a ceiling — what makes it spec-generic."""
    floor = SignalEnvelope("s", floor=100.0)
    assert floor.toward_violation(150.0, fraction=0.1) < 150.0
    ceiling = SignalEnvelope("s", ceiling=10.0)
    assert ceiling.toward_violation(5.0, fraction=0.1) > 5.0
    # Unbounded: nothing to move toward.
    assert SignalEnvelope("s").toward_violation(3.0, fraction=1.0) == 3.0


# --- start states ----------------------------------------------------------------------------


def test_the_derived_anchor_start_is_inside_the_anchors_safe_set(
    compiled: CompiledSafetyModel,
) -> None:
    """The property that matters: a violation must be the adversary's doing, not the start's."""
    state = initial_state(compiled)
    for term in compiled.keep_out_terms:
        assert keepout_barrier(term, list(state.position)) > 0.0
    assert scalar_violations(compiled, dict(state.signals)) == []
    # ...and it covers every signal the plant has to supply, not a hand-listed subset.
    assert set(state.signals) == set(compiled.predicate_table.signals)


def test_the_derived_start_is_deterministic(compiled: CompiledSafetyModel) -> None:
    assert initial_state(compiled) == initial_state(compiled)


def test_a_spec_with_no_keepout_geometry_starts_at_the_origin(
    compiled: CompiledSafetyModel,
) -> None:
    """What `astro-mine new safety` scaffolds: scalar bounds only, no geometry to avoid."""
    flat = compiled.model_copy(update={"keep_out_terms": []})
    assert safe_position(flat) == (0.0, 0.0, 0.0)
    assert scalar_violations(flat, safe_signals(flat)) == []


def test_a_spec_whose_geometry_covers_everything_is_reported_not_crashed(
    compiled: CompiledSafetyModel,
) -> None:
    """A half-space keep-out with an enormous margin has no interior within probe range."""
    swallowed = compiled.model_copy(
        update={
            "keep_out_terms": [
                compiled.keep_out_terms[-1].model_copy(update={"margin_m": 1e12}),
            ]
        }
    )
    with pytest.raises(FalsifyDeriveError, match="no start position clears"):
        safe_position(swallowed)


def test_the_probe_respects_the_requested_clearance(compiled: CompiledSafetyModel) -> None:
    position = safe_position(compiled, clearance_m=500.0)
    assert min(keepout_barrier(t, list(position)) for t in compiled.keep_out_terms) >= 500.0


# --- the attacks, on a spec that is not the anchor --------------------------------------------


def test_the_worst_case_drain_breaches_a_spec_it_was_not_written_for(
    compiled: CompiledSafetyModel,
) -> None:
    """The control must bite on an arbitrary spec, or `shield held` there would prove nothing.

    Renaming every signal is the sharp version of "not the anchor": the old drain named
    `battery_soc_j` and `chassis_temp_k` outright, so it would `KeyError` here rather than attack.
    """
    renamed = _with_renamed_signals(compiled)
    state = initial_state(renamed)
    assert scalar_violations(renamed, dict(state.signals)) == []  # starts safe

    steps = control_rollout(WorstCaseAdversary(renamed), spatial_dim=3, initial=state, horizon=120)
    violations = control_violations(steps, renamed, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
    assert violations, "the unshielded control found nothing — the harness would be vacuous"


def test_the_seeded_walk_crosses_a_renamed_specs_bounds(compiled: CompiledSafetyModel) -> None:
    """A walk drawn strictly inside the envelope could never trip a bound, so it must overshoot."""
    renamed = _with_renamed_signals(compiled)
    adversary = SeededAdversary(7, compiled=renamed)
    signals = safe_signals(renamed)
    crossed = False
    for index in range(200):
        signals = adversary.next_signals(index, signals)
        if scalar_violations(renamed, signals):
            crossed = True
            break
    assert crossed, "the seeded walk never left the safe set, so the shield is never asked"


def test_the_seeded_walk_is_reproducible_from_the_seed(compiled: CompiledSafetyModel) -> None:
    start = safe_signals(compiled)
    first = SeededAdversary(11, compiled=compiled).next_signals(0, start)
    second = SeededAdversary(11, compiled=compiled).next_signals(0, start)
    assert first == second


def test_the_anchor_default_walk_is_unchanged_without_a_compiled_model() -> None:
    """Omitting `compiled=` keeps the anchor's historical walk bit-identical (backwards compat)."""
    from astro_mine.guard.falsify.adversary import ANCHOR_SAFE_SIGNALS

    legacy = SeededAdversary(3).next_signals(0, dict(ANCHOR_SAFE_SIGNALS))
    assert set(legacy) == set(ANCHOR_SAFE_SIGNALS)


def test_zero_drain_leaves_signals_untouched(compiled: CompiledSafetyModel) -> None:
    """`drain_fraction=0` is the spatial-attack-only mode the ONNX end-to-end test relies on."""
    start = safe_signals(compiled)
    assert WorstCaseAdversary(compiled, drain_fraction=0.0).next_signals(0, start) == start


def _with_renamed_signals(compiled: CompiledSafetyModel) -> CompiledSafetyModel:
    """The anchor with every signal key renamed — same bounds, none of the hardcoded names."""
    table = compiled.predicate_table
    return compiled.model_copy(
        update={
            "predicate_table": table.model_copy(
                update={"signals": [f"acme_{key}" for key in table.signals]}
            )
        }
    )
