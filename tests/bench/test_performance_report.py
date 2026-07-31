"""The performance report — the measured speedup, published beside the Scorecard (bench#31).

Two things are under test, and the second matters more than the first.

1. **The shape is the contract.** :class:`PerformanceReport` mirrors the JSON-able dicts Sim's
   ``FidelitySpeedupRunner.reports[seed].as_provenance()`` emits, and the fixtures below are copied
   **verbatim** from a real measurement's ``results.json`` (the anchor-world DEM-vs-surrogate run;
   see the zoo entry's ``RESULTS.md``). That is what pins the cross-repo wire contract *without
   importing Sim* — these tests run with no Sim, no onnxruntime, and no DEM solver in the
   environment, which is exactly the property that keeps Bench ``core`` + ``pydantic``.

2. **The Scorecard is untouched.** Bench's whole reproducibility guarantee rests on a scorecard hash
   that is a pure function of the run (bench.md §§69-72; ``LUNAR-TR-006``; ``LUNAR-DR-004``). A
   wall-clock number anywhere inside it would break ``assert_score_reproducible`` on *every* run. So
   there is an explicit regression test here that a scorecard is **bit-identical whether or not a
   performance report was produced alongside it** — the invariant that makes the sidecar design
   correct rather than merely convenient.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.bench.baseline import BaselinePolicy, reference_episode_runner, run
from astro_mine.bench.report import (
    PerformanceReport,
    SeedSpeedup,
    SurrogateIdentity,
    performance_report,
    toolchain_stamp,
)
from astro_mine.bench.scenario import ResolvedScenario
from astro_mine.bench.zoo import load_scenario
from astro_mine.core.scoring import EpisodeTrace

FIDELITY_ID = "lunar-polar-ice-excavation-fidelity-v1"

#: The excavator revision this task pins — 0.2.0, the first to declare a `tool` contact element
#: (astro-mine-fleet#37). Pinned by digest so a re-pin to a different asset fails loudly here.
EXCAVATOR_020 = "sha256:d576d7844625b25baec6496a45ad0a18a4da945c2cb0a5ebfa5c957d62fd5d35"


def _seed_row(
    seed: int,
    *,
    speedup: float | None = 3.0,
    admitted: bool = True,
    within_budget: bool = True,
    escalated: bool = False,
    holds_declared_budget: bool = True,
    is_claim: bool | None = None,
) -> dict[str, object]:
    """One seed's ``as_provenance()`` dict — the exact wire shape Sim emits."""
    return {
        "seed": seed,
        "admitted": admitted,
        "speedup": speedup,
        "within_budget": within_budget,
        "holds_declared_budget": holds_declared_budget,
        "escalated": escalated,
        "is_claim": (
            admitted and within_budget and not escalated if is_claim is None else is_claim
        ),
        "dem": {
            "engine": "dem_granular",
            "tier": "high",
            "steps": 200,
            "advance_wall_clock_s": 6.0,
            "sim_time_s": 10.0,
            "real_time_factor": 1.667,
            "mean_step_wall_clock_s": 0.03,
        },
        "surrogate": {
            "engine": "surrogate_granular",
            "tier": "surrogate",
            "steps": 200,
            "advance_wall_clock_s": 2.0,
            "sim_time_s": 10.0,
            "real_time_factor": 5.0,
            "mean_step_wall_clock_s": 0.01,
        },
        "declared_error_budget": {"pos_x": 0.00237, "vel_x": 0.0615},
        "admitted_tolerance": {"pos_x": 0.00237, "vel_x": 0.0615},
        "realized_error": {"pos_x": 0.0011, "vel_x": 0.031},
    }


def _report(rows: dict[int, dict[str, object]]) -> PerformanceReport:
    return performance_report(
        rows,
        scenario_id=FIDELITY_ID,
        spec_hash=load_scenario(FIDELITY_ID).spec_hash,
        runner="astro-mine-sim/fidelity-speedup",
        surrogate=SurrogateIdentity(name="excavation-gns", version="0.2.0"),
        toolchain=toolchain_stamp(),
    )


def test_report_round_trips_through_json() -> None:
    report = _report({3001: _seed_row(3001), 3002: _seed_row(3002)})
    restored = PerformanceReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.headline_speedup == report.headline_speedup


def test_producer_wire_shape_is_pinned() -> None:
    """A drifted producer key must fail *here*, at the boundary — not vanish silently.

    ``extra="forbid"`` is the whole point: if Sim renames a field in ``as_provenance()``, the field
    that carried the claim would otherwise be dropped on the floor and the report would still
    validate, still publish, and still be wrong.
    """
    drifted = _seed_row(3001) | {"speed_up": 3.0}
    with pytest.raises(ValidationError):
        _report({3001: drifted})


def test_headline_is_the_median_over_claiming_seeds_only() -> None:
    """Non-claiming seeds are excluded from the headline — not averaged into it.

    A seed that escalated ran partly on the *reference* solver, so its ratio is partly
    DEM-vs-DEM. It
    is not a slightly-worse data point; it is not a data point.
    """
    rows = {
        3001: _seed_row(3001, speedup=2.0),
        3002: _seed_row(3002, speedup=4.0),
        3003: _seed_row(3003, speedup=6.0),
        # Escalated mid-run: ratio is contaminated, and must not drag the median.
        3004: _seed_row(3004, speedup=1.05, escalated=True, within_budget=False),
        # Never admitted: this seed ran DEM twice and has no speedup to report at all.
        3005: _seed_row(3005, speedup=1.01, admitted=False),
    }
    report = _report(rows)
    assert {s.seed for s in report.claiming_seeds} == {3001, 3002, 3003}
    assert report.headline_speedup == 4.0  # median of (2, 4, 6), NOT of (1.01, 1.05, 2, 4, 6)
    assert report.is_claim


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"admitted": False}, "the tier was never admitted: the run was DEM twice over"),
        ({"escalated": True}, "the tier fell back to the reference solver mid-run"),
        ({"within_budget": False}, "the tier breached the tolerance it was admitted under"),
        ({"speedup": None}, "a tier's wall-clock was unmeasurable"),
    ],
)
def test_a_disqualified_run_yields_no_claim(kwargs: dict[str, object], why: str) -> None:
    """Speed alone is not a result (``LUNAR-TR-002``). Every one of these publishes nothing."""
    report = _report({3001: _seed_row(3001, **kwargs)})  # type: ignore[arg-type]
    assert report.claiming_seeds == ()
    assert report.headline_speedup is None, why
    assert not report.is_claim


def test_worst_realized_error_travels_with_the_headline() -> None:
    """The error half of the claim: a speedup without its realized error is not a result."""
    rows = {
        3001: _seed_row(3001) | {"realized_error": {"pos_x": 0.0011, "vel_x": 0.031}},
        3002: _seed_row(3002) | {"realized_error": {"pos_x": 0.0019, "vel_x": 0.028}},
        # Escalated — its (larger) error must not enter the claim's worst case either.
        3003: _seed_row(3003, escalated=True, within_budget=False)
        | {"realized_error": {"pos_x": 0.9, "vel_x": 0.9}},
    }
    report = _report(rows)
    assert report.worst_realized_error == {"pos_x": 0.0019, "vel_x": 0.031}


class _MeasuringRunner:
    """A stand-in for Sim's ``FidelitySpeedupRunner``: scores a trace *and* accrues speedup rows.

    It reproduces the only two structural facts Bench relies on — it satisfies the ``EpisodeRunner``
    protocol (``__call__(resolved, policy, seed) -> EpisodeTrace``), and the speedup rides out as
    **runner state** (``.reports``) rather than through the trace. That is the seam, and this is it
    exercised with no Sim in the environment.
    """

    __name__ = "astro-mine-sim/fidelity-speedup"

    def __init__(self) -> None:
        self.reports: dict[int, dict[str, object]] = {}

    def __call__(self, resolved: ResolvedScenario, policy: object, seed: int) -> EpisodeTrace:
        self.reports[seed] = _seed_row(seed)
        return reference_episode_runner(resolved, policy, seed)  # type: ignore[arg-type]


def test_scorecard_hash_is_identical_whether_or_not_a_report_was_produced() -> None:
    """**The load-bearing regression.** Measuring a run must not change the run's score.

    This is what makes the sidecar design correct rather than merely convenient. If a duration — or
    anything derived from one — ever leaked into the Scorecard, ``content_hash`` would cover it,
    ``assert_score_reproducible`` would fail unconditionally, and Bench would have traded its
    reproducibility guarantee (bench.md §207) for a performance number.

    So: score the same scenario twice, once through a plain runner and once through a runner that
    *also* emits a full performance report, and demand the two scores be identical in everything but
    the runner identity — i.e. the wall-clock measurement never reaches the scorecard. (The runner
    id itself *is* part of the scorecard now, by design — G1.8 — but that is legible provenance, not
    a leaked duration; the two runners here are deliberately distinct.)
    """
    spec = load_scenario(FIDELITY_ID)
    policy = BaselinePolicy()

    plain = run(spec, policy, runner=reference_episode_runner)

    measuring = _MeasuringRunner()
    measured = run(spec, policy, runner=measuring)
    report = _report(measuring.reports)

    # The report exists, is a real claim, and carries a wall-clock number...
    assert report.is_claim
    assert report.headline_speedup == 3.0
    assert report.seeds[0].dem.advance_wall_clock_s > 0.0

    # ...that does not leak into the score. The per-metric results are bit-identical, and the *only*
    # scorecard field that differs is the runner identity — legitimate provenance folded into the
    # hash (G1.8), never anything derived from the wall-clock measurement.
    assert measured.metrics == plain.metrics
    measured_dump = measured.model_dump(mode="json")
    plain_dump = plain.model_dump(mode="json")
    assert {k: v for k, v in measured_dump.items() if k != "runner"} == {
        k: v for k, v in plain_dump.items() if k != "runner"
    }
    assert measured.runner != plain.runner  # the fixture vs the fidelity-speedup runner


def test_the_fidelity_scenario_is_the_task_it_claims_to_be() -> None:
    """The scenario's cadence and its excavator are both load-bearing, so both are pinned.

    - The tick (``max_sim_seconds / horizon_steps``) must be **0.05 s**: the contact-scale tick the
      granular tiers integrate at (Sim's ``_CONTACT_DT_S``). The task *declares* the cadence it is
      scored at rather than letting the runner silently override it.
    - The excavator must be **0.2.0** — the first revision with a `tool` contact element. Pinned at
      0.1.0 the scenario would resolve to a *wheeled rover*, reach no granular tier, and measure
      nothing (astro-mine-fleet#37).
    """
    spec = load_scenario(FIDELITY_ID)
    assert spec.episode.max_sim_seconds is not None
    assert spec.episode.max_sim_seconds / spec.episode.horizon_steps == pytest.approx(0.05)

    excavators = [ref for ref in spec.content.fleet if ref.id == "astro-mine.fleet.excavator"]
    assert [ref.content_hash for ref in excavators] == [EXCAVATOR_020]
    # The comparison is about excavation physics: a fleet carrying agents the surrogate has nothing
    # to do with would only dilute the ratio toward 1.0.
    assert len(spec.content.fleet) == 1


def test_seed_speedup_defaults_are_not_silently_claiming() -> None:
    """An absent ``speedup`` must not be readable as a claim."""
    row = SeedSpeedup.model_validate(
        {
            "seed": 1,
            "admitted": True,
            "within_budget": True,
            "holds_declared_budget": True,
            "escalated": False,
            "is_claim": True,
            "dem": {
                "engine": "d",
                "tier": "high",
                "steps": 1,
                "advance_wall_clock_s": 1.0,
                "sim_time_s": 1.0,
            },
            "surrogate": {
                "engine": "s",
                "tier": "surrogate",
                "steps": 1,
                "advance_wall_clock_s": 1.0,
                "sim_time_s": 1.0,
            },
        }
    )
    assert row.speedup is None
    # `is_claim` is the producer's verdict, but a claim with no ratio contributes no headline.
    report = _report({1: row.model_dump(mode="json")})
    assert report.headline_speedup is None
    assert not report.is_claim
