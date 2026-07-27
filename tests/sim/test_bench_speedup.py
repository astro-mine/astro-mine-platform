"""The surrogate speedup benchmark — DEM vs. surrogate, same seed, same task (#51).

Surrogate's phase exit criterion is a *demonstrated speedup at a published, calibrated error bound*
on a Bench scenario (surrogate.md §8, §12; ``LUNAR-TR-002``). astro-mine-bench#31 is the Bench half
and cannot be built until Sim can produce the number. This is the Sim half.

Two gaps had to close for it to be possible at all, and both are asserted here:

1. Sim had **no wall-clock anywhere** — every ``elapsed_s`` in the codebase is *sim* time
   (see ``test_timing.py``).
2. The Bench adapter never selected a fidelity tier: a TOOL-bearing asset could only ever route to
   the reduced-order ``granular`` kind, and the ``Scenario`` was always built with the default
   ``FidelityPolicy``. So the DEM/surrogate ladder was **unreachable from a Bench spec** — there was
   no path from a Bench ``ScenarioSpec`` to a ``FidelityPolicy.error_budget``.

The surrogate tier is exercised against the same **frozen bundle fixture** the tier's own suite
uses (built offline by ``scripts/gen_surrogate_fixture.py``) — Sim never imports
``astro_mine.surrogate``.

The episode is short and its tick is *contact*-scale, not the Bench spec's 60 s mission cadence: a
DEM bed sub-steps ``dt_s / dt_internal_s`` times per step, so a 60 s tick means ~78 000 sub-steps
per step per agent. That is the reason ``dt_s`` is a dial on the adapter at all.

**No speed threshold is asserted.** A CI runner's absolute throughput is not a stable assertion,
and a test that fails when a shared runner is busy is noise, not a gate. What is asserted is the
*contract*: that both tiers run the same seeded task, that the ratio and the realized-vs-declared
error are reported, and that a speedup is only ever claimed inside a bound that actually held.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from astro_mine.bench.baseline import BaselinePolicy
from astro_mine.bench.baseline._runner import EpisodeRunner
from astro_mine.bench.scenario import ContentRef, ScenarioSpec, resolve_scenario
from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.core.sadf.enums import FidelityTier
from astro_mine.core.units import Epoch
from astro_mine.core.world import RegolithParams, SurfacePoint
from astro_mine.hub.supply_chain import make_verifier
from astro_mine.sim.bench import FidelitySpeedupRunner, SimEpisodeRunner, dynamics_for_asset
from astro_mine.sim.engines.surrogate import load_surrogate_tier
from astro_mine.sim.runtime.scenario import DemGranularDynamics, GranularDynamics
from astro_mine.sim.scheduler import FidelityPolicy
from tests.sim.test_bench_runner import _AnchorWorld, _excavator_asset, _spec, publish_content

_FIXTURE = Path(__file__).parent / "fixtures" / "surrogate"


def _spec_with_excavator(content: dict[str, Any]) -> ScenarioSpec:
    """The fixture spec, now also pinning the excavator — the only asset that reaches the granular
    ladder, and therefore the only one a DEM-vs-surrogate comparison has anything to say about."""
    spec = _spec(content)
    pins = spec.content
    return spec.model_copy(
        update={
            "content": pins.model_copy(
                update={
                    "fleet": (
                        *pins.fleet,
                        ContentRef(
                            id="astro-mine.fleet.excavator",
                            content_hash=content["excavator_digest"],
                        ),
                    )
                }
            )
        }
    )


class _InDomainWorld(_AnchorWorld):
    """The pinned world, with regolith inside the surrogate's declared trust region.

    A surrogate is only valid on the domain it was **trained** on, and the fixture tier declares its
    band in its own manifest: density [1400, 1600], friction [0.4, 0.7], restitution 0.3.

    The runner suite's world sits a hair outside it — ``friction_angle_deg=35`` means
    ``tan(35°) = 0.70021``, which is 0.0002 *above* the friction bound. The tier notices (its ONNX
    graph carries the trust-region gate), reports ``in_domain=False``, and the engine escalates to
    the DEM reference on the first query. That is the tier behaving *correctly* — and it is why a
    speedup benchmark has to be run on terrain the surrogate was actually trained for, or it
    measures the reference solver against itself and calls the ~1.0 ratio a result."""

    def sample(
        self, position: tuple[float, float, float], *, epoch: Epoch | None = None
    ) -> SurfacePoint:
        point = super().sample(position, epoch=epoch)
        return replace(
            point,
            regolith=RegolithParams(
                bulk_density_kg_m3=1500.0,  # mid-band
                friction_angle_deg=28.0,  # tan(28°) = 0.532, comfortably inside [0.4, 0.7]
                bearing_capacity_pa=5.0e4,
            ),
        )


@pytest.fixture
def content(tmp_path: Path) -> dict[str, Any]:
    """The published content the runner suite uses, with the world swapped for an in-domain one."""
    published = publish_content(tmp_path)
    published["factories"] = {
        **published["factories"],
        PluginKind.WORLD_PROVIDER.value: lambda manifest, layers: _InDomainWorld(),
    }
    return published


@pytest.fixture
def tier():
    """The frozen, signed surrogate bundle — the same fixture the tier's own suite loads."""
    return load_surrogate_tier(
        (_FIXTURE / "excavation_surrogate.onnxbundle").read_bytes(),
        PluginManifest.model_validate_json((_FIXTURE / "manifest.json").read_text()),
        verifier=make_verifier(
            trusted_public_key_pem=(_FIXTURE / "signer_public_key.pem").read_bytes()
        ),
    )


#: A *task* tolerance, stated in physical terms rather than as a multiple of what the tier claims.
#: LUNAR-TR-002 makes this the operative bound: the scheduler admits a tier when its declared budget
#: fits inside the task's tolerance, and the engine holds it to the task's number.
#:
#: It must sit **above** the tier's declared budget for the tier to be admitted at all. That budget
#: is now the honest *rollout* budget (astro-mine-surrogate#23) — the worst deviation over the
#: horizon Sim re-anchors at, not a single step — so for this deliberately small CI fixture model it
#: is loose (~12 cm bed, ~1.8 m/s). A task that tolerates 20 cm of bed deviation and 2.5 m/s admits
#: it; a real, well-trained tier would declare a far tighter budget and admit under a far tighter
#: task. The ``impossible`` and OOD tests below drive the reject / escalate paths on purpose.
_TASK_TOLERANCE = {"pos_x": 0.20, "pos_z": 0.20, "vel_x": 2.5, "vel_z": 2.5}

#: A short horizon: a DEM bed is CPU-bound, and the *contract* under test (both tiers run, both are
#: timed, the bound is checked) needs a handful of ticks, not a benchmark-length episode. The
#: production speedup number is run at the scenario's real horizon; this is the gate, not the claim.
_HORIZON = 4


def _speedup_runner(
    content: dict[str, Any], tmp_path: Path, tier: Any, **kwargs: Any
) -> FidelitySpeedupRunner:
    return FidelitySpeedupRunner(
        store=content["store"],
        provider_factories=content["factories"],
        surrogate=tier,
        recording_dir=tmp_path / "mcap",
        horizon_steps=_HORIZON,
        **kwargs,
    )


# --- the fidelity plumbing (a Bench run can now reach either tier) ----------------


def test_an_excavator_reaches_the_dem_tier_only_when_it_is_selected(
    content: dict[str, Any],
) -> None:
    """The routing gap: a TOOL-bearing asset used to have exactly one destination."""
    excavator = _excavator_asset(content)

    reduced = dynamics_for_asset(excavator)
    dem = dynamics_for_asset(excavator, dem_tier=True)

    assert isinstance(reduced, GranularDynamics)  # the old, only, destination
    assert isinstance(dem, DemGranularDynamics)  # the high-fidelity reference tier


def test_a_bench_driven_run_can_carry_a_fidelity_policy(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The other half of the gap: the Scenario was always built with the default FidelityPolicy, so
    a Bench-driven run could never leave the coarsest tier however the surrogate was configured."""
    budget = {"pos_x": 0.05, "pos_z": 0.05, "vel_x": 0.1, "vel_z": 0.1}
    runner = SimEpisodeRunner(
        store=content["store"],
        provider_factories=content["factories"],
        recording_dir=tmp_path / "mcap",
        horizon_steps=_HORIZON,
        dem_tier=True,
        fidelity=FidelityPolicy(error_budget=budget),
    )

    run = runner.resolve(resolve_scenario(_spec_with_excavator(content)), seed=1001)

    assert run.scenario.fidelity.error_budget == budget
    assert any(isinstance(a.dynamics, DemGranularDynamics) for a in run.scenario.agents)


# --- the deliverable --------------------------------------------------------------


def test_the_same_seeded_task_runs_at_both_tiers_and_reports_the_ratio(
    content: dict[str, Any], tmp_path: Path, tier: Any
) -> None:
    """The number surrogate.md §8 calls 'the deliverable that proves the package'."""
    runner = _speedup_runner(content, tmp_path, tier, tolerance=_TASK_TOLERANCE)

    report, _ = runner.measure(
        resolve_scenario(_spec_with_excavator(content)), BaselinePolicy(), seed=1001
    )

    # Both tiers really ran, on the same seeded task, and each says which tier it was.
    assert report.dem.tier == FidelityTier.ARTICULATED.value
    assert report.surrogate.tier == FidelityTier.SURROGATE.value
    assert report.admitted, "the scheduler did not admit the surrogate tier"
    assert report.dem.steps == report.surrogate.steps > 0
    assert report.dem.sim_time_s == pytest.approx(report.surrogate.sim_time_s)

    # ... and the ratio exists. Its *magnitude* is deliberately not asserted: a CI runner's absolute
    # speed is not a stable gate. That the two tiers are separately measurable is the contract.
    assert report.speedup is not None and report.speedup > 0.0
    assert report.dem.advance_wall_clock_s > 0.0
    assert report.surrogate.advance_wall_clock_s > 0.0


def test_the_realized_error_is_measured_against_the_bound_the_tier_ran_under(
    content: dict[str, Any], tmp_path: Path, tier: Any
) -> None:
    """The error half of the deliverable: the tier is re-validated against a live DEM reference bed,
    and the report carries the *declared* budget, the tolerance it actually ran under, and the
    realized deviation — not just a ratio."""
    runner = _speedup_runner(content, tmp_path, tier, tolerance=_TASK_TOLERANCE)

    report, _ = runner.measure(
        resolve_scenario(_spec_with_excavator(content)), BaselinePolicy(), seed=1001
    )

    # The declared budget is read off the artifact's own Core manifest; the admitted tolerance is
    # the task's. LUNAR-TR-002 makes the *task's* the operative one.
    assert report.declared_error_budget == tier.recommended_error_budget
    assert report.admitted_tolerance == _TASK_TOLERANCE
    assert report.realized_error, "the surrogate tier never re-validated against its DEM reference"
    assert set(report.realized_error) <= set(tier.recommended_error_budget)

    # Substitution held for the whole episode, so this run really is a speedup claim.
    assert report.within_budget and not report.escalated
    assert report.is_claim
    assert report.speedup is not None


def test_the_re_anchored_tier_holds_its_own_published_bound(
    content: dict[str, Any], tmp_path: Path, tier: Any
) -> None:
    """The tier holds the budget its manifest advertises — because the two are now made consistent.

    This test used to assert the opposite: the tier *overshot* its published budget mid-episode.
    That was a real defect, and fixing it is what astro-mine-surrogate#23 is. The budget is now
    calibrated over the same rollout horizon Sim grades at, and Sim re-anchors the bed to DEM every
    ``budget_horizon_steps`` so it never drifts past that horizon. So the realized deviation is
    bounded by the declared budget by construction: a well-made tier holds its own bound, and
    in-flight escalation is left to the trust-region (OOD) path, not a budget it was always going to
    miss. The verdict is still reported per channel, kept apart from the task-tolerance verdict."""
    runner = _speedup_runner(content, tmp_path, tier, tolerance=_TASK_TOLERANCE)

    report, _ = runner.measure(
        resolve_scenario(_spec_with_excavator(content)), BaselinePolicy(), seed=1001
    )

    assert report.within_budget  # it met the task's tolerance ...
    assert report.holds_declared_budget  # ... and the (now honest) budget it published, too
    over = {
        c: (v, report.declared_error_budget[c])
        for c, v in report.realized_error.items()
        if v > report.declared_error_budget[c]
    }
    assert not over, f"re-anchored, no channel should exceed its declared budget; got {over}"


def test_an_out_of_domain_excursion_escalates_and_forfeits_the_claim(
    content: dict[str, Any], tmp_path: Path, tier: Any
) -> None:
    """The runtime fail-safe (``LUNAR-TR-002``), now via the path a well-calibrated tier can still
    take: an out-of-trust-region query.

    A budget breach in flight no longer happens for a correctly-made tier — its realized error is
    bounded by its declared budget, which is bounded by the admitting task tolerance
    (astro-mine-surrogate#23). What *can* still force a mid-episode fallback is a query outside the
    trust region. Driving the blade below the tier's ``tool_speed`` band takes it out of domain on
    the first query; the engine escalates to DEM ground truth and stays there. The wall-clock ratio
    still exists — but it is not a *claim*, and the report says so rather than letting a number that
    was partly DEM-vs-DEM pass as a surrogate speedup."""
    # tool_speed 0.04 is below the tier's trust region [0.05, 0.08] — OOD on the first query.
    runner = _speedup_runner(
        content, tmp_path, tier, tolerance=_TASK_TOLERANCE, tool_speed_mps=0.04
    )

    report, _ = runner.measure(
        resolve_scenario(_spec_with_excavator(content)), BaselinePolicy(), seed=1001
    )

    assert report.escalated
    assert not report.is_claim
    assert any(not o.within_budget for o in report.outcomes)


def test_a_surrogate_that_cannot_meet_the_tolerance_is_not_substituted(
    content: dict[str, Any], tmp_path: Path, tier: Any
) -> None:
    """The honest-failure path: with a tolerance far tighter than the tier declares, the scheduler
    refuses it and the engine falls back to DEM. There is then no speedup to report — and the report
    says so rather than quietly comparing DEM against DEM and calling the ~1.0 ratio a result."""
    impossible = {channel: 1e-12 for channel in tier.recommended_error_budget}
    runner = _speedup_runner(content, tmp_path, tier, tolerance=impossible)

    report, _ = runner.measure(
        resolve_scenario(_spec_with_excavator(content)), BaselinePolicy(), seed=1001
    )

    assert not report.admitted
    assert report.surrogate.tier == FidelityTier.ARTICULATED.value  # it fell back to DEM
    assert not report.realized_error  # nothing was substituted, so nothing was measured against


# --- it hands the result to Bench through the runner seam --------------------------


def test_it_satisfies_benchs_episode_runner_protocol(
    content: dict[str, Any], tmp_path: Path, tier: Any
) -> None:
    """Bench never imports Sim (bench.md §2.2), and its runner contract forbids wall-clock *inside*
    a run — so the two-tier execution and the timing happen behind the injected seam, and the number
    rides out as runner state, exactly as ``recordings`` does."""
    runner: EpisodeRunner = _speedup_runner(content, tmp_path, tier)
    assert callable(runner)

    trace = runner(resolve_scenario(_spec_with_excavator(content)), BaselinePolicy(), seed=1001)

    # `__call__` returns the *surrogate*-tier trace — the tier under test, the physics a scored run
    # would actually use. The DEM run is the reference it was measured against, not the result.
    assert trace.observations
    assert runner.reports[1001].surrogate.tier == FidelityTier.SURROGATE.value
    assert runner.reports[1001].speedup is not None
    assert 1001 in runner.recordings
