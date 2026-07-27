"""The surrogate speedup measurement — DEM vs. surrogate, same seed, same task (#51).

Surrogate's phase exit criterion is a *demonstrated speedup at a published, calibrated error bound*
on a Bench scenario: the 10^2-10^4x wall-clock number surrogate.md §8 calls "the deliverable that
proves the package". This module produces it.

**Why it lives here and not in Bench.** Bench's contracts forbid Bench from doing it, normatively: a
conforming ``EpisodeRunner`` "MUST be a pure, deterministic function of ``(scenario_hash, policy,
seed)`` — no wall-clock" (``bench/baseline/_runner.py``), and a metric "MUST be a pure,
deterministic function of the trace". A speedup is neither: a ratio of two *durations* over runs at
different fidelities. And Bench never imports Sim (bench.md §2.2). So the two-tier execution and the
timing happen behind the injected runner seam — here — and the number rides out to Bench as runner
state, the same way :attr:`~astro_mine.sim.bench.SimEpisodeRunner.recordings` does.

**What it measures.** The same seeded episode is run twice on the same resolved content:

* the **DEM tier** — the high-fidelity soft-sphere granular solver, the reference physics;
* the **surrogate tier** — the learned tier, admitted only if its declared
  ``recommended_error_budget`` (carried in its Core manifest) satisfies the task tolerance, per the
  RM-P1-SIM-03 scheduler. If it is not admissible the engine falls back to DEM and the run is
  honest about it: :attr:`SpeedupReport.admitted` is ``False`` and there is no speedup to claim.

The two wall-clocks give the ratio; the surrogate engine's own re-validation against a DEM reference
bed gives the **realized** error, which is compared against the budget it was **declared** under.
Speed without that bound is not a result — ``LUNAR-TR-002`` requires Sim to refuse substitution
beyond task tolerance, and a speedup outside the budget is exactly the claim that requirement exists
to stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from astro_mine.core.sadf.enums import FidelityTier
from astro_mine.sim.bench._runner import SimEpisodeRunner
from astro_mine.sim.bench._scenario import ResolvedRun
from astro_mine.sim.engines.dem._engine import build_dem_engine
from astro_mine.sim.engines.surrogate._engine import build_scheduled_granular_engine
from astro_mine.sim.scheduler import FidelityPolicy

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from astro_mine.bench.metrics import EpisodeTrace, ScoringContext
    from astro_mine.bench.scenario import ResolvedScenario
    from astro_mine.core.policy import Policy
    from astro_mine.core.provenance.model import ErrorBudgetOutcome
    from astro_mine.sim.engines import RegimeEngine
    from astro_mine.sim.engines.surrogate._loader import LoadedSurrogate
    from astro_mine.sim.runtime.content import BundleStore, ProviderFactory
    from astro_mine.sim.runtime.episode import Trace
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import Scenario
    from astro_mine.sim.runtime.timing import EngineTiming

__all__ = ["FidelitySpeedupRunner", "SpeedupReport"]

#: The default tick a granular comparison runs at (s). A DEM bed's stable internal timestep is
#: ~0.8 ms, so it sub-steps ``dt_s / dt_internal_s`` times per tick; a Bench spec's *mission*
#: cadence (the anchor's is 60 s) would mean ~78 000 sub-steps per step per agent. Excavation is a
#: contact-scale process and is benchmarked as one — the DEM tier's own suite uses the same 0.05 s.
_CONTACT_DT_S = 0.05

#: The tool-speed channel of a granular surrogate's trust region, as its Core manifest declares it.
_TOOL_SPEED_CHANNEL = "tool_speed"


def _trust_region_midpoint(surrogate: LoadedSurrogate) -> float | None:
    """The centre of the surrogate's declared tool-speed trust region, or ``None`` if it declares
    none.

    A surrogate is only valid inside the domain it was trained on, and it is the artifact that says
    where that is (its manifest's ``trust_region``). Benchmarking it at a speed outside that band
    measures nothing: the first query is out-of-domain, the engine escalates to DEM, and the
    "speedup" is a DEM-vs-DEM ratio of ~1. So the default is read off the tier rather than
    hard-coded here — and a tier that declares no band gets no opinion from us."""
    bounds = surrogate.trust_region.get(_TOOL_SPEED_CHANNEL) if surrogate.trust_region else None
    if not bounds:
        return None
    low, high = bounds.get("low"), bounds.get("high")
    if low is None or high is None:
        return None
    return (float(low) + float(high)) / 2.0


@dataclass(frozen=True, slots=True)
class SpeedupReport:
    """One seed's DEM-vs-surrogate result: what it cost, and whether it stayed honest.

    :attr:`speedup` is the headline (DEM wall-clock / surrogate wall-clock), but it is only a
    *claim* while :attr:`within_budget` holds — see :attr:`realized_error` vs.
    :attr:`declared_error_budget`. A report whose surrogate was never admitted
    (:attr:`admitted` ``False``) ran DEM twice and has no speedup to report."""

    seed: int
    admitted: bool
    dem: EngineTiming
    surrogate: EngineTiming
    #: The budget the surrogate tier *declares* it holds — read from its Core manifest
    #: (``LoadedSurrogate.recommended_error_budget``), not from anything this run measured.
    declared_error_budget: Mapping[str, float]
    #: The **task** tolerance the tier was actually admitted and re-validated under. This is the
    #: operative bound: ``LUNAR-TR-002`` says Sim must refuse substitution beyond *task* tolerance,
    #: and the manifest's budget is the surrogate's recommendation, not the task's requirement. A
    #: task may tolerate more than the surrogate advertises (and then substitution is legitimate),
    #: or less (and then the tier is never admitted at all).
    admitted_tolerance: Mapping[str, float]
    #: The worst per-channel deviation the surrogate engine actually realized against its DEM
    #: reference bed, over every re-validation in the run. Empty if the tier never re-validated.
    realized_error: Mapping[str, float]
    #: The engine's per-re-validation verdicts, as persisted to the Parquet error-budget report.
    outcomes: tuple[ErrorBudgetOutcome, ...]

    @property
    def speedup(self) -> float | None:
        """DEM wall-clock / surrogate wall-clock, or ``None`` if either was unmeasurable."""
        dem_s = self.dem.advance_wall_clock_s
        surrogate_s = self.surrogate.advance_wall_clock_s
        if dem_s <= 0.0 or surrogate_s <= 0.0:
            return None
        return dem_s / surrogate_s

    @property
    def escalated(self) -> bool:
        """Whether the surrogate fell back to the DEM reference mid-run.

        It escalates on either failure: a query outside its trust region (``in_domain``), or a
        re-validated deviation past the task tolerance. Both are recorded as a failed outcome, and
        both mean the same thing here — part of this run's wall-clock was spent on the *reference*
        solver, so the ratio understates the surrogate and the substitution did not hold for the
        whole episode. A material fact about the measurement, not a footnote."""
        return any(not o.within_budget for o in self.outcomes)

    @property
    def within_budget(self) -> bool:
        """Whether every re-validated deviation stayed inside the tolerance it ran under.

        This is the **engine's own verdict**, not a recomputation: the engine compared each channel
        against the admitted tolerance at the moment it re-validated, and a second opinion derived
        from summary statistics could only disagree with it by being wrong."""
        return all(o.within_budget for o in self.outcomes if o.metric != _OOD_METRIC)

    @property
    def holds_declared_budget(self) -> bool:
        """Whether the realized deviation also stayed inside the budget the surrogate *advertised*.

        Distinct from :attr:`within_budget`: a tier can satisfy a lenient task while overshooting
        its own published bound. That is a fact about the *artifact* (its ``ErrorReport`` oversells
        it), not about this run, and it is worth surfacing separately rather than averaging away."""
        return all(
            channel in self.declared_error_budget and value <= self.declared_error_budget[channel]
            for channel, value in self.realized_error.items()
        )

    @property
    def is_claim(self) -> bool:
        """Whether this run supports a speedup *claim* at all.

        Speed alone is not a result (``LUNAR-TR-002``). The tier must have been admitted, it must
        have held the tolerance it ran under, and it must never have fallen back to the reference
        solver — otherwise the ratio is partly DEM-vs-DEM and measures nothing."""
        return self.admitted and self.within_budget and not self.escalated

    def as_provenance(self) -> dict[str, Any]:
        """The JSON-able summary — the speedup claim and the bounds it was made under."""
        return {
            "seed": self.seed,
            "admitted": self.admitted,
            "speedup": self.speedup,
            "within_budget": self.within_budget,
            "holds_declared_budget": self.holds_declared_budget,
            "escalated": self.escalated,
            "is_claim": self.is_claim,
            "dem": self.dem.as_provenance(),
            "surrogate": self.surrogate.as_provenance(),
            "declared_error_budget": dict(self.declared_error_budget),
            "admitted_tolerance": dict(self.admitted_tolerance),
            "realized_error": dict(self.realized_error),
        }


#: The outcome the surrogate engine records when a live query leaves its trust region. It is an
#: *escalation event*, not a deviation: its ``value`` is the OOD margin and its ``tolerance`` is
#: 0.0, so folding it in with the per-channel deviations would both invent an error channel the
#: surrogate never declared a budget for and — because no budget can contain it — make every OOD
#: run look like a budget breach. They are different facts, and the report keeps them apart.
_OOD_METRIC = "in_domain"


def _worst_realized(outcomes: tuple[ErrorBudgetOutcome, ...]) -> dict[str, float]:
    """The worst per-channel deviation-vs-DEM across every re-validation in the run.

    The engine records one outcome per re-validation per channel; the *claim* has to be made against
    the worst one, not the last or the mean — a surrogate that drifted out of budget once did not
    hold its bound."""
    worst: dict[str, float] = {}
    for outcome in outcomes:
        # `metric` / `value` are optional on the Core type (an outcome may be a bare pass/fail
        # verdict with no measured channel); such a row carries no deviation to be worst.
        if outcome.metric is None or outcome.value is None or outcome.metric == _OOD_METRIC:
            continue
        previous = worst.get(outcome.metric)
        if previous is None or outcome.value > previous:
            worst[outcome.metric] = outcome.value
    return worst


class _GranularRunner(SimEpisodeRunner):
    """A :class:`SimEpisodeRunner` restricted to the scenario's **granular** agents.

    The speedup claim is about the *excavation* task — the workload the surrogate substitutes for
    (surrogate.md §8) — and both granular engines own only ``dem_granular`` agents
    (``build_dem_engine`` skips the rest). Running them over a whole heterogeneous fleet would
    therefore leave the orbiter, the rover and the plant unadvanced and unobserved: a broken
    episode, and a ratio diluted by agents the surrogate has nothing to do with.

    So the comparison is scoped to the agents it is actually about. A scenario pinning no excavator
    has no granular task to benchmark, and says so rather than reporting a meaningless 1.0."""

    def resolve(self, resolved: ResolvedScenario, seed: int) -> ResolvedRun:
        run = super().resolve(resolved, seed)
        granular = tuple(a for a in run.scenario.agents if a.dynamics.kind == "dem_granular")
        if not granular:
            raise ValueError(
                f"scenario {resolved.scenario_id!r} pins no excavator (no asset declares a TOOL "
                "contact element), so it has no granular task to benchmark. The DEM-vs-surrogate "
                "speedup is a claim about excavation physics."
            )
        return ResolvedRun(
            run.scenario.model_copy(update={"agents": granular}),
            world_provider=run.world_provider,
            resource_field=run.resource_field,
            content_hashes=run.content_hashes,
            connectivity=None,  # comms masking is orthogonal to a physics-cost comparison
            scoring=run.scoring,
            belief=run.belief,
        )


class FidelitySpeedupRunner:
    """Runs a Bench scenario at both fidelity tiers and reports the speedup at its error bound.

    It **is** a Bench ``EpisodeRunner``: ``__call__(resolved, policy, seed) -> EpisodeTrace``
    returns the *surrogate-tier* trace — the tier under test, the one whose physics a scored run
    would actually use. The DEM run is the reference it is measured against, not the result. The
    speedup numbers ride out on :attr:`reports`, since an ``EpisodeTrace`` has nowhere to carry a
    duration and must not (Bench's runner contract forbids wall-clock inside the run).

    ``surrogate`` is the loaded tier under test; its Core manifest carries the
    ``recommended_error_budget`` that both admits it (via the RM-P1-SIM-03 scheduler) and is the
    bound its realized error is judged against. ``tolerance`` overrides the task tolerance the tier
    is admitted under — by default the surrogate's own declared budget, i.e. "substitute it wherever
    it claims to be good enough"."""

    __name__ = "astro-mine-sim/fidelity-speedup"

    def __init__(
        self,
        *,
        store: BundleStore,
        surrogate: LoadedSurrogate,
        provider_factories: dict[str, ProviderFactory] | None = None,
        horizon_steps: int | None = None,
        dt_s: float = _CONTACT_DT_S,
        tool_speed_mps: float | None = None,
        recording_dir: Path | str | None = None,
        scoring_context: ScoringContext | None = None,
        tolerance: Mapping[str, float] | None = None,
        revalidate_every: int | None = None,
        verify: bool = True,
    ) -> None:
        self._surrogate = surrogate
        # Re-validate at the horizon the tier's budget was calibrated to hold at, unless the caller
        # overrides. The budget only bounds the drift over `budget_horizon_steps` (surrogate#23);
        # grading it over a longer rollout checks a bound the producer never made — the engine
        # refuses that, so honouring the declared horizon by default is what keeps a well-made tier
        # runnable without the caller having to know its internals.
        self._revalidate_every = (
            surrogate.budget_horizon_steps if revalidate_every is None else revalidate_every
        )
        self._tolerance: Mapping[str, float] = (
            dict(surrogate.recommended_error_budget) if tolerance is None else dict(tolerance)
        )
        # The engine each tier's run builds, captured as it is built. The Simulator keeps its engine
        # private (it is rebuilt per reset), so the factory closure is the only seam through which a
        # caller can reach the live AdaptiveGranularEngine afterwards — and its realized
        # error-budget outcomes are the whole error half of the claim.
        self._built: list[RegimeEngine] = []
        self._last_surrogate_run: tuple[Trace, ResolvedRun]

        common: dict[str, Any] = {
            "store": store,
            "provider_factories": provider_factories,
            "horizon_steps": horizon_steps,
            # A granular tier integrates *contact*, not mission time: at a Bench spec's 60 s tick a
            # DEM bed would sub-step ~78 000 times per step per agent. The comparison is run at a
            # contact-scale tick on both sides — the ratio is only meaningful if the two tiers are
            # asked to model the same seconds, which they are.
            "dt_s": dt_s,
            # The blade sweep speed. SADF cannot carry it (see `sim_scenario_from_spec`), so the
            # benchmark sets it — and it must land inside the surrogate's declared trust region or
            # the tier goes out-of-domain on its first query and escalates straight back to DEM,
            # leaving nothing to measure. Default: the midpoint of the tier's own trust region.
            "tool_speed_mps": (
                _trust_region_midpoint(surrogate) if tool_speed_mps is None else tool_speed_mps
            ),
            "recording_dir": recording_dir,
            "scoring_context": scoring_context,
            "verify": verify,
            # Both tiers need the excavator on the DEM granular block: the DEM run *is* it, and the
            # surrogate run substitutes for it. Without this the asset routes to the reduced-order
            # granular kind and neither tier is reachable from a Bench spec at all.
            "dem_tier": True,
        }
        self._dem = _GranularRunner(
            engine_factory=self._dem_factory,
            fidelity=FidelityPolicy(pinned_tier=FidelityTier.ARTICULATED),
            **common,
        )
        self._surrogate_runner = _GranularRunner(
            engine_factory=self._surrogate_factory,
            # The error budget is what *lets* the scheduler admit the surrogate tier; with no budget
            # `build_scheduled_granular_engine` falls back to DEM by design (LUNAR-TR-002: no
            # substitution without a declared, satisfied bound).
            fidelity=FidelityPolicy(error_budget=dict(self._tolerance)),
            **common,
        )
        self._reports: dict[int, SpeedupReport] = {}

    def _dem_factory(self, scenario: Scenario, rng: RngStreams) -> RegimeEngine:
        engine: RegimeEngine = build_dem_engine(scenario, rng)
        self._built.append(engine)
        return engine

    def _surrogate_factory(self, scenario: Scenario, rng: RngStreams) -> RegimeEngine:
        engine = build_scheduled_granular_engine(
            scenario,
            rng,
            self._surrogate,
            policy=scenario.fidelity,
            revalidate_every=self._revalidate_every,
        )
        self._built.append(engine)
        return engine

    @property
    def reports(self) -> dict[int, SpeedupReport]:
        """The speedup report for each seed run so far — the deliverable (surrogate.md §8, §12)."""
        return dict(self._reports)

    @property
    def recordings(self) -> dict[int, Path]:
        """The surrogate-tier MCAP written for each seed — the trace a scored run would use."""
        return self._surrogate_runner.recordings

    def measure(
        self, resolved: ResolvedScenario, policy: Policy, seed: int
    ) -> tuple[SpeedupReport, EpisodeTrace]:
        """Run ``seed`` at both tiers; report the ratio + the realized-vs-declared error.

        Returns the report and the **surrogate-tier** :class:`EpisodeTrace` — the tier under test —
        so a caller driving this through Bench's seam scores the run it just measured rather than
        paying for a third episode."""
        dem_timing, _ = self._run_tier(self._dem, resolved, policy, seed)
        surrogate_timing, surrogate_engine = self._run_tier(
            self._surrogate_runner, resolved, policy, seed
        )

        # `build_scheduled_granular_engine` returns an AdaptiveGranularEngine only when the
        # scheduler admitted the surrogate tier; otherwise it hands back a plain DEM engine. The
        # engine's own declared tier is therefore the admission verdict — read it off what really
        # ran, rather than re-deriving the scheduler's decision and risking the two disagreeing.
        outcomes = tuple(getattr(surrogate_engine, "error_budget_outcomes", ()) or ())
        report = SpeedupReport(
            seed=seed,
            admitted=surrogate_timing.tier == FidelityTier.SURROGATE.value,
            dem=dem_timing,
            surrogate=surrogate_timing,
            declared_error_budget=dict(self._surrogate.recommended_error_budget),
            admitted_tolerance=dict(self._tolerance),
            realized_error=_worst_realized(outcomes),
            outcomes=outcomes,
        )
        self._reports[seed] = report

        trace, run = self._last_surrogate_run
        return report, self._surrogate_runner.to_episode_trace(trace, run)

    def _run_tier(
        self, runner: SimEpisodeRunner, resolved: ResolvedScenario, policy: Policy, seed: int
    ) -> tuple[EngineTiming, RegimeEngine | None]:
        """Run one tier, returning its measured wall-clock and the engine that spent it.

        The Simulator rebuilds its engine per reset and keeps it private, so the factory closure is
        the only seam through which the live engine can be reached afterwards — and the surrogate
        engine's realized error-budget outcomes are the entire error half of the claim."""
        del self._built[:]
        result = runner.run(resolved, policy, seed)
        if runner is self._surrogate_runner:
            self._last_surrogate_run = result
        # The engine the Simulator drove is a TimedEngine wrapper; what we captured in the factory
        # is the tier's own engine, whose surface carries the error-budget outcomes.
        return runner.timings[seed], (self._built[-1] if self._built else None)

    def __call__(self, resolved: ResolvedScenario, policy: Policy, seed: int) -> EpisodeTrace:
        """The Bench ``EpisodeRunner`` seam — returns the surrogate-tier trace (the tier under
        test); the speedup lands on :attr:`reports`."""
        _, trace = self.measure(resolved, policy, seed)
        return trace
