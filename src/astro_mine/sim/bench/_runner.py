"""The Sim-backed Bench runners — real-physics scoring + the determinism gate (RM-P0-SIM-11).

Bench ships two injectable, Core-typed runner seams and a dependency-clean **stand-in** behind each:

- ``bench.baseline.EpisodeRunner`` — ``(resolved, policy, seed) -> EpisodeTrace``, the *scoring*
path.
  Its default (``reference_episode_runner``) is documented as "a deterministic trace fixture, **not
  a
  physics engine**".
- ``bench.harness.Runner`` — ``(resolved, seed) -> RunOutcome``, the *determinism gate*. Its default
  (``reference_runner``) is "a pure, seeded function of the scenario hash ... enough to exercise the
  reproducibility oracle **without a physics engine**".

This module is the real thing behind both. The dependency direction is one-way and deliberate
(conventions.md §1.1; bench.md §2.2): **Bench never imports Sim**, so the runner that satisfies
Bench's
seams lives *here*, in the Sim repo, and Bench receives it by injection. It is optional
(``astro-mine-platform[sim-bench]``) and nothing in Sim's runtime imports it.

The path is real end to end: the scenario's pinned content is resolved through the RM-P1-SIM-01
:class:`~astro_mine.sim.runtime.ContentResolver` (never a hand-authored fixture), the episode is run
by
the actual stepping core under the injected Core :class:`~astro_mine.core.policy.Policy`, an
**MCAP**
recording is written with the run's provenance and its resolved content hashes, and the trace is
scored
by Bench's own metric set.

**Determinism.** Bench's runner contract requires a pure function of ``(scenario_hash, policy,
seed)``
— no wall-clock, no global RNG, no network. A Sim run already is one: the same seed drives the same
``RngStreams``, and :attr:`~astro_mine.sim.runtime.Trace.content_hash` is the byte-for-byte
reproducibility key the RM-P0-SIM-10 gate compares. The harness runner reports exactly that hash as
its
``determinism_key``, so Bench's gate and Sim's gate check *the same artifact* rather than two
lookalikes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.scoring import EpisodeScorer, RunOutcome, ScoringRefused
from astro_mine.sim.bench._policy import ValueChainPolicy, mode_table
from astro_mine.sim.bench._scenario import ResolvedRun, sim_scenario_from_spec
from astro_mine.sim.bench._scoring import episode_trace_from
from astro_mine.sim.kernels import furnish_metakernel
from astro_mine.sim.recording import record_episode
from astro_mine.sim.runtime import open_bundle_store
from astro_mine.sim.runtime.content import (
    ContentPin,
    ContentResolver,
    ScenarioContent,
    cargo_capacity_kg,
    describe_unresolved,
    is_digger,
)
from astro_mine.sim.runtime.timing import TimingRecorder

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astro_mine.bench.scenario import ResolvedScenario, ScenarioSpec
    from astro_mine.core.messages.model import Observation
    from astro_mine.core.policy import DecisionContext, Policy
    from astro_mine.core.resource import ResourceField
    from astro_mine.core.scoring import EpisodeTrace, ScoringContext
    from astro_mine.sim.comms import ConnectivitySource
    from astro_mine.sim.engines import EngineFactory
    from astro_mine.sim.runtime.content import BundleStore, ProviderFactory
    from astro_mine.sim.runtime.episode import Trace
    from astro_mine.sim.runtime.timing import EngineTiming
    from astro_mine.sim.scheduler import FidelityPolicy

__all__ = ["SIM_RUNNER_ID", "SimEpisodeRunner", "SimHarnessRunner", "sim_runner_provider"]

#: The runner identity Bench stamps onto a :class:`~astro_mine.bench.harness.Result`, so a
#: leaderboard entry records *which* runner produced it — a reference-fixture score and a
#: real-physics score are never confusable.
SIM_RUNNER_ID = "astro-mine-sim/0.1.0"


class SimEpisodeRunner:
    """A **Sim-backed** ``bench.baseline.EpisodeRunner`` — real physics behind Bench's scoring path.

    Satisfies Bench's protocol structurally: ``runner(resolved, policy, seed) -> EpisodeTrace``.
    Inject
    it with ``bench.baseline.run(spec, policy, runner=SimEpisodeRunner(store=...))``.

    Each call resolves the scenario's pinned content, runs the episode under the injected policy,
    **records an MCAP** (with the run's provenance and its resolved content hashes), and projects
    the
    trace into the :class:`EpisodeTrace` Bench scores. The MCAP is the artifact boundary the two
    components meet at (conventions.md §1.1): Bench can equally score it by reading the file back.

    ``store`` is the content store the pins resolve from. ``horizon_steps`` caps the episode below
    the
    spec's horizon — the anchor's is 43 200 ticks (a full lunar month), which is a *benchmark* run,
    not
    a smoke test. ``recording_dir`` is where the MCAPs land (a temp dir by default; pass one to keep
    them). ``scoring_context`` injects the **belief** fields Sim will not fabricate — see
    :mod:`astro_mine.sim.bench._scoring`.

    ``connectivity`` supplies the comms model (RM-P0-SIM-08): without it every observation leaves
    ``Observation.comms`` unset, and Bench's ``comms_robustness`` — the scoring definition of
    degrade-not-collapse — has nothing to measure and scores *not applicable*. It is an injected
    **override**, in the same sense as ``world_provider`` / ``resource_field``: a scenario that pins
    a link bundle now resolves its own ContactPlan through the content path (#53), so the default is
    the *pinned* comms model and this argument is only for a caller that wants a different one.
    A scenario with no link pin and no injected source runs unmasked, exactly as before.

    ``dem_tier`` and ``fidelity`` are the physics-fidelity dials (#51): ``dem_tier`` routes
    excavators to the high-fidelity DEM granular block, and ``fidelity`` is the
    :class:`~astro_mine.sim.scheduler.FidelityPolicy` tiers are admitted under (a pinned tier, or
    an ``error_budget`` a surrogate must hold to be substituted — ``LUNAR-TR-002``). Together they
    let a Bench-driven run be pinned to the DEM tier or the surrogate tier, which is what
    :class:`~astro_mine.sim.bench.FidelitySpeedupRunner` compares. A granular tier additionally
    needs ``dt_s``: a DEM bed integrates *contact*, so it must be driven at a contact-scale tick
    (~0.05 s), not a Bench spec's 60 s *mission* cadence — see :func:`sim_scenario_from_spec`."""

    #: Bench falls back to ``getattr(runner, "__name__", repr(runner))`` for a callable object's
    #: identity; give it the real one so a Result records that Sim produced the run.
    __name__ = SIM_RUNNER_ID

    def __init__(
        self,
        *,
        store: BundleStore,
        provider_factories: dict[str, ProviderFactory] | None = None,
        engine_factory: EngineFactory | None = None,
        resource_field: ResourceField | None = None,
        horizon_steps: int | None = None,
        dt_s: float | None = None,
        recording_dir: Path | str | None = None,
        scoring_context: ScoringContext | None = None,
        discovery_threshold: float = 0.0,
        contact_tier: bool = False,
        dem_tier: bool = False,
        tool_speed_mps: float | None = None,
        fidelity: FidelityPolicy | None = None,
        connectivity: ConnectivitySource | None = None,
        timing: TimingRecorder | None = None,
        verify: bool = True,
        allow_unresolved_content: bool = False,
    ) -> None:
        self._store = store
        self._provider_factories = provider_factories
        self._engine_factory = engine_factory
        self._resource_field = resource_field
        self._horizon_steps = horizon_steps
        self._dt_s = dt_s
        self._recording_dir = Path(recording_dir) if recording_dir is not None else None
        self._scoring_context = scoring_context
        self._discovery_threshold = discovery_threshold
        self._contact_tier = contact_tier
        self._dem_tier = dem_tier
        self._tool_speed_mps = tool_speed_mps
        self._fidelity = fidelity
        self._connectivity = connectivity
        self._timing = timing
        self._verify = verify
        self._allow_unresolved_content = allow_unresolved_content
        self._recordings: dict[int, Path] = {}
        self._timings: dict[int, EngineTiming] = {}

    @property
    def recordings(self) -> dict[int, Path]:
        """The MCAP written for each seed run so far — the artifact Bench scores across."""
        return dict(self._recordings)

    @property
    def timings(self) -> dict[int, EngineTiming]:
        """The measured wall-clock of each seed run so far (#51).

        Runner *state*, not a return value: Bench's ``EpisodeRunner`` hands back an
        ``EpisodeTrace``, which has nowhere to carry a duration — and rightly so, since a
        conforming runner must be a pure function of ``(scenario_hash, policy, seed)`` with no
        wall-clock in it. The measurement therefore rides out beside the artifacts, exactly as
        :attr:`recordings` does."""
        return dict(self._timings)

    def resolve(self, resolved: ResolvedScenario, seed: int) -> ResolvedRun:
        """Resolve the scenario's pinned content into a runnable Sim scenario (RM-P1-SIM-01).

        **Refuses a blind run.** If a pin resolved by digest but rebuilt no provider — the producer
        package that supplies its entry point is not installed — this raises rather than scoring
        (#67). A scorecard is a published claim, and there is no honest use for scoring the anchor
        against a world that was never loaded: the run would report `nights_survived` for a mission
        with no measured night and `comms_robustness` for a swarm nothing ever masked, while every
        digest in its provenance says the content was there.

        The library tier stays tolerant on purpose — `Simulator` and `run_episode` legitimately run
        provider-less, and injection through `provider_factories` is the documented pattern. Only
        the *scoring* path refuses. Pass ``allow_unresolved_content=True`` to opt out deliberately.

        The refusal is :class:`~astro_mine.bench.baseline.ScoringRefused` — Bench's own type for a
        runner declining to score, which its CLI presents as an actionable error rather than a
        traceback (``astro-mine-bench#79``). It was a bare ``RuntimeError``, which meant the only
        way to tell a deliberate refusal from an engine bug was to read the message text.
        """
        run = self._resolved_run(resolved, seed)
        # The scoring path's kernel pool (#80). Bench passes a content store and nothing else — it
        # has no vocabulary for SPICE and must not grow one (conventions.md §1.1) — so the runner
        # reads $ASTRO_MINE_SPICE_METAKERNEL itself, exactly as it already resolves its store from
        # $ASTRO_MINE_HUB_REGISTRY. Furnished here rather than in __init__ because only now is the
        # epoch window known, which is what lets a short kernel set fail in the first second of a
        # 30-day episode instead of ~18,000 ticks in.
        furnish_metakernel(scenario=run.scenario)
        if run.unresolved and not self._allow_unresolved_content:
            raise ScoringRefused(
                "refusing to score this scenario: "
                + describe_unresolved(run.unresolved)
                + "\nA scorecard is a claim about a run, and this run would not have modelled "
                "the content it pins. Install the producers above, or pass "
                "`SimEpisodeRunner(allow_unresolved_content=True)` to score anyway."
            )
        return run

    def _resolved_run(self, resolved: ResolvedScenario, seed: int) -> ResolvedRun:
        return sim_scenario_from_spec(
            resolved,
            store=self._store,
            seed=seed,
            provider_factories=self._provider_factories,
            verify=self._verify,
            horizon_steps=self._horizon_steps,
            dt_s=self._dt_s,
            contact_tier=self._contact_tier,
            dem_tier=self._dem_tier,
            tool_speed_mps=self._tool_speed_mps,
            fidelity=self._fidelity,
        )

    def run(
        self, resolved: ResolvedScenario, policy: Policy, seed: int
    ) -> tuple[Trace, ResolvedRun]:
        """Run one episode on real physics and record it — the shared core of both runner seams."""
        run = self.resolve(resolved, seed)
        directory = self._recording_dir or Path(tempfile.gettempdir())
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{resolved.scenario_id}-{seed}.mcap"
        recorder = TimingRecorder() if self._timing is None else self._timing
        trace = record_episode(
            run.scenario,
            path,
            seed=seed,
            policy=policy,
            engine_factory=self._engine_factory,
            # The *pinned* world and resource field drive the run: power/thermal and the framing
            # sensors sample the resolved world, and the resource sensors sample the resolved
            # (sealed) field. A caller may still inject a field explicitly (a sealed per-seed
            # realization is Prospect's to make, not the bundle's).
            world_provider=run.world_provider,
            resource_field=self._resource_field or run.resource_field,  # type: ignore[arg-type]
            # The pinned comms model (#53), unless the caller injected one. Without either the run
            # is comms-blind: `Observation.comms` stays unset and Bench's `comms_robustness` scores
            # *not applicable* (issue #52).
            connectivity=self._connectivity or run.connectivity,
            content_hashes=run.content_hashes,
            # A blind run says so in its own recording (#67). Only reachable with
            # `allow_unresolved_content=True`, since `resolve` otherwise refuses.
            unresolved=run.unresolved,
            timing=recorder,
        )
        self._recordings[seed] = path
        self._timings[seed] = recorder.snapshot()
        return trace, run

    def to_episode_trace(self, trace: Trace, run: ResolvedRun) -> EpisodeTrace:
        """Project a completed run into the :class:`EpisodeTrace` Bench's metric set scores."""
        return episode_trace_from(
            trace,
            run.scenario,
            world=run.world_provider,
            context=self._scoring_context,
            discovery_threshold=self._discovery_threshold,
            # What the *scenario* pins outranks this runner's constructor defaults (bench#63).
            scoring=run.scoring,
            belief=run.belief,
        )

    def __call__(self, resolved: ResolvedScenario, policy: Policy, seed: int) -> EpisodeTrace:
        """Bench's ``EpisodeRunner`` seam: a scorable trace from a real, recorded physics run."""
        trace, run = self.run(resolved, policy, seed)
        return self.to_episode_trace(trace, run)


class SimHarnessRunner:
    """A **Sim-backed** ``bench.harness.Runner`` — Bench's determinism gate, on real physics.

    Satisfies Bench's protocol structurally: ``runner(resolved, seed) -> RunOutcome``. Inject it
    with
    ``bench.harness.assert_reproducible(spec, SimHarnessRunner(...), runner_id=SIM_RUNNER_ID)``.

    ``determinism_key`` is Sim's own :attr:`~astro_mine.sim.runtime.Trace.content_hash` — the very
    hash
    the RM-P0-SIM-10 gate compares — so Bench's reproducibility oracle and Sim's check the **same
    artifact**, not two lookalikes. That is the whole point of wiring the real runner in: the gate
    now
    proves that *the physics* reproduces, where before it proved only that a hash of the scenario
    hash
    reproduced.

    A harness runner takes no policy (the gate measures the environment's reproducibility, not a
    submission's), so it drives the episode open-loop unless a ``policy`` is supplied.

    **The scorer is injected, and required.** A ``RunOutcome`` carries metric values, and resolving
    a scenario's metric references to implementations is the benchmark's job, not the engine's.
    Sim used to call ``bench.metrics.resolve_metrics``/``score`` directly — reaching for a
    collaborator it could be given, which is what §3.3 forbids and what made ``sim -> bench`` a
    runtime lateral edge pointing up the layer table. It now takes a Core
    :class:`~astro_mine.core.scoring.EpisodeScorer`; Bench passes its own
    :func:`~astro_mine.bench.metrics.scored_metric_values` when it constructs the runner. There is
    no default, because a default would have to be a Bench import."""

    __name__ = SIM_RUNNER_ID

    def __init__(
        self,
        episodes: SimEpisodeRunner,
        *,
        scorer: EpisodeScorer,
        policy: Policy | None = None,
    ) -> None:
        self._episodes = episodes
        self._scorer = scorer
        self._policy = policy

    def __call__(self, resolved: ResolvedScenario, seed: int) -> RunOutcome:
        """Bench's ``Runner`` seam: the run's determinism key plus its scored metrics."""
        policy: Policy = self._policy or _OpenLoopPolicy()
        trace, run = self._episodes.run(resolved, policy, seed)
        episode = self._episodes.to_episode_trace(trace, run)
        values = self._scorer(
            {seed: episode},
            resolved.spec.metrics,
            scenario_id=resolved.scenario_id,
            runner=SIM_RUNNER_ID,
        )
        return RunOutcome(
            determinism_key=trace.content_hash,  # Sim's own gate key — one artifact, not two
            # A `RunOutcome` carries `dict[str, float]`, so a *not-applicable* metric (a
            # belief-quality
            # metric with no injected belief; a discovery that never happened) cannot be represented
            # as
            # None here. It is reported as 0.0 **only** in this reproducibility fingerprint — the
            # authoritative score is the Scorecard from `bench.baseline.run`, which keeps None as
            # None.
            metrics={
                metric: (0.0 if value is None else float(value)) for metric, value in values.items()
            },
        )


class _OpenLoopPolicy:
    """The no-op Core :class:`~astro_mine.core.policy.Policy` the determinism gate drives with.

    The gate asks "does *the environment* reproduce", not "does a submission score the same", so it
    steps open-loop — an empty action batch each tick, which is exactly what
    :func:`~astro_mine.sim.runtime.run_episode` does with no policy injected."""

    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        return ActionBatch()


#: Env the ``sim`` runner reads for its content store when Bench passes none (its ``score`` CLI
#: does): a local OCI-layout Hub registry, the same convention the ``astro-mine-sim run`` CLI uses.
_REGISTRY_ENV = "ASTRO_MINE_HUB_REGISTRY"


class _SimRunnerProvider:
    """The ``sim`` runner registered into Bench's ``astro_mine.bench.runners`` group (RM-P0-SIM-11).

    Bench discovers this by name and injects the runner it returns — it never imports Sim
    (conventions.md §1.1; bench.md §2.2). ``episode_runner`` / ``harness_runner`` wrap the
    already-shipping Sim runners; the content ``store`` resolves from ``$ASTRO_MINE_HUB_REGISTRY``
    when Bench passes none (its ``score`` CLI does), with a clear error — never a traceback — when
    neither a store nor the env is available (CX-LOCAL). A store passed explicitly (a path or a live
    store) is honoured, so a future Bench ``--registry`` drops in unchanged.
    """

    runner_id = SIM_RUNNER_ID

    def episode_runner(self, store: object | None = None) -> SimEpisodeRunner:
        return SimEpisodeRunner(store=self._resolve_store(store))

    def harness_runner(
        self, store: object | None = None, *, scorer: EpisodeScorer
    ) -> SimHarnessRunner:
        return SimHarnessRunner(
            SimEpisodeRunner(store=self._resolve_store(store)), scorer=scorer
        )

    def default_policy(self, spec: ScenarioSpec, store: object | None = None) -> Policy:
        """The anchor baseline for this scenario — a value-chain policy (#61, #64).

        Commands the modes each asset's capabilities imply *and* the two directives the value chain
        needs: a continuous dig for excavators, and a shuttle between dig site and plant for
        carriers. Mode alone moves nothing and digs nothing, so a mode-only baseline scored
        ``water_mass = 0.0`` however well the physics was coupled — the anchor's dig site and plant
        are ~12 km apart.

        Bench asks for this through its optional ``DefaultPolicyProvider`` seam when the caller
        names no policy, so ``astro-mine-bench score --runner sim`` stops falling back to a mode
        five of the anchor's six assets never declare. **The spec is what makes it honest**: the
        roster is resolved from ``spec.content.fleet``, i.e. the same pinned digests the episode
        runs on, rather than by artifact tag — a tag-resolved roster could describe a different
        asset version than the one scored (CX-REPRO).

        Only the fleet pins are pulled. ``ScenarioContent.world`` is optional, so building a mode
        table costs six SADF reads rather than materializing the world bundle, which for the anchor
        is ~460 MB of terrain this policy has no use for.
        """
        content = ScenarioContent(
            fleet=tuple(
                ContentPin(id=ref.id, reference=ref.content_hash) for ref in spec.content.fleet
            )
        )
        resolver = ContentResolver(self._resolve_store(store))
        resolved = resolver.resolve(content)
        assets = {agent_id: r.asset for agent_id, r in resolved.assets.items()}
        return ValueChainPolicy(
            modes=mode_table(assets),
            diggers=frozenset(a for a, asset in assets.items() if is_digger(asset)),
            carriers=frozenset(
                a for a, asset in assets.items() if cargo_capacity_kg(asset) is not None
            ),
            plants=frozenset(
                a
                for a, asset in assets.items()
                if asset.payload is not None and asset.payload.isru is not None
            ),
        )

    @staticmethod
    def _resolve_store(store: object | None) -> BundleStore:
        if isinstance(store, (str, Path)):
            return open_bundle_store(store)
        if store is not None:
            return cast("BundleStore", store)
        registry = os.environ.get(_REGISTRY_ENV)
        if not registry:
            raise RuntimeError(
                f"the 'sim' runner needs a content store: set ${_REGISTRY_ENV} to a local "
                "OCI-layout Hub registry (populate one with `astro-mine-bench fetch <scenario>`)."
            )
        return open_bundle_store(registry)


#: The ``sim`` runner provider Bench resolves through the ``astro_mine.bench.runners`` entry-point
#: group (registered in pyproject.toml). Satisfies Bench's ``BenchRunnerProvider`` contract.
sim_runner_provider = _SimRunnerProvider()

if TYPE_CHECKING:
    # Compile-time conformance to Bench's contract, so protocol drift is caught by mypy, not only by
    # the entry-point round-trip test. (Import is TYPE_CHECKING-only; this is the adapter package.)
    from astro_mine.bench.baseline import BenchRunnerProvider

    _: BenchRunnerProvider = sim_runner_provider
