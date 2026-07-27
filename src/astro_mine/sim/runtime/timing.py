"""Wall-clock instrumentation for the stepping core — the real-time factor (sim.md §10).

Every ``elapsed_s`` elsewhere in Sim is *sim* time: the modeled seconds a scenario advances.
This module measures the other axis — the **wall-clock** seconds spent computing them — which
is the numerator of the surrogate's defining claim: 10^2-10^4x speedup over the high-fidelity
DEM solver at a declared error bound (surrogate.md §8; ``LUNAR-TR-002``).

The measurement rides on a :class:`TimedEngine`, a pass-through
:class:`~astro_mine.sim.engines.RegimeEngine` that brackets the one hot call — ``advance`` —
and accumulates into a caller-held :class:`TimingRecorder`. It composes through the existing
``engine_factory`` seam, so no engine implements anything and no engine is edited.

**Timing never enters a content hash.** It is non-deterministic by nature: the same seeded
episode is byte-identical run to run but never takes the same nanoseconds twice. So a
:class:`~astro_mine.sim.runtime.episode.Trace` carries its timing on a field that
:meth:`~astro_mine.sim.runtime.episode.Trace.to_canonical_json` does not serialize, and the
MCAP envelope carries it as a sibling of the environment stamp, outside ``content_hash``
(:mod:`astro_mine.sim.recording`) — the same treatment, for the same reason: recorded for
audit, excluded from the reproducibility key. Bench's ``EpisodeRunner`` contract states the
rule from the other side: a conforming runner is a pure function of
``(scenario_hash, policy, seed)`` with *no wall-clock* (``bench/baseline/_runner.py``).
Measuring the run from outside is fine; letting the measurement back into the run is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.sim.engines import CouplingState, EngineDescriptor, EngineFactory, RegimeEngine
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = ["EngineTiming", "TimedEngine", "TimingRecorder", "timed_engine_factory"]

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class EngineTiming:
    """One engine's measured wall-clock over an episode, and the tier that spent it.

    ``tier`` is the engine's *declared* fidelity tier (its
    :class:`~astro_mine.sim.engines.EngineDescriptor`), which is what makes this a
    **per-tier** measurement: a DEM-tier run and a surrogate-tier run of the same seeded
    scenario each report their own wall-clock, and their ratio is the speedup claim.

    :attr:`real_time_factor` is modeled seconds per wall-clock second — sim.md §10's
    "step wall-clock vs. sim-time" signal. Above 1.0 the run is faster than real time."""

    engine: str
    tier: str
    steps: int
    advance_wall_clock_s: float
    sim_time_s: float

    @property
    def real_time_factor(self) -> float | None:
        """Modeled seconds per wall-clock second, or ``None`` if nothing was measured.

        A zero-duration measurement is reported as ``None`` rather than ``inf``: a clock
        too coarse to see the work is an absent measurement, not an infinitely fast one."""
        if self.advance_wall_clock_s <= 0.0:
            return None
        return self.sim_time_s / self.advance_wall_clock_s

    @property
    def mean_step_wall_clock_s(self) -> float | None:
        """Mean wall-clock seconds per ``advance`` call, or ``None`` if no step ran."""
        if self.steps <= 0:
            return None
        return self.advance_wall_clock_s / self.steps

    def as_provenance(self) -> dict[str, Any]:
        """The JSON-able form carried outside every reproducibility hash."""
        return {
            "engine": self.engine,
            "tier": self.tier,
            "steps": self.steps,
            "advance_wall_clock_s": self.advance_wall_clock_s,
            "sim_time_s": self.sim_time_s,
            "real_time_factor": self.real_time_factor,
            "mean_step_wall_clock_s": self.mean_step_wall_clock_s,
        }


@dataclass(slots=True)
class TimingRecorder:
    """The caller-held sink a :class:`TimedEngine` accumulates into.

    The recorder outlives the engine, which the stepping core rebuilds on every ``reset``
    (:meth:`~astro_mine.sim.runtime.episode.Simulator.reset`). :meth:`rewind` is therefore
    called per reset so a re-run measures that run, not the sum of every run before it —
    while the *handle* stays valid across resets for the caller that injected it."""

    engine: str = ""
    tier: str = ""
    steps: int = 0
    advance_ns: int = 0
    sim_time_s: float = 0.0
    #: Every completed episode's timing, in order — one entry per ``reset``.
    episodes: list[EngineTiming] = field(default_factory=list)

    def rewind(self, *, engine: str, tier: str) -> None:
        """Bank the episode just finished (if any) and start a fresh one."""
        if self.steps > 0:
            self.episodes.append(self.snapshot())
        self.engine = engine
        self.tier = tier
        self.steps = 0
        self.advance_ns = 0
        self.sim_time_s = 0.0

    def record_advance(self, *, elapsed_ns: int, dt_s: float) -> None:
        """Accumulate one ``advance`` call."""
        self.steps += 1
        self.advance_ns += elapsed_ns
        self.sim_time_s += dt_s

    def snapshot(self) -> EngineTiming:
        """The timing accumulated since the last :meth:`rewind`."""
        return EngineTiming(
            engine=self.engine,
            tier=self.tier,
            steps=self.steps,
            advance_wall_clock_s=self.advance_ns / _NS_PER_S,
            sim_time_s=self.sim_time_s,
        )

    def as_provenance(self) -> dict[str, Any] | None:
        """The JSON-able envelope for the current episode, or ``None`` if nothing ran."""
        if self.steps <= 0:
            return None
        return self.snapshot().as_provenance()


class TimedEngine:
    """A pass-through :class:`~astro_mine.sim.engines.RegimeEngine` that times ``advance``.

    Structural, not nominal: it satisfies the engine Protocol by delegation, so it wraps any
    engine — reduced-order, DEM, surrogate, or the multi-domain coupler — without either side
    knowing. Only ``advance`` is bracketed; it is the single hot call the stepping core makes
    per tick (:meth:`~astro_mine.sim.runtime.episode.Simulator.step`), and the one whose cost
    the DEM-vs-surrogate ratio is about. The two ``perf_counter_ns`` reads it adds per step are
    nanoseconds against a DEM step's milliseconds, so the instrument does not move what it
    measures."""

    __slots__ = ("_inner", "_recorder")

    def __init__(self, inner: RegimeEngine, recorder: TimingRecorder) -> None:
        self._inner = inner
        self._recorder = recorder
        descriptor = inner.descriptor
        recorder.rewind(engine=descriptor.name, tier=descriptor.fidelity.tier.value)

    @property
    def inner(self) -> RegimeEngine:
        """The wrapped engine — for a caller that needs the engine's own surface (e.g. an
        :class:`~astro_mine.sim.engines.surrogate.AdaptiveGranularEngine`'s realized
        error-budget outcomes), which the Protocol does not carry. Run provenance also reads it,
        to see past the instrument to the engines that actually ran (#65)."""
        return self._inner

    @property
    def descriptor(self) -> EngineDescriptor:
        return self._inner.descriptor

    def apply_actions(self, actions: ActionBatch) -> None:
        self._inner.apply_actions(actions)

    def advance(self, dt_s: float) -> None:
        started_ns = perf_counter_ns()
        try:
            self._inner.advance(dt_s)
        finally:
            self._recorder.record_advance(elapsed_ns=perf_counter_ns() - started_ns, dt_s=dt_s)

    def export_coupling_state(self) -> CouplingState:
        return self._inner.export_coupling_state()

    def import_coupling_state(self, state: CouplingState) -> None:
        self._inner.import_coupling_state(state)

    def retire(self, agent_ids: Iterable[str]) -> None:
        self._inner.retire(agent_ids)


def timed_engine_factory(factory: EngineFactory, recorder: TimingRecorder) -> EngineFactory:
    """Wrap ``factory`` so every engine it builds is timed into ``recorder``.

    The stepping core rebuilds the engine per episode, so the wrapping has to happen at the
    *factory*, not at one engine instance — this is that seam."""

    def build(scenario: Scenario, rng: RngStreams) -> RegimeEngine:
        return TimedEngine(factory(scenario, rng), recorder)

    return build
