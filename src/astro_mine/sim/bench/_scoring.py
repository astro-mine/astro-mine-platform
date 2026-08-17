# SPDX-License-Identifier: Apache-2.0
"""Sim ``Trace`` → Bench ``EpisodeTrace`` — the scored-input projection (bench.md §6).

Bench scores an :class:`~astro_mine.bench.metrics.EpisodeTrace`: an ordered stream of **Core**
``Observation``s plus a :class:`~astro_mine.bench.metrics.ScoringContext` of scorer-only parameters.
The observation stream is exactly what a Sim run produces, so that half is a projection. The
``ScoringContext`` is the interesting half, and this module is careful about it.

**What Sim can honestly supply, and what it cannot.**

Sim *can* derive, from the run itself and the content it resolved:

- ``water_species`` / ``discovery_species`` — read off the assets' SADF sensor declarations (the
ISRU
  storage gauge's species, and the prospecting sensor's), so they are the species the run actually
  reports rather than a guess;
- ``night_intervals`` — computed by sampling the **pinned world's** illumination across the
episode's
  own epochs. A lunar night is a fact about the world, and the world is resolved content, so this is
  measured, not assumed;
- ``survivable_temperature_k`` — the coldest operating temperature the assets' SADF thermal budgets
  declare.

Sim will not **fabricate** the belief fields (``prior_belief``, ``belief_history``, ``psr_cells``).
A belief is [Prospect](prospect.md)'s: Sim renders sensor observations *of* a sealed field and
deliberately never maintains a posterior over it (sim.md §5 — "never a point guess"). Synthesizing
one here would be exactly the fabrication the architecture forbids.

What Sim *does* do, since #66, is **carry a real one**. A scenario's prospect pin rebuilds a
conditionable belief through the ``prior_recipe`` producer factory, and
:mod:`astro_mine.sim.bench._belief` replays the run's own prospecting readings against it — real
prior, real observations, Prospect's own Bayes, reached without importing Prospect
(conventions.md §1.1). That is not fabrication; it is the difference between inferring a posterior
and inventing one.

The distinction survives in the degraded case, which is the part worth keeping honest: with no
producer installed there is no belief, the fields stay at their defaults, and
``information_gain`` / ``psr_area_characterized`` score *not applicable* rather than zero. A run
that observed nothing likewise yields nothing. Both are asserted by tests rather than papered over.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, cast

from astro_mine.core.messages.model import Observation
from astro_mine.core.sadf.enums import SensorKind
from astro_mine.core.scoring import EpisodeTrace, ScoringContext
from astro_mine.core.world import IlluminationState
from astro_mine.sim.bench._belief import BeliefSource, belief_fields_for

if TYPE_CHECKING:
    from astro_mine.bench.scenario import ScoringSpec
    from astro_mine.core.world import WorldProvider
    from astro_mine.sim.runtime.episode import Trace
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = ["episode_trace_from", "night_intervals", "observations_from", "scoring_context_for"]

#: The sensor kinds that observe the resource being *prospected for* (as opposed to the resource
#: being stored). The discovery-latency metric thresholds their readings.
_DISCOVERY_KINDS = frozenset(
    {
        SensorKind.NEUTRON_SPECTROMETER,
        SensorKind.NIR_SPECTROMETER,
        SensorKind.GPR,
        SensorKind.DRILL_ASSAY,
        SensorKind.MASS_SPECTROMETER,
    }
)


def observations_from(trace: Trace) -> tuple[Observation, ...]:
    """The Core observation stream of a Sim :class:`~astro_mine.sim.runtime.Trace`, in log order.

    The canonical frames a Trace carries are exactly the per-agent observation maps Bench scores, so
    this flattens them back into Core models — the same projection Bench's own MCAP reader performs
    on
    a recorded run, which is why the two paths agree."""
    observations: list[Observation] = []
    for frame in trace.frames:
        for payload in frame.get("observations", {}).values():
            observations.append(Observation.model_validate(payload))
    return tuple(observations)


def night_intervals(
    world: WorldProvider | None,
    scenario: Scenario,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[float, float], ...]:
    """The ``(start_s, end_s)`` sim-time windows the site is **in shadow**, from the pinned world.

    Measured, not assumed: the episode's own epochs are sampled against the resolved world's
    illumination model, so the night windows the survival metric scores against are the ones the
    run's
    power/thermal evolution actually experienced. With no world resolved, there are no measured
    windows and the tuple is empty (degrade, don't lie — a fabricated night would silently change
    the
    score)."""
    if world is None:
        return ()
    windows: list[tuple[float, float]] = []
    start: float | None = None
    for tick in range(scenario.horizon_steps + 1):
        sim_time_s = tick * scenario.dt_s
        epoch = type(scenario.start_epoch)(
            tdb_seconds=scenario.start_epoch.tdb_seconds + sim_time_s,
            scale=scenario.start_epoch.scale,
        )
        dark = world.sample(position, epoch=epoch).illumination.state is IlluminationState.SHADOW
        if dark and start is None:
            start = sim_time_s
        elif not dark and start is not None:
            windows.append((start, sim_time_s))
            start = None
    if start is not None:
        windows.append((start, scenario.horizon_steps * scenario.dt_s))
    return tuple(windows)


def _species(
    scenario: Scenario,
    *,
    kind: SensorKind | None = None,
    kinds: frozenset[SensorKind] | None = None,
) -> str | None:
    """The resource species the assets' matching sensors report (first declared, in agent order)."""
    for agent in scenario.agents:
        for sensor in agent.sensors:
            matches = (
                sensor.kind is kind if kind is not None else bool(kinds and sensor.kind in kinds)
            )
            if matches and sensor.resource is not None:
                return sensor.resource.species
    return None


def _survivable_temperature_k(scenario: Scenario) -> float | None:
    """The coldest temperature the assets' SADF thermal budgets declare they survive.

    The **survival** range is the right floor for the nights-survived metric (SADF names it
    "survival
    floor for lunar night"); an asset that declares only an operating range falls back to that."""
    minima: list[float] = []
    for agent in scenario.agents:
        thermal = agent.thermal
        if thermal is None:
            continue
        band = thermal.survival_range_k or thermal.operating_range_k
        minima.append(band.min)
    return min(minima) if minima else None


def _overlay(derived: ScoringContext, override: ScoringContext) -> ScoringContext:
    """Lay the fields a caller actually set over the derived context.

    ``override`` used to *replace* the derivation wholesale, which quietly destroyed everything
    Sim had measured honestly: a caller injecting only a belief also discarded the
    ``night_intervals`` read off the pinned world and the ``survivable_temperature_k`` read off the
    fleet's thermal budgets, so ``nights_survived`` silently stopped scoring. With the scenario now
    able to pin scoring parameters too, a replacing override would discard those as well — so the
    seam merges instead.

    "Actually set" means *differs from the dataclass default*. A caller who deliberately sets a
    field back to its default is therefore indistinguishable from one who left it alone; that is
    the same trade the spec's content address makes, and it is the right one here because every
    default in :class:`ScoringContext` means "not supplied" rather than a meaningful value.
    """
    blank = ScoringContext()
    updates = {
        f.name: getattr(override, f.name)
        for f in fields(ScoringContext)
        if getattr(override, f.name) != getattr(blank, f.name)
    }
    return replace(derived, **updates)


def scoring_context_for(
    scenario: Scenario,
    *,
    world: WorldProvider | None = None,
    discovery_threshold: float = 0.0,
    scoring: ScoringSpec | None = None,
    override: ScoringContext | None = None,
) -> ScoringContext:
    """The scorer-only context a Sim run can honestly supply (see the module docstring).

    Three sources, in precedence order — **spec-pinned beats caller-supplied beats default**:

    1. ``scoring`` — what the *scenario* pins (bench#63). A task that states its own
       ``cell_area_m2`` or thresholds outranks a runner constructed by someone who did not know
       what the task wanted. Each field is optional; ``None`` means the scenario pins nothing and
       the next source applies.
    2. ``discovery_threshold`` — the runner's constructor argument, for a caller scoring a
       scenario that pins none.
    3. :class:`ScoringContext`'s own defaults. Two of these are traps rather than neutral values —
       ``characterized_variance_threshold=0.0`` is unsatisfiable and ``discovery_threshold=0.0``
       trips at tick 0 — which is precisely why (1) exists.

    ``override`` injects the **belief** fields (``prior_belief`` / ``belief_history`` /
    ``psr_cells``), which are Prospect's to produce and which Sim will not fabricate. It is
    overlaid, not substituted — see :func:`_overlay`.

    ``ScoringSpec.psr_region`` is deliberately **not** consumed here. Resolving a region to the
    opaque cell ids the metric wants requires the Bench/Prospect cell-id convention that
    astro-mine-sim#66 settles, and ``psr_area_characterized`` additionally needs a
    ``belief_history`` nothing supplies yet — so resolving it alone would move no metric. It rides
    with the belief work.
    """
    water = _species(scenario, kind=SensorKind.RESOURCE_STORAGE) or "water"
    discovery = _species(scenario, kinds=_DISCOVERY_KINDS) or water
    kwargs: dict[str, Any] = {
        "water_species": water,
        "discovery_species": discovery,
        "discovery_threshold": discovery_threshold,
        "night_intervals": night_intervals(world, scenario),
        "survivable_temperature_k": _survivable_temperature_k(scenario),
    }
    if scoring is not None:
        pinned = {
            "cell_area_m2": scoring.cell_area_m2,
            "characterized_variance_threshold": scoring.characterized_variance_threshold,
            "discovery_threshold": scoring.discovery_threshold,
        }
        kwargs.update({k: v for k, v in pinned.items() if v is not None})
    derived = ScoringContext(**kwargs)
    return derived if override is None else _overlay(derived, override)


def episode_trace_from(
    trace: Trace,
    scenario: Scenario,
    *,
    world: WorldProvider | None = None,
    context: ScoringContext | None = None,
    discovery_threshold: float = 0.0,
    scoring: ScoringSpec | None = None,
    belief: object | None = None,
) -> EpisodeTrace:
    """Project a Sim run into the :class:`EpisodeTrace` Bench's metric set scores.

    ``belief`` is the conditionable belief the scenario's prospect pin rebuilt. When one is
    present, the run's own prospecting readings are replayed against it and the resulting prior /
    posterior fill the belief-quality fields — the fields this module's docstring says Sim will not
    *fabricate*. Conditioning a real prior on real observations is not fabrication; inventing a
    posterior with no prior behind it would be, which is why absence stays not-applicable.
    """
    observations = observations_from(trace)
    derived = scoring_context_for(
        scenario,
        world=world,
        discovery_threshold=discovery_threshold,
        scoring=scoring,
        override=context,
    )
    if belief is not None:
        fields = belief_fields_for(
            cast("BeliefSource", belief),
            observations,
            scenario,
            psr_region=None
            if scoring is None or scoring.psr_region is None
            else (
                scoring.psr_region.lat_deg,
                scoring.psr_region.lon_deg,
            ),
        )
        if fields:
            derived = replace(derived, **fields)  # type: ignore[arg-type]
    return EpisodeTrace(observations=observations, context=derived)
