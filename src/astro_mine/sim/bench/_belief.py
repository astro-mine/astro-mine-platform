"""Driving a real Prospect belief from a run's own observations (#66).

Sim will not fabricate a belief. It renders sensor observations *of* a sealed field and
deliberately never maintains a posterior over one (``sim.md`` §5, and the module docstring of
:mod:`astro_mine.sim.bench._scoring` says so at length) — so ``information_gain`` and
``psr_area_characterized`` scored not-applicable on every Sim-backed run, and the anchor's
headline science claim went unmeasured end to end.

The missing piece was never the observations: a run already emits, every tick, exactly what a
Bayesian update needs — a value, its realized ``noise_sigma`` likelihood, the observing agent's
pose, the epoch, and the instrument tag. What was missing is something to *condition*, and that
is Prospect's to own (``prospect.md`` §6).

So Sim asks for one. :class:`BeliefSource` is the shape it needs, declared **here** and satisfied
structurally by whatever the ``prior_recipe`` producer hands back — the same arrangement as
:class:`~astro_mine.sim.comms.ConnectivitySource`, which Link's sampler satisfies without Sim
importing Link (``conventions.md`` §1.1). Every value crossing the seam is a Core type, so no
Prospect name appears in this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astro_mine.bench.metrics import BeliefSnapshot
from astro_mine.core.sadf.enums import SensorKind

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from astro_mine.core.messages.model import Observation, SensorReading
    from astro_mine.core.resource import FieldDistribution
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = ["BeliefSource", "belief_fields_for", "prospecting_sensors"]

#: The sensor kinds that observe the resource being *prospected for*. A belief conditions on these
#: and on nothing else: a storage gauge reads a tank, not the ground, and feeding its readings to
#: a resource field would condition the belief on the swarm's own inventory.
PROSPECTING_KINDS = frozenset(
    {
        SensorKind.NEUTRON_SPECTROMETER,
        SensorKind.NIR_SPECTROMETER,
        SensorKind.GPR,
        SensorKind.DRILL_ASSAY,
        SensorKind.MASS_SPECTROMETER,
    }
)


@runtime_checkable
class BeliefSource(Protocol):
    """A conditionable belief over the scenario's pinned prior, addressed by opaque cell id.

    Structurally identical to Prospect's ``GriddedBelief``: a caller that has Prospect installed
    gets one from the ``prior_recipe`` entry point, since Sim depends only on this shape and never
    on the Prospect package. Implementations are immutable — :meth:`observe` returns a new belief —
    so a prior can be read after conditioning.

    Cell ids are **opaque**. Sim neither mints nor parses them; it only checks that the ids from a
    prior and a posterior correspond, which they do by construction because both come from one
    source. The grid and the projection behind them are the producer's.
    """

    def observe(
        self, readings: Iterable[tuple[SensorReading, Sequence[float], float]]
    ) -> BeliefSource:
        """Condition on ``(reading, position, time_s)`` triples; return the updated belief."""
        ...

    def cells(self) -> Mapping[str, FieldDistribution]:
        """Every cell's current distribution, keyed by cell id."""
        ...

    def cells_in_region(
        self, *, lat_deg: tuple[float, float], lon_deg: tuple[float, float]
    ) -> frozenset[str]:
        """The ids of cells whose centres fall in a planetocentric lat/lon window."""
        ...


def prospecting_sensors(scenario: Scenario) -> dict[str, frozenset[str]]:
    """``{agent_id: {sensor name}}`` for the sensors that observe the prospected resource."""
    found: dict[str, frozenset[str]] = {}
    for agent in scenario.agents:
        names = {s.name for s in agent.sensors if s.kind in PROSPECTING_KINDS}
        if names:
            found[agent.agent_id] = frozenset(names)
    return found


def _observed_readings(
    observations: Iterable[Observation], sensors: Mapping[str, frozenset[str]]
) -> list[tuple[SensorReading, tuple[float, float, float], float]]:
    """The run's prospecting readings, each with where and when it was taken.

    The position is the observing agent's own body-fixed pose — the same coordinates the sensor
    sampled the sealed field at (``runtime/episode.py`` renders every reading from
    ``self_state.pose``). Passing them through unchanged is what makes the belief condition in the
    frame its observations were rendered in; converting here would introduce a disagreement between
    what was measured and what is being inferred.
    """
    out: list[tuple[SensorReading, tuple[float, float, float], float]] = []
    for observation in observations:
        names = sensors.get(observation.agent_id)
        if not names:
            continue
        translation = observation.self_state.pose.translation_m
        at = (float(translation.x), float(translation.y), float(translation.z))
        for reading in observation.sensors:
            if reading.sensor in names:
                out.append((reading, at, float(observation.sim_time_s)))
    return out


def belief_fields_for(
    source: BeliefSource,
    observations: Iterable[Observation],
    scenario: Scenario,
    *,
    psr_region: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> dict[str, object]:
    """The belief half of a scoring context: prior, posterior history, and the PSR mask.

    Conditions ``source`` on every prospecting reading the run emitted, in tick order, and returns
    the ``prior_belief`` / ``belief_history`` / ``psr_cells`` fields the belief-quality metrics
    consume. The result is empty when the run observed nothing, so a run that prospected nothing
    still reports *not applicable* rather than a fabricated zero.

    **One update, not a chain.** Both metrics read only the final snapshot, and a belief re-fuses
    from the prior over its whole log on every call — so conditioning once over the ordered log is
    both what the metrics need and the only shape that is not quadratic in episode length. The
    replay property makes the single batch identical to an incremental chain.

    **Scoped to the PSR when the scenario pins one.** The anchor asks for *"ice-probability
    posterior uncertainty reduced ≥ X% over the target PSR"* (``scenarios/1`` §3), not over the
    whole grid — and a field-wide sum would be dominated by the tens of thousands of cells nobody
    flew over, where the gain is ~0. Scoping also keeps both metrics on one cell set, so
    ``psr_area_characterized`` measures the same ground ``information_gain`` scores.
    """
    sensors = prospecting_sensors(scenario)
    if not sensors:
        return {}
    readings = _observed_readings(observations, sensors)
    if not readings:
        return {}

    psr_cells = (
        frozenset()
        if psr_region is None
        else source.cells_in_region(lat_deg=psr_region[0], lon_deg=psr_region[1])
    )
    scope = psr_cells or None

    def _snapshot(belief: BeliefSource) -> dict[str, FieldDistribution]:
        cells = belief.cells()
        return dict(cells) if scope is None else {c: cells[c] for c in scope if c in cells}

    prior = _snapshot(source)
    posterior = _snapshot(source.observe(readings))
    if not prior or not posterior:
        return {}

    last_time = readings[-1][2]
    return {
        "prior_belief": prior,
        "belief_history": (BeliefSnapshot(sim_time_s=last_time, cells=posterior),),
        "psr_cells": psr_cells,
    }
