"""The Phase-0 reference metric set (bench.md §3; scenario §13; LUNAR-FR-009).

Seven metrics, each a plugin with SI-consistent units, a direction, and a cross-seed
aggregation rule. Information gain, PSR-area characterized, and discovery latency drive the
**M0.1** prospecting-only score; water mass, energy/kg, and nights survived complete **M0.2**
(comms robustness spans both). Each computes deterministically from an :class:`EpisodeTrace`.

Reference metrics ship as *replaceable examples* — metric definitions for multi-week ISRU
campaigns are an open, RFC-governed question (bench.md §11), so these encode a defensible
Phase-0 choice, not a final answer.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass

from astro_mine.bench.metrics._metric import Metric, MetricComputationError, MetricValue
from astro_mine.core.objective import MetricAggregation, MetricDirection
from astro_mine.core.scoring import EpisodeTrace

__all__ = [
    "REFERENCE_METRICS",
    "CommsRobustness",
    "DiscoveryLatency",
    "EnergyPerKg",
    "InformationGain",
    "NightsSurvived",
    "PsrAreaCharacterized",
    "WaterMass",
]


def _total_stored_water(trace: EpisodeTrace) -> float:
    """The fleet's cumulative stored water: the latest ``kg`` tank reading per agent, summed.

    **Channel order is the contract.** An ISRU storage gauge reports
    ``values = [stored_water_kg, extraction_energy_j]`` — two quantities on one reading, so that a
    single MCAP channel carries both the ``water_mass`` and ``energy_per_kg`` inputs (RFC-0003;
    sim.md §5). Stored mass is **channel 0**. This previously read ``values[-1]``, which on a
    two-channel gauge is the cumulative extraction *energy* in joules, reported as kilograms of
    water (astro-mine-sim#61). A single-channel reading is unaffected — ``values[0]`` is
    ``values[-1]`` — so the fixture path scores identically.
    """
    species = trace.context.water_species
    latest: dict[str, tuple[int, float]] = {}
    for obs in trace.observations:
        for reading in obs.sensors:
            if (
                reading.resource_species == species
                and reading.unit == "kg"
                and reading.valid
                and reading.values
            ):
                prev = latest.get(obs.agent_id)
                if prev is None or obs.tick >= prev[0]:
                    latest[obs.agent_id] = (obs.tick, reading.values[0])
    return math.fsum(value for _, value in latest.values())


@dataclass(frozen=True, slots=True)
class WaterMass:
    """Total water mass in the ISRU tanks at episode end (higher is better)."""

    name: str = "water_mass"
    version: str = "0.1.0"
    unit: str = "kg"
    direction: MetricDirection = MetricDirection.HIGHER_BETTER
    aggregation: MetricAggregation = MetricAggregation.MEAN

    def compute(self, trace: EpisodeTrace) -> MetricValue:
        return MetricValue(value=_total_stored_water(trace), unit=self.unit)


def _total_extraction_energy(trace: EpisodeTrace) -> float:
    """The fleet's cumulative ISRU extraction energy (J): the latest gauge reading per agent.

    Channel **1** of the storage gauge, the sibling of the stored mass on channel 0 (RFC-0003;
    see :func:`_total_stored_water` for why the channel order is the contract). Cumulative and
    monotonic like the mass, so the latest reading per agent is the total. A single-channel
    reading declares no extraction energy and contributes nothing.
    """
    species = trace.context.water_species
    latest: dict[str, tuple[int, float]] = {}
    for obs in trace.observations:
        for reading in obs.sensors:
            if (
                reading.resource_species == species
                and reading.unit == "kg"
                and reading.valid
                and len(reading.values) > 1
            ):
                prev = latest.get(obs.agent_id)
                if prev is None or obs.tick >= prev[0]:
                    latest[obs.agent_id] = (obs.tick, reading.values[1])
    return math.fsum(value for _, value in latest.values())


@dataclass(frozen=True, slots=True)
class EnergyPerKg:
    """Total swarm energy spent per kg of water produced (lower is better).

    **Energy is the whole cost of the water, not one bus's share of it.** Two sources are summed:

    - **Battery discharge** — positive drops in ``battery_soc_j`` across consecutive ticks, per
      agent. This is what driving, digging and hauling cost, and it is what the metric measured
      on its own until now.
    - **ISRU extraction energy** — channel 1 of the storage gauge. Sim tracks the extraction
      process on a *dedicated bus* that deliberately never touches the survival battery, so that
      productivity accounting cannot perturb the night-survival termination (``sim.md``;
      RM-P0-SIM-07). That is the right call for the simulation and the wrong one for this metric:
      it meant the single term most obviously belonging in "energy per kg of water" was **emitted
      every tick and read by nothing**, so a swarm's extraction cost was free.

    The defect was invisible while ``water_mass`` was structurally zero — the metric returns
    not-applicable whenever no water was produced — and would have started reporting a confident,
    understated number the moment extraction first worked (astro-mine-sim#64). Fixed in place
    rather than versioned, matching how ``water_mass``'s own channel defect was handled (#65):
    the metric's *meaning* never changed, only whether it measured it.
    """

    name: str = "energy_per_kg"
    version: str = "0.1.0"
    unit: str = "J/kg"
    direction: MetricDirection = MetricDirection.LOWER_BETTER
    aggregation: MetricAggregation = MetricAggregation.MEAN

    def compute(self, trace: EpisodeTrace) -> MetricValue:
        water = _total_stored_water(trace)
        by_agent: dict[str, list[tuple[int, float]]] = {}
        for obs in trace.observations:
            soc = obs.self_state.battery_soc_j
            if soc is not None:
                by_agent.setdefault(obs.agent_id, []).append((obs.tick, soc))
        discharges: list[float] = []
        for samples in by_agent.values():
            samples.sort()
            for (_, earlier), (_, later) in itertools.pairwise(samples):
                if later < earlier:
                    discharges.append(earlier - later)
        energy = math.fsum(discharges) + _total_extraction_energy(trace)
        if water <= 0.0:
            return MetricValue(value=None, unit=self.unit)  # not applicable: no water produced
        return MetricValue(value=energy / water, unit=self.unit)


@dataclass(frozen=True, slots=True)
class InformationGain:
    """Reduction in belief-field uncertainty over the campaign, in nats (higher is better).

    For each cell present in both the prior and the final posterior, the Gaussian entropy
    reduction ``0.5 * ln(var_prior / var_post)``; the metric is their sum, with the per-cell
    spread reported as its 1-sigma uncertainty. Not applicable without belief information."""

    name: str = "information_gain"
    version: str = "0.1.0"
    unit: str = "nat"
    direction: MetricDirection = MetricDirection.HIGHER_BETTER
    aggregation: MetricAggregation = MetricAggregation.MEAN

    def compute(self, trace: EpisodeTrace) -> MetricValue:
        ctx = trace.context
        if not ctx.prior_belief or not ctx.belief_history:
            return MetricValue(value=None, unit=self.unit)
        posterior = ctx.belief_history[-1].cells
        contributions: list[float] = []
        for cell, prior in ctx.prior_belief.items():
            post = posterior.get(cell)
            if post is None:
                continue
            if prior.variance <= 0.0 or post.variance <= 0.0:
                raise MetricComputationError(
                    f"information_gain requires positive belief variance for cell {cell!r}"
                )
            contributions.append(0.5 * math.log(prior.variance / post.variance))
        if not contributions:
            return MetricValue(value=None, unit=self.unit)
        spread = statistics.pstdev(contributions) if len(contributions) > 1 else 0.0
        return MetricValue(value=math.fsum(contributions), unit=self.unit, uncertainty=spread)


@dataclass(frozen=True, slots=True)
class PsrAreaCharacterized:
    """PSR area whose belief uncertainty was driven below threshold, in m² (higher is better).

    A PSR cell is characterized once its final posterior variance is at or below the scenario's
    ``characterized_variance_threshold``; the metric is the characterized cell count times the
    cell area. Not applicable without a PSR mask and belief history."""

    name: str = "psr_area_characterized"
    version: str = "0.1.0"
    unit: str = "m^2"
    direction: MetricDirection = MetricDirection.HIGHER_BETTER
    aggregation: MetricAggregation = MetricAggregation.MEAN

    def compute(self, trace: EpisodeTrace) -> MetricValue:
        ctx = trace.context
        if not ctx.psr_cells or not ctx.belief_history:
            return MetricValue(value=None, unit=self.unit)
        posterior = ctx.belief_history[-1].cells
        characterized = 0
        for cell in ctx.psr_cells:
            post = posterior.get(cell)
            if post is not None and post.variance <= ctx.characterized_variance_threshold:
                characterized += 1
        return MetricValue(value=characterized * ctx.cell_area_m2, unit=self.unit)


@dataclass(frozen=True, slots=True)
class NightsSurvived:
    """Number of lunar-night windows the fleet survived (higher is better).

    A night is survived if every agent kept ``battery_soc_j`` above zero — and, when a
    survivable temperature is set, ``temperature_k`` at or above it — at every sampled tick
    within the window. Aggregated by ``MIN`` across seeds: the guaranteed worst-case survival.
    Not applicable without night windows."""

    name: str = "nights_survived"
    version: str = "0.1.0"
    unit: str = "dimensionless"
    direction: MetricDirection = MetricDirection.HIGHER_BETTER
    aggregation: MetricAggregation = MetricAggregation.MIN

    def compute(self, trace: EpisodeTrace) -> MetricValue:
        ctx = trace.context
        if not ctx.night_intervals:
            return MetricValue(value=None, unit=self.unit)
        survived = 0
        for start, end in ctx.night_intervals:
            sampled = False
            alive = True
            for obs in trace.observations:
                if not start <= obs.sim_time_s <= end:
                    continue
                state = obs.self_state
                if state.battery_soc_j is None:
                    continue
                sampled = True
                depleted = state.battery_soc_j <= 0.0
                frozen = (
                    ctx.survivable_temperature_k is not None
                    and state.temperature_k is not None
                    and state.temperature_k < ctx.survivable_temperature_k
                )
                if depleted or frozen:
                    alive = False
                    break
            if sampled and alive:
                survived += 1
        return MetricValue(value=float(survived), unit=self.unit)


@dataclass(frozen=True, slots=True)
class CommsRobustness:
    """Fraction of ticks with Earth contact, in [0, 1] (higher is better).

    Not applicable when the trace carries no comms mask on any observation."""

    name: str = "comms_robustness"
    version: str = "0.1.0"
    unit: str = "dimensionless"
    direction: MetricDirection = MetricDirection.HIGHER_BETTER
    aggregation: MetricAggregation = MetricAggregation.MEAN

    def compute(self, trace: EpisodeTrace) -> MetricValue:
        total = 0
        connected = 0
        for obs in trace.observations:
            if obs.comms is None:
                continue
            total += 1
            if obs.comms.earth_contact:
                connected += 1
        if total == 0:
            return MetricValue(value=None, unit=self.unit)
        return MetricValue(value=connected / total, unit=self.unit)


@dataclass(frozen=True, slots=True)
class DiscoveryLatency:
    """Sim time to the first resource detection, in seconds (lower is better).

    The earliest ``sim_time_s`` at which a sensor reads the discovery species at or above the
    detection threshold. Aggregated by ``MEDIAN`` (robust to the censored "never discovered"
    seeds, which are not applicable rather than counted as zero)."""

    name: str = "discovery_latency"
    version: str = "0.1.0"
    unit: str = "s"
    direction: MetricDirection = MetricDirection.LOWER_BETTER
    aggregation: MetricAggregation = MetricAggregation.MEDIAN

    def compute(self, trace: EpisodeTrace) -> MetricValue:
        ctx = trace.context
        earliest: float | None = None
        for obs in trace.observations:
            detected = any(
                reading.resource_species == ctx.discovery_species
                and reading.valid
                and reading.values
                and max(reading.values) >= ctx.discovery_threshold
                for reading in obs.sensors
            )
            if detected and (earliest is None or obs.sim_time_s < earliest):
                earliest = obs.sim_time_s
        return MetricValue(value=earliest, unit=self.unit)


#: The Phase-0 reference metric set, in scenario-declaration order.
REFERENCE_METRICS: tuple[Metric, ...] = (
    WaterMass(),
    EnergyPerKg(),
    InformationGain(),
    PsrAreaCharacterized(),
    NightsSurvived(),
    CommsRobustness(),
    DiscoveryLatency(),
)
