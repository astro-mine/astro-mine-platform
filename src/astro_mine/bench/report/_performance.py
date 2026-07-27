"""The performance report — a measured speedup, published *beside* the Scorecard, never inside it.

A surrogate physics tier earns its place by being **faster than the solver it replaces, at a stated
error bound**. That number — DEM wall-clock / surrogate wall-clock, at the bound the substitution
actually held to — is the deliverable of ``surrogate.md`` §8/§12 (RM-P1-SURR-04 / RM-P1-SIM-03) and
of ``LUNAR-TR-002``. This module is where Bench publishes it.

## Why it is not a Metric, and not a Scorecard field

Both of the obvious homes are closed, and for the same underlying reason: **Bench's determinism
contract deliberately has no room for wall-clock.**

- **Not a Metric.** ``Metric.compute(trace)`` sees exactly one :class:`EpisodeTrace` and must be a
  *pure function* of it (``metrics/_metric.py``). A speedup is a ratio of two durations from two
  runs at two fidelities. There is no second trace to compare against and no timing channel to read
  — an ``EpisodeTrace`` is observations plus a :class:`ScoringContext`, and a conforming
  ``EpisodeRunner`` is forbidden from putting wall-clock into either (``baseline/_runner.py``).
- **Not a Scorecard field.** :attr:`Scorecard.content_hash` is taken over the *whole*
  ``model_dump()`` (``metrics/_score.py``). A duration in there would change every run, so
  :func:`~astro_mine.bench.baseline.assert_score_reproducible` and the determinism gate would fail
  **unconditionally** — trading the platform's reproducibility guarantee (bench.md §§69-72, §207;
  ``LUNAR-TR-006``; ``LUNAR-DR-004``) for a performance number. That is the wrong trade, and it is
  not a close call.

So the speedup rides in a **separate, non-hashed sidecar**. This is not an invention: Sim reached
the identical fork and made the identical choice — its ``Trace`` keeps timing on a field
``to_canonical_json`` does not serialize, and its MCAP envelope carries timing as a *sibling* of
``content_hash``, never inside it (``sim/runtime/timing.py``). The rule both sides are enforcing is
one rule: *measuring the run from outside is fine; letting the measurement back into the run is
not.*

The consequence to keep hold of: **a scorecard is bit-identical whether or not a performance report
was produced alongside it.** That is a property this package tests, not merely asserts.

## The shapes are Bench's own

:class:`PerformanceReport` is built from **plain dicts** — the JSON-able form Sim's
``SpeedupReport.as_provenance()`` emits — never from a Sim type. Bench stays ``core`` + ``pydantic``
and never imports Sim; the measurement reaches it through the injected ``EpisodeRunner`` seam and
lands here as data. That is also what lets these models be tested from fixtures with no Sim, no
onnxruntime, and no DEM solver in the room.

## What makes a number a claim

:attr:`SeedSpeedup.is_claim` is the whole discipline of this module. Speed alone is not a result: a
surrogate that was never admitted, or that breached its tolerance, or that escalated back to the
reference solver mid-run, produces a ratio that is partly *the reference solver against itself* — a
number with the shape of a result and none of the content. So the headline aggregates **only over
claiming seeds** (:attr:`PerformanceReport.headline_speedup`), and a report with no claiming seed
reports **no headline at all** rather than an average of noise.
"""

from __future__ import annotations

import os
import platform
import statistics
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PerformanceReport",
    "SeedSpeedup",
    "SurrogateIdentity",
    "TierTiming",
    "ToolchainStamp",
    "performance_report",
    "toolchain_stamp",
]


class _Model(BaseModel):
    """Frozen, extra-forbidding base — a typo'd key from a producer must fail loudly, not vanish."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TierTiming(_Model):
    """One fidelity tier's measured wall-clock over an episode, and the engine that spent it.

    Mirrors Sim's ``EngineTiming.as_provenance()``. ``advance_wall_clock_s`` brackets **only** the
    engine's ``advance`` — not scenario resolution, not content pulls, not world construction — so
    the ratio of two of these is a comparison of *physics cost*, which is the only thing a surrogate
    claims to change.
    """

    engine: str
    #: The engine's *declared* fidelity tier. This is what makes the pair a per-tier measurement:
    #: it is read off the engine that really ran, so a surrogate that was silently never admitted
    #: reports ``tier`` = the reference tier and the comparison is visibly DEM-vs-DEM.
    tier: str
    steps: int = Field(ge=0)
    advance_wall_clock_s: float = Field(ge=0.0)
    sim_time_s: float = Field(ge=0.0)
    #: Modeled seconds per wall-clock second. Above 1.0 the run is faster than real time.
    real_time_factor: float | None = None
    mean_step_wall_clock_s: float | None = None


class SeedSpeedup(_Model):
    """One seed's DEM-vs-surrogate result: what it cost, and whether it stayed honest.

    Mirrors Sim's ``SpeedupReport.as_provenance()``. The three bounds are deliberately kept apart
    rather than collapsed into a single "error" number, because they answer different questions:

    - :attr:`declared_error_budget` — what the surrogate *artifact* advertises it holds to (its
      calibrated ``ErrorReport``). A property of the tier, fixed before this run existed.
    - :attr:`admitted_tolerance` — what the **task** required, and what the engine actually
      re-validated against. ``LUNAR-TR-002``'s operative bound: Sim must refuse substitution beyond
      *task* tolerance, and a task may legitimately tolerate more (or less) than the artifact
      advertises.
    - :attr:`realized_error` — what the surrogate actually deviated by, worst-case per channel,
      measured against a DEM reference bed stepped from the same state.

    A tier can pass the task while overshooting its own published bound
    (:attr:`holds_declared_budget` ``False``): that is a fact about the *artifact* overselling
    itself, not about this run, and averaging it away would hide it.
    """

    seed: int
    #: Whether the scheduler admitted the surrogate tier at all. ``False`` means the run was DEM
    #: twice over and there is no speedup here to report.
    admitted: bool
    #: DEM wall-clock / surrogate wall-clock. ``None`` when either tier was unmeasurable.
    speedup: float | None = None
    #: Whether every re-validated deviation stayed inside the tolerance the run was admitted under.
    within_budget: bool
    #: Whether the realized deviation also stayed inside the budget the *artifact* advertised.
    holds_declared_budget: bool
    #: Whether the surrogate fell back to DEM mid-run — on an out-of-trust-region query or a
    #: breached tolerance. Either way part of this run's wall-clock was spent on the reference
    #: solver, so the ratio understates the surrogate *and* the substitution did not hold.
    escalated: bool
    #: Whether this seed supports a speedup claim: admitted, within budget, and never escalated.
    is_claim: bool
    dem: TierTiming
    surrogate: TierTiming
    declared_error_budget: dict[str, float] = Field(default_factory=dict)
    admitted_tolerance: dict[str, float] = Field(default_factory=dict)
    realized_error: dict[str, float] = Field(default_factory=dict)


class SurrogateIdentity(_Model):
    """The tier under test, pinned by content address — *which* surrogate this number is about.

    A performance claim about "a surrogate" is not a claim about anything. The digest is what makes
    the number attributable to a specific, immutable, signed artifact that a reader can pull and
    re-measure.
    """

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    #: The ``sha256:`` content address of the served bundle the speedup was measured against.
    content_hash: str | None = None
    #: The digest of the tier's calibrated ``ErrorReport`` — the bound it commits to, by hash.
    error_report_digest: str | None = None
    #: The ``sha256:`` content address of the sampling policy whose box *is* the tier's trust
    #: region. Recorded because a surrogate's declared domain is only as trustworthy as the
    #: declaration that produced it (astro-mine-surrogate#17).
    sampling_policy_hash: str | None = None


class ToolchainStamp(_Model):
    """The host the measurement ran on. **A performance claim without one is not reproducible.**

    A speedup is a ratio, so it cancels most host effects — but not all: the two tiers stress
    different code (a numpy O(N^2) DEM kernel vs. an ONNX Runtime session), and they do not scale
    alike with core count, SIMD width, or thread pinning. The ratio is a measurement *of this
    machine*, and a reader who cannot see the machine cannot judge the number or reproduce it.
    """

    python: str
    platform: str
    processor: str | None = None
    cpu_count: int | None = Field(default=None, ge=1)
    #: Free-form component versions (``{"astro-mine-sim": "0.1.dev26", ...}``) — the code that
    #: produced the physics on both sides of the ratio.
    packages: dict[str, str] = Field(default_factory=dict)


class PerformanceReport(_Model):
    """A scenario's measured DEM-vs-surrogate speedup — published beside its Scorecard.

    **Never folded into** :attr:`Scorecard.content_hash`: see the module docstring. This is the
    sidecar, and it is the whole point of the ``lunar-polar-ice-excavation-fidelity`` task — that
    scenario's headline result is *this*, not its scorecard.
    """

    scenario_id: str = Field(min_length=1)
    #: The ``sha256:`` content address of the ScenarioSpec measured — the task identity. Ties the
    #: number to an immutable task, so "2.9x" is a statement about something in particular.
    spec_hash: str
    #: The ``EpisodeRunner`` that produced the pair of runs (Sim's ``__name__``).
    runner: str = Field(min_length=1)
    surrogate: SurrogateIdentity
    toolchain: ToolchainStamp
    #: One row per seed measured, in seed order.
    seeds: tuple[SeedSpeedup, ...] = Field(min_length=1)

    @property
    def claiming_seeds(self) -> tuple[SeedSpeedup, ...]:
        """The seeds whose ratio is a *claim* — admitted, within budget, never escalated."""
        return tuple(s for s in self.seeds if s.is_claim and s.speedup is not None)

    @property
    def headline_speedup(self) -> float | None:
        """The published number: the **median** speedup over claiming seeds only.

        ``None`` when no seed supports a claim — a report that measured nothing publishes nothing,
        rather than an average of ratios that are partly the reference solver against itself.

        Median, not mean: a wall-clock ratio is a bounded-below quantity on a noisy shared host, and
        one descheduled surrogate run skews a mean upward in the direction that flatters us.
        """
        ratios = [s.speedup for s in self.claiming_seeds if s.speedup is not None]
        return statistics.median(ratios) if ratios else None

    @property
    def is_claim(self) -> bool:
        """Whether this report supports a published speedup claim at all."""
        return self.headline_speedup is not None

    @property
    def worst_realized_error(self) -> dict[str, float]:
        """The worst per-channel deviation across the claiming seeds — the error half of the claim.

        A speedup is only meaningful *at a stated error*, so this travels with
        :attr:`headline_speedup` and is reported next to the bound it must be read against.
        """
        worst: dict[str, float] = {}
        for seed in self.claiming_seeds:
            for channel, value in seed.realized_error.items():
                if value > worst.get(channel, float("-inf")):
                    worst[channel] = value
        return worst


def toolchain_stamp(packages: Mapping[str, str] | None = None) -> ToolchainStamp:
    """Stamp the host this process is running on (see :class:`ToolchainStamp` for why)."""
    return ToolchainStamp(
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor() or None,
        cpu_count=os.cpu_count(),
        packages=dict(packages or {}),
    )


def performance_report(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    scenario_id: str,
    spec_hash: str,
    runner: str,
    surrogate: SurrogateIdentity,
    toolchain: ToolchainStamp | None = None,
    packages: Mapping[str, str] | None = None,
) -> PerformanceReport:
    """Validate a producer's per-seed speedup dicts into a :class:`PerformanceReport`.

    ``reports`` maps seed → the **plain JSON-able dict** a producer emits (Sim's
    ``FidelitySpeedupRunner.reports[seed].as_provenance()``). Taking dicts rather than a Sim type is
    what keeps Bench ``core`` + ``pydantic``: no Sim class crosses this boundary, so these models
    are driven from fixtures in Bench's own tests with Sim nowhere in the environment.

    Validation is strict (``extra="forbid"``): if a producer's wire shape drifts, this raises here —
    at the boundary — rather than silently dropping the field that carried the claim.
    """
    rows = tuple(SeedSpeedup.model_validate(dict(reports[seed])) for seed in sorted(reports))
    return PerformanceReport(
        scenario_id=scenario_id,
        spec_hash=spec_hash,
        runner=runner,
        surrogate=surrogate,
        toolchain=toolchain if toolchain is not None else toolchain_stamp(packages),
        seeds=rows,
    )
