"""The runner contract + a deterministic reference runner (RM-P0-BENCH-04).

A :class:`Runner` maps a resolved scenario + seed to a :class:`RunOutcome`. The harness is
runner-agnostic: Sim (via ``run_episode``) is the real runner once the ``ScenarioSpec -> Sim
Scenario`` bridge and Hub-published content land (Phase 1). Until then :func:`reference_runner` — a
pure, seeded function of the scenario hash — drives the determinism oracle over the anchor's own
metric set, so the reproducibility machinery is real and gated before a physics engine is wired.

Backlog: RM-P0-BENCH-04 — https://github.com/astro-mine/astro-mine-bench/issues/4
"""

from __future__ import annotations

from typing import Protocol

from astro_mine.bench.harness._models import RunOutcome
from astro_mine.bench.scenario import ResolvedScenario
from astro_mine.bench.scenario._hash import content_hash

__all__ = ["REFERENCE_RUNNER_ID", "Runner", "reference_runner"]

#: The identity recorded on a Result for a reference-runner run.
REFERENCE_RUNNER_ID = "reference-runner/0.1.0"

_FRACTION_BITS = 48


class Runner(Protocol):
    """Maps a resolved scenario + seed to a deterministic :class:`RunOutcome`.

    A conforming runner MUST be a pure function of ``(resolved.scenario_hash, seed)`` — no
    wall-clock, no global RNG, no network — so same inputs return byte-identical output.
    """

    def __call__(self, resolved: ResolvedScenario, seed: int) -> RunOutcome: ...


def _unit_score(scenario_hash: str, seed: int, metric: str) -> float:
    """A deterministic pseudo-score in ``[0, 1)`` from the scenario hash, seed, and metric name."""
    digest = content_hash({"metric": metric, "scenario_hash": scenario_hash, "seed": seed})
    hex_start = len("sha256:")
    bucket = int(digest[hex_start : hex_start + _FRACTION_BITS // 4], 16)
    return bucket / float(1 << _FRACTION_BITS)


def reference_runner(resolved: ResolvedScenario, seed: int) -> RunOutcome:
    """A deterministic stand-in runner: synthesize per-metric scores from the scenario hash + seed.

    Produces one scalar per metric named in the resolved scenario's spec, plus a ``determinism_key``
    over the whole synthetic trace — enough to exercise the reproducibility oracle without a physics
    engine. Replaced by a Sim adapter once the content plumbing lands (see the module docstring).
    """
    metrics = {
        metric.name: _unit_score(resolved.scenario_hash, seed, metric.name)
        for metric in resolved.spec.metrics
    }
    determinism_key = content_hash(
        {"metrics": metrics, "scenario_hash": resolved.scenario_hash, "seed": seed}
    )
    return RunOutcome(determinism_key=determinism_key, metrics=metrics)
