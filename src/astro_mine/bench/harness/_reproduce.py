"""The reproducibility harness + determinism gate (RM-P0-BENCH-04; bench.md §10).

:func:`reproduce` runs a scenario ``runs`` times under a pinned seed set + lockfile and reports
whether every run produced the identical content-addressed :class:`Result`.
:func:`assert_reproducible` is the **gate**: it raises :class:`DeterminismError` on any drift, so CI
fails on non-reproducibility — Bench's determinism gate is the platform-wide reproducibility oracle
(bench.md §10; conventions.md §11). :func:`replay` re-executes a stored Result from its provenance
(sampled re-execution, bench.md §9).

Backlog: RM-P0-BENCH-04 — astro-mine-bench#4
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from astro_mine.bench._version import __version__
from astro_mine.bench.harness._models import ReproductionReport, Result, SeedResult
from astro_mine.bench.harness._runner import REFERENCE_RUNNER_ID, Runner, reference_runner
from astro_mine.bench.harness._toolchain import (
    build_toolchain,
    environment_stamp,
    lockfile_digest,
    resolve_lockfile,
)
from astro_mine.bench.scenario import ResolvedScenario, ScenarioSpec, resolve_scenario

__all__ = ["DeterminismError", "assert_reproducible", "replay", "reproduce"]


class DeterminismError(AssertionError):
    """Raised by the determinism gate when a scenario does not reproduce byte-for-byte."""


def _resolve_runner_id(runner: Runner, runner_id: str | None) -> str:
    if runner_id is not None:
        return runner_id
    if runner is reference_runner:
        return REFERENCE_RUNNER_ID
    return getattr(runner, "__name__", repr(runner))


def _aggregate(per_seed: Sequence[SeedResult]) -> dict[str, float]:
    """Mean of each metric across seeds (deterministic — same floats, same order)."""
    return {
        name: fmean(seed_result.metrics[name] for seed_result in per_seed)
        for name in per_seed[0].metrics
    }


def _run_once(
    resolved: ResolvedScenario, runner: Runner, seeds: Sequence[int], runner_id: str, lockfile: str
) -> Result:
    seed_results: list[SeedResult] = []
    for seed in seeds:
        outcome = runner(resolved, seed)
        seed_results.append(
            SeedResult(seed=seed, determinism_key=outcome.determinism_key, metrics=outcome.metrics)
        )
    per_seed = tuple(seed_results)
    return Result(
        scenario_id=resolved.scenario_id,
        scenario_spec_hash=resolved.spec_hash,
        scenario_hash=resolved.scenario_hash,
        core_interface_version=resolved.core_interface,
        core_schema_digest=resolved.core_schema_digest,
        content_hashes=resolved.content_hashes,
        runner=runner_id,
        code_version=__version__,
        environment_lockfile=lockfile,
        environment=environment_stamp(),
        per_seed=per_seed,
        aggregate=_aggregate(per_seed),
    )


def reproduce(
    spec: ScenarioSpec,
    runner: Runner = reference_runner,
    *,
    seeds: Sequence[int] | None = None,
    runs: int = 2,
    runner_id: str | None = None,
    lockfile: Path | str | None = None,
) -> ReproductionReport:
    """Run ``spec`` ``runs`` times under a pinned seed set + lockfile and compare determinism.

    Seeds default to the scenario's **public** seeds (held-out seeds stay embargoed until eval,
    bench.md §9). The pinned lockfile + runner identity fold into the resolved ``scenario_hash``.
    *lockfile* pins the environment the run executes in; it defaults to the benchmarked project's
    (:func:`resolve_lockfile`) and raises :class:`LockfileNotFound` when none can be resolved.
    """
    if runs < 2:
        raise ValueError("reproduce needs runs >= 2 to compare")
    run_seeds = tuple(seeds) if seeds is not None else spec.seeds.public
    rid = _resolve_runner_id(runner, runner_id)
    # Resolve once: the digest recorded on the Result and the digest folded into the scenario hash
    # must be of the same file, even if the working directory moves mid-run.
    resolved_lockfile = resolve_lockfile(lockfile)
    digest = lockfile_digest(resolved_lockfile)
    resolved = resolve_scenario(spec, toolchain=build_toolchain(rid, lockfile=resolved_lockfile))
    results = [_run_once(resolved, runner, run_seeds, rid, digest) for _ in range(runs)]
    hashes = tuple(result.result_hash for result in results)
    return ReproductionReport(
        scenario_id=spec.scenario_id,
        reproducible=len(set(hashes)) == 1,
        runs=runs,
        result_hash=hashes[0],
        result_hashes=hashes,
        result=results[0],
    )


def assert_reproducible(
    spec: ScenarioSpec,
    runner: Runner = reference_runner,
    *,
    seeds: Sequence[int] | None = None,
    runs: int = 2,
    runner_id: str | None = None,
    lockfile: Path | str | None = None,
) -> Result:
    """The determinism gate: run ``spec`` and raise :class:`DeterminismError` if it drifts.

    Returns the canonical :class:`Result` on success — the value a leaderboard entry would record.
    """
    report = reproduce(spec, runner, seeds=seeds, runs=runs, runner_id=runner_id, lockfile=lockfile)
    if not report.reproducible:
        raise DeterminismError(
            f"{report.scenario_id!r} did not reproduce across {report.runs} runs: "
            f"{report.result_hashes}"
        )
    return report.result


def replay(
    result: Result, runner: Runner = reference_runner, *, lockfile: Path | str | None = None
) -> bool:
    """Re-execute a stored Result from its pinned inputs and check the hash matches (bench.md §9).

    Reloads the (immutable) scenario from the zoo by id, re-runs its recorded seeds, and returns
    whether the freshly-computed :attr:`Result.result_hash` equals the stored one — the sampled
    re-execution integrity check. A mismatch means an engine/content/harness regression.
    """
    from astro_mine.bench.zoo import load_scenario

    seeds = tuple(seed_result.seed for seed_result in result.per_seed)
    report = reproduce(
        load_scenario(result.scenario_id),
        runner,
        seeds=seeds,
        runner_id=result.runner,
        lockfile=lockfile,
    )
    return report.reproducible and report.result_hash == result.result_hash
