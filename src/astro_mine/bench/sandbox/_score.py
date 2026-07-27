"""The scoring seam the leaderboard evaluates submissions through (bench#30; bench.md §9).

A :class:`PolicyScorer` maps *(scenario, a submitted ``policy_ref``, seeds)* to a
:class:`~astro_mine.bench.metrics.Scorecard`. It is deliberately typed on the **policy reference —
a string — not a Core ``Policy`` object**: an evaluator that accepts a ``Policy`` has already
imported and constructed untrusted code in its own process, which is exactly the posture bench.md
§9 forbids. The reference is resolved (imported) only *inside* the sandbox.

:class:`SandboxScorer` is the leaderboard's scorer: it fans the seeds out one sandboxed
:class:`~astro_mine.bench.sandbox.WorkerInvocation` at a time and folds the per-seed values the
workers hand back through :func:`~astro_mine.bench.metrics.aggregate_scores` — the *same* kernel the
in-process local tier and the Cloud collector use. So a sandboxed scorecard is **byte-identical** to
the workstation scorecard for the same inputs and seeds, and the leaderboard loses no
reproducibility by gaining the sandbox. It **fails closed**: a seed that times out, is killed by a
limit, crashes, or reports an error rejects the whole submission — a partially-executed policy is
never scored on the seeds that happened to finish.

:class:`InProcessScorer` is the trusted-only counterpart, for code *you wrote*: the local scoring
tier (``run(spec, policy)``) and the determinism gate. It is never the leaderboard's default, and
wiring it into a hosted service re-opens the very hole bench#30 closed.

Backlog: bench#30 — https://github.com/astro-mine/astro-mine-bench/issues/30
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from astro_mine.bench.baseline import REFERENCE_EPISODE_RUNNER_ID
from astro_mine.bench.metrics import Scorecard, aggregate_scores, resolve_metrics
from astro_mine.bench.sandbox._channel import SandboxOutcome, WorkerInvocation
from astro_mine.bench.sandbox._subprocess import Sandbox, SandboxError
from astro_mine.bench.scenario import ScenarioSpec

__all__ = ["InProcessScorer", "PolicyScorer", "SandboxScorer", "SubmissionExecutionError"]


class SubmissionExecutionError(SandboxError):
    """A submitted policy did not execute cleanly under its envelope — the submission is rejected.

    Carries the offending :class:`~astro_mine.bench.sandbox.SandboxOutcome` so the service can put
    the exact failure (timeout, memory kill, import error, seccomp denial) into the audit log and
    the job's terminal ``rejected`` detail, rather than a generic failure.
    """

    def __init__(self, message: str, *, outcome: SandboxOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


class PolicyScorer(Protocol):
    """Scores a **policy reference** (never a live Policy object) on a scenario's seeds."""

    def __call__(
        self, spec: ScenarioSpec, policy_ref: str, *, seeds: Sequence[int]
    ) -> Scorecard: ...


class SandboxScorer:
    """Score a submitted ``policy_ref`` by rolling every seed in a sandbox, then aggregating.

    The leaderboard's scorer for **both** intake paths — the local ``policy_ref`` submission and the
    Hub-digest one. Neither imports the submission into the evaluator: both hand a reference to the
    sandboxed eval worker and read a result document back.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    @property
    def sandbox(self) -> Sandbox:
        """The backend seeds are executed in (subprocess/seccomp, or a container)."""
        return self._sandbox

    def __call__(self, spec: ScenarioSpec, policy_ref: str, *, seeds: Sequence[int]) -> Scorecard:
        """Roll each seed out-of-process and aggregate; raise if any seed did not score cleanly."""
        ordered = tuple(sorted(seeds))
        if not ordered:
            raise ValueError("a submission needs at least one seed to score")
        metrics = resolve_metrics(spec.metrics)
        per_seed_by_metric: dict[str, list[float | None]] = {metric.name: [] for metric in metrics}

        for seed in ordered:
            outcome = self._sandbox.run(WorkerInvocation(spec.scenario_id, policy_ref, seed))
            if not outcome.scored or outcome.result is None:
                raise SubmissionExecutionError(
                    f"submission {policy_ref!r} did not execute cleanly on seed {seed} "
                    f"({outcome.status}: {outcome.detail})",
                    outcome=outcome,
                )
            reported = {metric.metric: metric.value for metric in outcome.result.metrics}
            for metric in metrics:
                if metric.name not in reported:
                    raise SubmissionExecutionError(
                        f"submission {policy_ref!r} returned no value for metric "
                        f"{metric.name!r} on seed {seed}",
                        outcome=outcome,
                    )
                per_seed_by_metric[metric.name].append(reported[metric.name])

        # The sandboxed worker runs the reference runner in this Bench env; a Sim-image worker's
        # runner id is not carried back through WorkerResult yet (a follow-up to G1.1). Stamp the
        # reference id explicitly rather than defaulting silently.
        return aggregate_scores(
            metrics,
            per_seed_by_metric,
            ordered,
            scenario_id=spec.scenario_id,
            runner=REFERENCE_EPISODE_RUNNER_ID,
        )


class InProcessScorer:
    """**Trusted code only**: import the policy and score it in this process (no sandbox).

    The scorer for the local tier and the determinism gate — where the policy is one *you* wrote and
    already trust, and where paying for a process per seed would break the "score a baseline in an
    afternoon" promise (charter §13; CX-LOCAL). Wiring it into a hosted
    :class:`~astro_mine.bench.leaderboard.LeaderboardService` would execute community code inside
    the evaluator, which bench.md §9 forbids — the service never selects it.
    """

    def __call__(self, spec: ScenarioSpec, policy_ref: str, *, seeds: Sequence[int]) -> Scorecard:
        from astro_mine.bench.baseline import run
        from astro_mine.bench.leaderboard._eval import resolve_policy

        return run(spec, resolve_policy(policy_ref), seeds=tuple(sorted(seeds)))
