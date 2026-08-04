"""Submit-policy-we-run evaluation + integrity (RM-P0-BENCH-06; bench.md §9).

The leaderboard's trust model: rather than believe reported numbers, Bench **runs the policy
itself** on the scenario's embargoed **held-out seeds** and re-executes a sample to verify the
result reproduces (bench.md §9). This module is the dependency-clean core of that — no FastAPI,
no database — so it is fully testable on its own:

- :func:`validate_policy_ref` checks a submitted reference's **shape** without importing it;
- :func:`resolve_policy` imports a ``"module:attribute"`` policy reference — **inside the sandbox**;
- :func:`load_heldout_seeds` discloses the sealed held-out seeds at evaluation time;
- :func:`evaluate` scores on the held-out seeds and re-runs a sample for the integrity verdict;
- :func:`build_submission` binds the run to a content-addressed :class:`Submission`;
- :func:`rank` orders submissions by a scenario's primary metric.

Since bench#30, :func:`evaluate` takes a **policy reference and a**
:class:`~astro_mine.bench.sandbox.PolicyScorer`, not a live ``Policy``: the evaluator must never
import a submission into its own process (bench.md §9), so the string crosses into a sandboxed
worker and only a scorecard comes back. :func:`resolve_policy` still exists — it is what the worker
calls, *inside* the sandbox.

Backlog: RM-P0-BENCH-06 — https://github.com/astro-mine/astro-mine-bench/issues/6
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from astro_mine.bench.leaderboard._models import (
    Integrity,
    LeaderboardEntry,
    MetricScore,
    Submission,
    SubmissionRequest,
)
from astro_mine.bench.metrics import Scorecard
from astro_mine.bench.sandbox import PolicyScorer
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.scenario._hash import content_hash
from astro_mine.core.policy import Policy

__all__ = [
    "EMBARGO_ROOT",
    "EMBARGO_ROOT_ENV",
    "PolicyReferenceError",
    "build_submission",
    "evaluate",
    "load_heldout_seeds",
    "rank",
    "resolve_embargo_root",
    "resolve_policy",
    "validate_policy_ref",
]

#: The embargoed held-out seed sets, at the repo root above ``src/`` (excluded from the wheel;
#: the leaderboard runs from the repo and discloses them at evaluation time — bench.md §9).
#:
#: Correct when the leaderboard and the seeds share a checkout, and **wrong for every deployment**,
#: which is what :data:`EMBARGO_ROOT_ENV` exists to fix.
EMBARGO_ROOT = Path(__file__).resolve().parents[4] / "embargo"

#: Env var naming the directory the sealed seed sets live in; unset ⇒ :data:`EMBARGO_ROOT`.
#:
#: **A hosted leaderboard could not score a single submission without this** (#15). The path above
#: is derived from this module's own location, so on an installed wheel it points inside
#: ``site-packages`` and finds nothing — while ``embargo/`` ships with ``astro-mine-api``, the
#: repository the hosted leaderboard actually runs from. Every submission answered 404 "no held-out
#: seed set", and nothing caught it because the three places that score in-process each rebound the
#: keyword default first; a *served* process had no way to.
#:
#: Read per call rather than bound at import, which is the whole of why an override is possible:
#: ``load_heldout_seeds`` used to carry :data:`EMBARGO_ROOT` as a keyword default, so rebinding the
#: module attribute reached nobody and the caller in ``_service`` passes no argument.
EMBARGO_ROOT_ENV = "ASTRO_MINE_BENCH_EMBARGO_ROOT"


def resolve_embargo_root() -> Path:
    """Where the sealed held-out seed sets live: ``$ASTRO_MINE_BENCH_EMBARGO_ROOT``, else the
    repo-relative default. Resolved on every call, so a deployment can set it after import."""
    configured = os.environ.get(EMBARGO_ROOT_ENV)
    return Path(configured).expanduser() if configured else EMBARGO_ROOT


class PolicyReferenceError(ValueError):
    """Raised when a submitted ``policy_ref`` cannot be resolved to a Core :class:`Policy`."""


def validate_policy_ref(policy_ref: str) -> str:
    """Check a submitted reference's **shape** — without importing it (bench#30).

    The public edge must reject an obviously malformed ``policy_ref`` with a 400, but it must not
    *import* one to find out whether it works: importing is executing, and the submission is
    untrusted (bench.md §9). So the edge checks the ``module:attribute`` shape here, and the
    sandboxed worker discovers an unimportable reference and hands the failure back as data.

    Raises :class:`PolicyReferenceError` on a reference that is not ``module:attribute``.
    """
    module_name, separator, attribute = policy_ref.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise PolicyReferenceError(f"policy_ref must be 'module:attribute', got {policy_ref!r}")
    return policy_ref


def resolve_policy(policy_ref: str) -> Policy:
    """Import a ``"module:attribute"`` reference and resolve it to a Core :class:`Policy`.

    The attribute may be a Policy instance, a Policy class, or a zero-arg factory. Raises
    :class:`PolicyReferenceError` on a malformed reference, an import/attribute failure, or an
    object that is not a Policy (does not expose ``decide``).

    .. warning::
       This **imports and instantiates** the referenced code. It is called by the eval worker
       *inside* the sandbox (and by the trusted local tier for a policy you wrote), never by the
       leaderboard evaluator on a submission (bench#30).
    """
    if ":" not in policy_ref:
        raise PolicyReferenceError(f"policy_ref must be 'module:attribute', got {policy_ref!r}")
    module_name, _, attribute = policy_ref.partition(":")
    try:
        obj = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise PolicyReferenceError(f"cannot import policy_ref {policy_ref!r}: {exc}") from exc
    candidate = obj() if isinstance(obj, type) or not hasattr(obj, "decide") else obj
    if not (hasattr(candidate, "decide") and callable(candidate.decide)):
        raise PolicyReferenceError(f"policy_ref {policy_ref!r} is not a Policy (no decide method)")
    return cast(Policy, candidate)


def load_heldout_seeds(scenario_id: str, *, embargo_root: Path | None = None) -> tuple[int, ...]:
    """Disclose the embargoed held-out seed set for ``scenario_id`` (bench.md §9).

    ``embargo_root`` defaults to :func:`resolve_embargo_root` — the environment, else the
    repo-relative path. **Defaulting to ``None`` rather than to the path is the fix rather than a
    style preference**: a keyword default is evaluated once at import, so the old signature made
    the location unconfigurable by any deployment that had already imported this module, which is
    every one of them.

    Raises :class:`FileNotFoundError` if no sealed set is present — running from an installed wheel
    with no ``$ASTRO_MINE_BENCH_EMBARGO_ROOT`` set is the way to meet that (#15).
    """
    root = embargo_root if embargo_root is not None else resolve_embargo_root()
    path = root / scenario_id / "heldout_seeds.json"
    if not path.is_file():
        raise FileNotFoundError(f"no held-out seed set for {scenario_id!r} at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(int(seed) for seed in payload["seeds"])


def _sample_reproduces(card: Scorecard, resample: Scorecard) -> bool:
    """Whether every re-executed seed's per-metric value matches the full run (bench.md §9)."""
    by_metric = {m.metric: dict(zip(m.seeds, m.per_seed, strict=True)) for m in card.metrics}
    for metric in resample.metrics:
        original = by_metric.get(metric.metric)
        if original is None:
            return False
        for seed, value in zip(metric.seeds, metric.per_seed, strict=True):
            if original.get(seed) != value:
                return False
    return True


def evaluate(
    spec: ScenarioSpec,
    policy_ref: str,
    *,
    seeds: Sequence[int],
    scorer: PolicyScorer,
    sample_size: int = 1,
) -> tuple[Scorecard, Integrity]:
    """Score ``policy_ref`` on the held-out ``seeds`` and re-run a sample for the integrity verdict.

    ``scorer`` is the execution seam (bench#30): the leaderboard passes a
    :class:`~astro_mine.bench.sandbox.SandboxScorer`, so the submitted reference is imported and run
    **out-of-process, under a no-egress envelope**, and only a :class:`Scorecard` comes back.

    Returns the held-out :class:`Scorecard` and ``"verified"`` when a re-executed sample of the
    seeds reproduces its per-seed values, else ``"flagged"`` (non-determinism or tampering —
    bench.md §9). ``sample_size`` re-runs are the integrity baseline; full re-execution is the
    audit tool, not the steady state (bench.md §11).
    """
    card = scorer(spec, policy_ref, seeds=seeds)
    sample = tuple(sorted(seeds))[: max(1, sample_size)]
    resample = scorer(spec, policy_ref, seeds=sample)
    integrity: Integrity = "verified" if _sample_reproduces(card, resample) else "flagged"
    return card, integrity


def build_submission(
    request: SubmissionRequest,
    card: Scorecard,
    integrity: Integrity,
    *,
    source: str | None = None,
    provenance_hash: str | None = None,
) -> Submission:
    """Bind a scored run to a content-addressed :class:`Submission` record.

    ``source`` (a Hub image-manifest digest for a community submission) and ``provenance_hash`` (the
    stored :class:`ProvenanceBundle` digest) record a hosted submission's lineage (RM-P1-BENCH-10);
    both default to ``None`` for a local ``policy_ref`` submission. When ``source`` is present it is
    folded into the content address, so two artifacts with different Hub identities never collide.
    """
    scores = tuple(
        MetricScore(
            metric=aggregate.metric,
            unit=aggregate.unit,
            direction=aggregate.direction.value,
            aggregation=aggregate.aggregation.value,
            value=aggregate.value,
            dispersion=aggregate.dispersion,
            n=aggregate.n,
        )
        for aggregate in card.metrics
    )
    submission_id = content_hash(
        {
            "policy_ref": request.policy_ref,
            "scenario_id": request.scenario_id,
            "scorecard": card.content_hash,
            "source": source,
        }
    )
    return Submission(
        submission_id=submission_id,
        scenario_id=request.scenario_id,
        policy_ref=request.policy_ref,
        method=request.method,
        author=request.author,
        scorecard_hash=card.content_hash,
        runner=card.runner,
        integrity=integrity,
        scores=scores,
        source=source,
        provenance_hash=provenance_hash,
    )


def rank(submissions: Sequence[Submission]) -> list[LeaderboardEntry]:
    """Order ``submissions`` by the scenario's **primary metric** (its first scored metric).

    Best-first honouring the metric's direction (higher/lower-better); not-applicable values sort
    last; ties break by ``submission_id`` for a deterministic order. Multi-objective / Pareto
    ranking is a pluggable strategy deferred to Phase 1 (bench.md §11).
    """
    if not submissions:
        return []
    primary = submissions[0].scores[0]
    metric_name, higher_better = primary.metric, primary.direction == "higher_better"

    def _primary(submission: Submission) -> MetricScore | None:
        return next((s for s in submission.scores if s.metric == metric_name), None)

    def _key(submission: Submission) -> tuple[bool, float, str]:
        score = _primary(submission)
        value = score.value if score is not None else None
        if value is None:
            return (True, 0.0, submission.submission_id)
        return (False, -value if higher_better else value, submission.submission_id)

    entries: list[LeaderboardEntry] = []
    for position, submission in enumerate(sorted(submissions, key=_key), start=1):
        score = _primary(submission)
        entries.append(
            LeaderboardEntry(
                rank=position,
                submission_id=submission.submission_id,
                method=submission.method,
                author=submission.author,
                integrity=submission.integrity,
                primary_metric=metric_name,
                primary_value=score.value if score is not None else None,
                primary_unit=score.unit if score is not None else "",
                source=submission.source,
                provenance_hash=submission.provenance_hash,
            )
        )
    return entries
