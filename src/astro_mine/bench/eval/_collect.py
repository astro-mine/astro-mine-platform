"""Result collection — pull MCAP + Parquet back from Cloud and ingest (RM-P1-BENCH-11).

The other side of the fan-out: given the Cloud
:class:`~astro_mine.cloud.submission.result.RunResult` of each seed, read the per-seed
``metrics.parquet`` (the columnar result; bench.md §5) and ``trace.mcap`` (the raw trace) back from
the content-addressed artifact store, aggregate the per-seed values into the *same*
content-addressed :class:`~astro_mine.bench.metrics.Scorecard` the local scoring path produces
(:func:`~astro_mine.bench.metrics.aggregate_scores`), and bind it to a leaderboard
:class:`~astro_mine.bench.leaderboard._models.Submission`
(:func:`~astro_mine.bench.leaderboard.build_submission`, RM-P0-BENCH-06). Because the aggregation
kernel is shared with :func:`~astro_mine.bench.metrics.score`, a cluster-collected scorecard is
**byte-identical** to the workstation scorecard for the same inputs + seeds — the reproducibility
gate (bench.md §7).

Integrity: a seed whose rollout failed, dropped an output, whose raw ``trace.mcap`` provenance seed
disagrees with its ``metrics.parquet`` seed, or **whose artifact does not name the runner that
produced it** flags the submission (bench.md §9); a batch whose seeds were rolled by *different*
runners is refused outright, because those seeds are not one result (bench#64). ``pyarrow``
(the ``[cloud]`` extra) and the ``mcap`` reader (behind :mod:`~astro_mine.bench.recording`, the
``[recording]`` extra) are imported lazily; Bench never imports Sim.

Backlog: RM-P1-BENCH-11 — https://github.com/astro-mine/astro-mine-bench/issues/19
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from astro_mine.bench.eval._plan import METRICS_OUTPUT, TRACE_OUTPUT
from astro_mine.bench.leaderboard._eval import build_submission
from astro_mine.bench.metrics import Metric, aggregate_scores, resolve_metrics

if TYPE_CHECKING:
    from astro_mine.bench.leaderboard._models import Integrity, Submission, SubmissionRequest
    from astro_mine.bench.scenario import ScenarioSpec
    from astro_mine.cloud.submission.result import RunResult
    from astro_mine.core.artifacts import ArtifactStore

__all__ = ["collect_submission", "read_metrics_parquet"]


def read_metrics_parquet(data: bytes) -> list[dict[str, object]]:
    """Decode a worker's ``metrics.parquet`` bytes to per-metric rows (lazily importing pyarrow)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(pa.BufferReader(data))
    rows: list[dict[str, object]] = table.to_pylist()
    return rows


def _trace_seed(data: bytes) -> int | None:
    """Decode a worker's ``trace.mcap`` and return its provenance seed via the recording reader."""
    from astro_mine.bench.recording import decode_recording

    with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)
    try:
        return decode_recording(path).seed
    finally:
        path.unlink(missing_ok=True)


def collect_submission(
    spec: ScenarioSpec,
    request: SubmissionRequest,
    run_results: Sequence[RunResult],
    *,
    artifact_store: ArtifactStore,
    metrics: Sequence[Metric] | None = None,
    source: str | None = None,
    provenance_hash: str | None = None,
) -> Submission:
    """Aggregate one submission's per-seed Cloud results into a leaderboard :class:`Submission`.

    Reads each seed's ``metrics.parquet`` (and cross-checks its ``trace.mcap`` provenance seed) from
    ``artifact_store`` by content address, aggregates the values into a content-addressed Scorecard,
    and binds it via :func:`~astro_mine.bench.leaderboard.build_submission`. A failed/inconsistent
    seed flags the submission's integrity. Raises ``ValueError`` if no seed produced a scorable
    result.
    """
    metric_set = tuple(metrics) if metrics is not None else resolve_metrics(spec.metrics)
    per_seed_by_metric: dict[str, dict[int, float | None]] = {m.name: {} for m in metric_set}
    seeds_seen: set[int] = set()
    runners_seen: set[str] = set()
    integrity: Integrity = "verified"

    for result in run_results:
        if not result.ok or METRICS_OUTPUT not in result.outputs:
            integrity = "flagged"
            continue
        rows = read_metrics_parquet(artifact_store.get(result.outputs[METRICS_OUTPUT]))
        parquet_seed = rows[0]["seed"] if rows else None
        trace_seed = (
            _trace_seed(artifact_store.get(result.outputs[TRACE_OUTPUT]))
            if TRACE_OUTPUT in result.outputs
            else None
        )
        if not isinstance(parquet_seed, int) or (
            trace_seed is not None and trace_seed != parquet_seed
        ):
            integrity = "flagged"
            continue
        seed_runner = rows[0].get("runner") if rows else None
        if not isinstance(seed_runner, str) or not seed_runner:
            # Fail closed: a seed whose artifact does not name its runner cannot be attributed, and
            # a scorecard that guesses is worse than one that refuses (bench#64, G1.8).
            integrity = "flagged"
            continue
        runners_seen.add(seed_runner)
        seeds_seen.add(parquet_seed)
        for row in rows:
            name = row["metric"]
            value = row["value"]
            if isinstance(name, str) and (value is None or isinstance(value, float)):
                per_seed_by_metric.setdefault(name, {})[parquet_seed] = value

    if not seeds_seen:
        raise ValueError("no successful rollouts to score in this evaluation batch")
    if len(runners_seen) > 1:
        # Seeds rolled by different runners are not one result. Refuse rather than pick a winner.
        raise ValueError(
            "this evaluation batch mixes runners, so its seeds are not comparable: "
            f"{', '.join(sorted(runners_seen))}"
        )

    seeds = tuple(sorted(seeds_seen))
    per_seed = {
        metric.name: [per_seed_by_metric.get(metric.name, {}).get(seed) for seed in seeds]
        for metric in metric_set
    }
    # The runner is read off the artifacts, never assumed: each worker stamps the id it actually
    # resolved, and a batch that omits it or mixes runners has already been refused above. This is
    # what makes a Sim-rolled scale-out scorecard distinguishable from a fixture one (bench#64).
    card = aggregate_scores(
        metric_set,
        per_seed,
        seeds,
        scenario_id=spec.scenario_id,
        runner=next(iter(runners_seen)),
    )
    return build_submission(
        request, card, integrity, source=source, provenance_hash=provenance_hash
    )
