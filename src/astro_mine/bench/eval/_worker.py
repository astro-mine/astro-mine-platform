"""The single-seed rollout worker — the argv Cloud runs per fanned-out seed (RM-P1-BENCH-11).

This is the command a planned :class:`~astro_mine.bench.eval._plan.PlannedEvaluation` fans out
(``python -m astro_mine.bench eval-worker …``): it rolls **one** seed of a scenario and writes the
two artifacts Bench collects — a per-seed ``metrics.parquet`` (the columnar result; bench.md §5) and
a ``trace.mcap`` (the raw episode trace, in Sim's recording shape so
:func:`~astro_mine.bench.recording.decode_recording` reads it back; bench.md §6). The *same argv*
runs in CI (Cloud's local backend, this Bench env → the ``fixture`` runner) and on a real cluster
(inside the Sim rollout image → ``--runner sim``): that is how "the Sim rollout runner is supplied
via Cloud, not imported" is realized — the runner resolves **by name** through
:func:`~astro_mine.bench.baseline.load_runner_provider` and the ``astro_mine.bench.runners``
entry-point group, so Bench ships no Sim code and never imports ``astro_mine.sim``.

**The resolved runner id travels back with the result** (bench#64). Before this, the worker drove
``reference_episode_runner`` unconditionally — so the fixture rolled every seed even inside the Sim
image, and :mod:`~astro_mine.bench.eval._collect` stamped ``fixture/0.1.0`` on the scorecard by
construction. A scorecard that cannot name what produced it is not provenance (G1.8), so the id is
now written into **both** hand-back paths: the ``runner`` column of the per-seed Parquet
(``--emit artifacts``) and the ``runner`` field of the
:class:`~astro_mine.bench.sandbox.WorkerResult` (``--emit json``, the sandboxed leaderboard path).
The collector fails closed if it is missing or disagrees across seeds.

**This same argv is also the sandboxed evaluation worker** (bench#30): the leaderboard executes
every submitted policy by running *this command* inside a
:class:`~astro_mine.bench.sandbox.Sandbox`, rather than importing the policy into the evaluator
(bench.md §9). ``--emit json`` selects that mode: the worker writes only the structured
:class:`~astro_mine.bench.sandbox.WorkerResult` hand-back document (``result.json``) and skips the
Parquet/MCAP artifacts — so the sandboxed path needs neither ``pyarrow`` nor ``mcap``, and a policy
that fails to import, raises, or blows a metric up is reported back **as data** (``ok=false`` plus
an ``error``) instead of as an exception in the evaluator. It is the *worker*, inside the sandbox,
that resolves (imports) the untrusted ``--policy-ref``.

``pyarrow`` (the ``[cloud]`` extra) and ``mcap`` (the ``[recording]`` extra) are imported **lazily**
at write time, so importing this module stays dependency-clean (core + pydantic).

Backlog: RM-P1-BENCH-11 — astro-mine-bench#19;
bench#30 — astro-mine-bench#30
"""

from __future__ import annotations

import argparse
import json
import os
import resource
from collections.abc import Mapping, Sequence
from pathlib import Path

from astro_mine.bench.baseline import load_runner_provider
from astro_mine.bench.eval._plan import DEFAULT_RUNNER, METRICS_OUTPUT, SEED_ENV, TRACE_OUTPUT
from astro_mine.bench.leaderboard._eval import resolve_policy
from astro_mine.bench.metrics import resolve_metrics
from astro_mine.bench.recording import FRAMES_TOPIC, PROVENANCE_ATTACHMENT
from astro_mine.bench.sandbox import (
    WORKER_RESULT,
    ResourceUsage,
    WorkerMetric,
    WorkerResult,
)
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.scenario._hash import content_hash
from astro_mine.bench.zoo import load_scenario
from astro_mine.core.scoring import EpisodeTrace

__all__ = ["DEFAULT_RUNNER", "METRICS_COLUMNS", "REGISTRY_ENV", "run_worker"]

#: The columns of the per-seed metrics Parquet (bench.md §5) — the transparent per-metric record.
#: ``runner`` (bench#64) is the id of the runner that actually rolled the seed; the collector reads
#: it rather than assuming, and refuses a batch that omits it.
METRICS_COLUMNS = (
    "seed",
    "runner",
    "metric",
    "unit",
    "direction",
    "aggregation",
    "version",
    "value",
)

#: The Cloud env var naming the content store an engine-backed runner reads. The Sim rollout image
#: sets it; the fixture ignores it. Mirrors the ``score`` CLI's store precedence minus
#: ``--registry`` — a fanned-out worker's store is a property of its image, not of its argv.
REGISTRY_ENV = "ASTRO_MINE_HUB_REGISTRY"

#: The Cloud env var the run harness exports the outputs directory as (``cloud`` submission/_run).
OUTPUTS_ENV = "ASTRO_MINE_OUTPUTS"
#: The sweep's derived per-variant seed, a fallback if the explicit eval-seed env is unset.
FALLBACK_SEED_ENV = "ASTRO_MINE_SEED"
#: ``artifacts`` writes the Parquet + MCAP the Cloud collector reads (RM-P1-BENCH-11); ``json``
#: writes only the sandbox hand-back document (bench#30) — no ``pyarrow`` / ``mcap`` needed.
EMIT_CHOICES = ("artifacts", "json")


def _resolve_seed(args_seed: int | None, env: Mapping[str, str]) -> int:
    """Resolve the scenario seed from ``--seed``, then :data:`SEED_ENV`, then the sweep seed."""
    if args_seed is not None:
        return args_seed
    for key in (SEED_ENV, FALLBACK_SEED_ENV):
        raw = env.get(key)
        if raw is not None and raw != "":
            return int(raw)
    raise ValueError(f"no seed: pass --seed or set ${SEED_ENV}")


def _write_metrics_parquet(
    path: Path, *, seed: int, runner: str, rows: Sequence[Mapping[str, object]]
) -> None:
    """Write the per-seed metric values to ``path`` as Parquet (lazily importing pyarrow).

    ``runner`` is stamped on every row so the collector can read the producing runner off the
    artifact instead of assuming one (bench#64). It is constant within a seed by construction —
    one worker process rolls one seed with one runner.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("seed", pa.int64()),
            ("runner", pa.string()),
            ("metric", pa.string()),
            ("unit", pa.string()),
            ("direction", pa.string()),
            ("aggregation", pa.string()),
            ("version", pa.string()),
            ("value", pa.float64()),
        ]
    )
    table = pa.table(
        {
            "seed": [seed] * len(rows),
            "runner": [runner] * len(rows),
            "metric": [row["metric"] for row in rows],
            "unit": [row["unit"] for row in rows],
            "direction": [row["direction"] for row in rows],
            "aggregation": [row["aggregation"] for row in rows],
            "version": [row["version"] for row in rows],
            "value": [row["value"] for row in rows],
        },
        schema=schema,
    )
    pq.write_table(table, path)


def _write_trace_mcap(path: Path, *, seed: int, scenario_id: str, trace: EpisodeTrace) -> None:
    """Write ``trace`` to ``path`` as an MCAP in Sim's recording shape (lazily importing mcap)."""
    from mcap.writer import Writer

    observations = [observation.model_dump(mode="json") for observation in trace.observations]
    provenance = {
        "content_hash": content_hash(observations),
        "run": {"seed": seed},
        "scenario_id": scenario_id,
        "environment": {},
    }
    with path.open("wb") as handle:
        writer = Writer(handle)
        writer.start(profile="astro-mine-sim", library="astro-mine-bench")
        schema_id = writer.register_schema(name="frame", encoding="jsonschema", data=b"{}")
        channel_id = writer.register_channel(
            topic=FRAMES_TOPIC, message_encoding="json", schema_id=schema_id
        )
        for sequence, observation in enumerate(observations):
            frame = {
                "kind": "step",
                "observations": {observation["agent_id"]: observation},
            }
            writer.add_message(
                channel_id=channel_id,
                log_time=sequence,
                data=json.dumps(frame).encode(),
                publish_time=sequence,
                sequence=sequence,
            )
        writer.add_attachment(
            create_time=0,
            log_time=0,
            name=PROVENANCE_ATTACHMENT,
            media_type="application/json",
            data=json.dumps(provenance).encode(),
        )
        writer.finish()


def _self_usage() -> ResourceUsage:
    """This worker's own ``getrusage`` figures — advisory telemetry for the evaluator (bench#30).

    Self-reported, so it is *not* a control: the caps are enforced by the kernel (rlimits/cgroups),
    which untrusted code cannot lie its way past. Wall-clock is measured by the parent.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return ResourceUsage(
        wall_seconds=0.0,
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        # ru_maxrss is kilobytes on Linux.
        max_rss_bytes=max(usage.ru_maxrss, 0) * 1024,
    )


def _write_worker_result(path: Path, result: WorkerResult) -> None:
    """Write the structured hand-back document — the only thing that leaves the sandbox."""
    path.write_text(result.model_dump_json(), encoding="utf-8")


def run_worker(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None) -> int:
    """Roll one seed of a scenario, write its outputs, and return the process exit code.

    Loads the zoo scenario, resolves the submitted ``policy_ref``, drives the ``--runner`` named
    runner for the resolved seed, and computes each pinned metric's per-seed value. A runner that is
    not registered (``sim`` without ``astro-mine-platform[sim-bench]``) is reported back as a
    structured
    failure like any other, never a traceback. Always writes the
    structured :class:`~astro_mine.bench.sandbox.WorkerResult` hand-back document (``result.json``);
    with ``--emit artifacts`` (the default, the Cloud scale-out path) it *also* writes the per-seed
    ``metrics.parquet`` + ``trace.mcap``. Deterministic in the seed + inputs, so a collected run
    reproduces the workstation run byte-for-byte.

    Returns ``0`` when the seed scored and ``1`` when the submission failed — a failure is written
    into ``result.json`` as data (``ok=false`` + ``error``), because the caller is an evaluator that
    must never execute, import, or trust anything this process produced.
    """
    environment = os.environ if env is None else env
    # The argv this process is actually launched with (`_plan.build_command`,
    # `sandbox._subprocess.worker_command`), not a CLI verb: `eval-worker` is deliberately not one
    # (cli.md §10), so a `prog` naming a component binary printed a usage line nobody could type.
    parser = argparse.ArgumentParser(prog="python -m astro_mine.bench eval-worker")
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--policy-ref", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-out", default=METRICS_OUTPUT)
    parser.add_argument("--trace-out", default=TRACE_OUTPUT)
    parser.add_argument("--result-out", default=WORKER_RESULT)
    parser.add_argument("--emit", choices=EMIT_CHOICES, default="artifacts")
    parser.add_argument(
        "--runner",
        default=DEFAULT_RUNNER,
        metavar="NAME",
        help=(
            f"runner to roll the seed with (default: {DEFAULT_RUNNER}). '{DEFAULT_RUNNER}' is a "
            "deterministic trace fixture, not a physics engine; 'sim' runs the real Sim engine and "
            "needs astro-mine-platform[sim-bench] plus a content store in the image. The "
            "resolved runner id "
            "is written into the outputs and stamped on the collected scorecard."
        ),
    )
    args = parser.parse_args(argv)

    seed = _resolve_seed(args.seed, environment)
    output_dir = Path(args.output_dir or environment.get(OUTPUTS_ENV, "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / args.result_out
    registry = environment.get(REGISTRY_ENV) or None

    try:
        trace, rows, spec, runner_id = _roll_seed(
            args.scenario_id, args.policy_ref, seed, runner=args.runner, store=registry
        )
    except Exception as exc:
        # The submitted policy is untrusted: an unresolvable reference, a policy that raises, or a
        # metric that blows up is a *result*, handed back over the channel, not an evaluator crash.
        _write_worker_result(
            result_path,
            WorkerResult(
                ok=False,
                scenario_id=args.scenario_id,
                policy_ref=args.policy_ref,
                seed=seed,
                error=f"{type(exc).__name__}: {exc}",
                usage=_self_usage(),
            ),
        )
        return 1

    _write_worker_result(
        result_path,
        WorkerResult(
            ok=True,
            scenario_id=spec.scenario_id,
            policy_ref=args.policy_ref,
            seed=seed,
            runner=runner_id,
            metrics=tuple(
                WorkerMetric(
                    metric=str(row["metric"]),
                    version=str(row["version"]),
                    unit=str(row["unit"]),
                    direction=str(row["direction"]),
                    aggregation=str(row["aggregation"]),
                    value=None if row["value"] is None else float(row["value"]),  # type: ignore[arg-type]
                )
                for row in rows
            ),
            usage=_self_usage(),
        ),
    )

    if args.emit == "artifacts":
        _write_metrics_parquet(
            output_dir / args.metrics_out, seed=seed, runner=runner_id, rows=rows
        )
        _write_trace_mcap(
            output_dir / args.trace_out, seed=seed, scenario_id=spec.scenario_id, trace=trace
        )
    return 0


def _roll_seed(
    scenario_id: str,
    policy_ref: str,
    seed: int,
    *,
    runner: str = DEFAULT_RUNNER,
    store: object | None = None,
) -> tuple[EpisodeTrace, list[dict[str, object]], ScenarioSpec, str]:
    """Resolve the scenario + the (untrusted) policy, roll the seed, and compute its metrics.

    ``runner`` names a provider in the ``astro_mine.bench.runners`` entry-point group — resolved by
    name, never imported directly, so this module stays Sim-free (conventions.md §1.1). Returns the
    provider's ``runner_id`` alongside the trace so the caller can stamp what actually ran.
    """
    from astro_mine.bench.scenario import resolve_scenario

    spec = load_scenario(scenario_id)
    provider = load_runner_provider(runner)
    episode_runner = provider.episode_runner(store)
    # The untrusted import happens *here* — inside the sandbox — and never in the evaluator.
    policy = resolve_policy(policy_ref)
    resolved = resolve_scenario(spec)
    trace = episode_runner(resolved, policy, seed)
    metrics = resolve_metrics(spec.metrics)

    rows: list[dict[str, object]] = [
        {
            "metric": metric.name,
            "unit": metric.unit,
            "direction": metric.direction.value,
            "aggregation": metric.aggregation.value,
            "version": metric.version,
            "value": metric.compute(trace).value,
        }
        for metric in metrics
    ]
    return trace, rows, spec, provider.runner_id
