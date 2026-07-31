"""Scale-out evaluation on Cloud (RM-P1-BENCH-11).

Covers the planner (seeds x submissions -> Cloud sweeps, CPU-Argo vs GPU-KubeRay routing, budgets),
the single-seed worker (MCAP + Parquet), the dispatch seam (Cloud ``DryRunClient`` byte-identical
equivalence, no live cluster), the collector (byte-identical scorecard, integrity), and the
orchestrator (fair-share admission, per-submission budget caps, completion events, ranking) — all
without a cluster and **without importing Sim**.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from astro_mine.bench.baseline import REFERENCE_EPISODE_RUNNER_ID, BaselinePolicy, run
from astro_mine.bench.eval import (
    METRICS_OUTPUT,
    REFERENCE_ROLLOUT_IMAGE,
    SEED_ENV,
    TRACE_OUTPUT,
    AdmissionDenied,
    CloudBatchDispatcher,
    EvaluationTarget,
    LocalBatchDispatcher,
    assert_batch_reproducible,
    collect_submission,
    plan_batch,
    read_metrics_parquet,
    run_evaluation_batch,
    run_worker,
)
from astro_mine.bench.eval._worker import (
    FALLBACK_SEED_ENV,
    METRICS_COLUMNS,
    OUTPUTS_ENV,
)
from astro_mine.bench.harness import DeterminismError
from astro_mine.bench.leaderboard import InMemoryStore, SubmissionRequest, rank
from astro_mine.bench.recording import decode_recording
from astro_mine.bench.sandbox import WORKER_RESULT, WorkerResult
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID, load_scenario
from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
from astro_mine.cloud.runs import SUBJECT, CollectingPublisher
from astro_mine.cloud.sched import BudgetExceeded, CostRates
from astro_mine.cloud.submission.cluster import DryRunClient
from astro_mine.cloud.submission.result import RunResult

BASELINE_REF = "astro_mine.bench.baseline:BaselinePolicy"
IDLE_REF = "tests.bench._factories:idle_baseline"
SEEDS = (1001, 1002, 1003)
ANCHOR_METRIC_COUNT = 7
#: The repo root (holds ``tests/``) — exposed to worker subprocesses via ``PYTHONPATH`` so a
#: ``tests.bench._factories:…`` policy_ref resolves inside the Cloud-run rollout (a test-only
#: crutch; a
#: real submission's policy resolves from its rollout image / Hub, never the local tests package).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def anchor() -> ScenarioSpec:
    return load_scenario(ANCHOR_SCENARIO_ID)


@pytest.fixture
def store() -> Iterator[FilesystemArtifactStore]:
    with tempfile.TemporaryDirectory(prefix="astro-mine-eval-test-") as tmp:
        yield FilesystemArtifactStore(root=tmp)


def _target(
    scenario_id: str, *, policy_ref: str = BASELINE_REF, **kwargs: object
) -> EvaluationTarget:
    request = SubmissionRequest(scenario_id=scenario_id, policy_ref=policy_ref, method="baseline")
    return EvaluationTarget(request=request, image=REFERENCE_ROLLOUT_IMAGE, **kwargs)  # type: ignore[arg-type]


# --- planner (RM-P1-BENCH-11: plan seeds x submissions onto Cloud) ------------------------------


def test_plan_batch_cpu_fans_out_over_seeds(anchor: ScenarioSpec) -> None:
    plan = plan_batch(anchor, [_target(anchor.scenario_id)], seeds=SEEDS, max_parallel=2)
    (planned,) = plan.evaluations
    assert not planned.distributed
    assert planned.sweep.grid == {SEED_ENV: list(SEEDS)}
    assert planned.sweep.max_parallel == 2
    assert planned.sweep.size() == len(SEEDS)
    base = planned.sweep.base
    assert base.outputs == [METRICS_OUTPUT, TRACE_OUTPUT]
    assert "eval-worker" in base.command and anchor.scenario_id in base.command
    assert base.core_interface_version == "0.1.0"  # from the pinned env interface
    assert planned.admission_request == {"cpu": 1.0}  # default one core
    assert plan.budget_caps == {}  # no compute_budget set


def test_plan_batch_gpu_routes_to_kuberay_with_mig(anchor: ScenarioSpec) -> None:
    plan = plan_batch(
        anchor, [_target(anchor.scenario_id, gpu=True, mig_profile="1g.10gb")], seeds=SEEDS
    )
    (planned,) = plan.evaluations
    assert planned.distributed  # → select_engine routes to Ray/KubeRay
    assert planned.sweep.base.resource_request is not None
    assert planned.sweep.base.resource_request.mig_profile == "1g.10gb"
    # a 1g slice of a 7-way a100-80gb card is priced at 1/7 of a GPU
    assert planned.admission_request == {"nvidia.com/gpu": pytest.approx(1.0 / 7.0)}


def test_plan_batch_gpu_whole_card_without_mig(anchor: ScenarioSpec) -> None:
    plan = plan_batch(anchor, [_target(anchor.scenario_id, gpu=True)], seeds=SEEDS)
    (planned,) = plan.evaluations
    assert planned.sweep.base.resource_request is not None
    assert planned.sweep.base.resource_request.gpu == 1
    assert planned.admission_request == {"nvidia.com/gpu": 1.0}


def test_plan_batch_cpu_quantity_and_budget(anchor: ScenarioSpec) -> None:
    plan = plan_batch(
        anchor,
        [_target(anchor.scenario_id, cpu="500m", memory="2Gi", compute_budget=5.0)],
        seeds=SEEDS,
    )
    (planned,) = plan.evaluations
    assert planned.admission_request == {"cpu": 0.5}
    assert planned.sweep.base.resource_request is not None
    assert planned.sweep.base.resource_request.cpu == "500m"
    assert planned.cap_key is not None
    assert plan.budget_caps == {planned.cap_key: 5.0}
    assert planned.cost_per_seed > 0.0


def test_plan_batch_rejects_empty_seeds(anchor: ScenarioSpec) -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        plan_batch(anchor, [_target(anchor.scenario_id)], seeds=())


def test_plan_batch_rejects_scenario_mismatch(anchor: ScenarioSpec) -> None:
    request = SubmissionRequest(scenario_id="some-other-scenario", policy_ref=BASELINE_REF)
    target = EvaluationTarget(request=request, image=REFERENCE_ROLLOUT_IMAGE)
    with pytest.raises(ValueError, match="!= spec"):
        plan_batch(anchor, [target], seeds=SEEDS)


def test_plan_batch_rejects_bad_mig_profile(anchor: ScenarioSpec) -> None:
    with pytest.raises(ValueError, match="MIG profile"):
        plan_batch(
            anchor, [_target(anchor.scenario_id, gpu=True, mig_profile="9g.99gb")], seeds=SEEDS
        )


def test_plan_batch_rejects_unpinned_image(anchor: ScenarioSpec) -> None:
    request = SubmissionRequest(scenario_id=anchor.scenario_id, policy_ref=BASELINE_REF)
    target = EvaluationTarget(request=request, image="ghcr.io/astro-mine/sim:latest")  # no digest
    with pytest.raises(ValueError, match="unpinned image"):
        plan_batch(anchor, [target], seeds=SEEDS)


# --- worker (RM-P1-BENCH-11: the single-seed rollout argv) --------------------------------------


def test_worker_writes_parquet_and_decodable_mcap(anchor: ScenarioSpec, tmp_path: Path) -> None:
    rc = run_worker(
        ["--scenario-id", anchor.scenario_id, "--policy-ref", BASELINE_REF, "--seed", "1001"],
        env={OUTPUTS_ENV: str(tmp_path)},
    )
    assert rc == 0
    rows = read_metrics_parquet((tmp_path / METRICS_OUTPUT).read_bytes())
    assert len(rows) == ANCHOR_METRIC_COUNT
    assert all(row["seed"] == 1001 for row in rows)
    assert {"metric", "unit", "direction", "aggregation", "version", "value"} <= set(rows[0])
    decoded = decode_recording(tmp_path / TRACE_OUTPUT)  # the recording reader round-trips the MCAP
    assert decoded.seed == 1001
    assert decoded.trace.observations


def test_worker_reads_seed_from_env(anchor: ScenarioSpec, tmp_path: Path) -> None:
    assert (
        run_worker(
            ["--scenario-id", anchor.scenario_id, "--policy-ref", BASELINE_REF],
            env={SEED_ENV: "1002", OUTPUTS_ENV: str(tmp_path)},
        )
        == 0
    )
    assert read_metrics_parquet((tmp_path / METRICS_OUTPUT).read_bytes())[0]["seed"] == 1002


def test_worker_falls_back_to_sweep_seed_env(anchor: ScenarioSpec, tmp_path: Path) -> None:
    assert (
        run_worker(
            [
                "--scenario-id",
                anchor.scenario_id,
                "--policy-ref",
                BASELINE_REF,
                "--output-dir",
                str(tmp_path),
            ],
            env={FALLBACK_SEED_ENV: "1003"},
        )
        == 0
    )
    assert read_metrics_parquet((tmp_path / METRICS_OUTPUT).read_bytes())[0]["seed"] == 1003


def test_worker_requires_a_seed(anchor: ScenarioSpec, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no seed"):
        run_worker(["--scenario-id", anchor.scenario_id, "--policy-ref", BASELINE_REF], env={})


def test_eval_worker_via_python_m(anchor: ScenarioSpec, tmp_path: Path) -> None:
    """The argv Cloud actually fans out — `python -m astro_mine.bench eval-worker …`.

    Bench's user-facing verbs moved to astro-mine-cli, but this one did not: it is the
    single-seed rollout Cloud builds per seed (RM-P1-BENCH-11) and the container backend
    re-runs inside a sandbox, so it stays reachable exactly where its callers look for it.
    """
    from astro_mine.bench.__main__ import main

    rc = main(
        [
            "eval-worker",
            "--scenario-id",
            anchor.scenario_id,
            "--policy-ref",
            BASELINE_REF,
            "--seed",
            "1001",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / METRICS_OUTPUT).is_file()


# --- dispatch seam (RM-P1-BENCH-11: DryRunClient equivalence, no cluster) ------------------------


def test_cloud_dispatcher_cpu_records_argo_fanout_and_k8s_jobs(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    (planned,) = plan_batch(
        anchor, [_target(anchor.scenario_id)], seeds=SEEDS, max_parallel=4
    ).evaluations
    client = DryRunClient()
    dispatcher = CloudBatchDispatcher(client=client)
    results = dispatcher.dispatch(planned, store=store)
    assert [r.status for r in results] == ["succeeded"] * len(SEEDS)
    # the whole sweep compiles to one Argo Workflow (the fan-out DAG + parallelism back-pressure)
    (workflow,) = dispatcher.sweep_manifests
    assert workflow["kind"] == "Workflow"
    assert workflow["spec"]["parallelism"] == 4
    assert len(workflow["spec"]["templates"][0]["dag"]["tasks"]) == len(SEEDS)
    # each CPU seed routes to a plain K8s Job (select_engine on distributed=False)
    assert [m["kind"] for m in client.dispatched] == ["Job"] * len(SEEDS)


def test_cloud_dispatcher_gpu_routes_each_seed_to_rayjob(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    (planned,) = plan_batch(
        anchor, [_target(anchor.scenario_id, gpu=True, mig_profile="1g.10gb")], seeds=SEEDS
    ).evaluations
    client = DryRunClient()
    dispatcher = CloudBatchDispatcher(client=client)
    dispatcher.dispatch(planned, store=store)
    assert [m["kind"] for m in client.dispatched] == ["RayJob"] * len(SEEDS)  # KubeRay
    assert (
        dispatcher.sweep_manifests == []
    )  # GPU rollouts are per-seed RayJobs, not an Argo fan-out
    assert dispatcher.client is client


def test_dryrun_dispatch_is_byte_identical_to_local(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    (planned,) = plan_batch(anchor, [_target(anchor.scenario_id)], seeds=SEEDS).evaluations
    cloud = CloudBatchDispatcher(client=DryRunClient()).dispatch(planned, store=store)
    local = LocalBatchDispatcher().dispatch(planned, store=store)
    for a, b in zip(cloud, local, strict=True):
        assert a.outputs == b.outputs
        assert a.run_context.content_address() == b.run_context.content_address()


# --- collector (RM-P1-BENCH-11: collect MCAP/Parquet, byte-identical scorecard) ------------------


def test_collect_scorecard_is_byte_identical_to_local_run(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    (planned,) = plan_batch(anchor, [_target(anchor.scenario_id)], seeds=SEEDS).evaluations
    results = LocalBatchDispatcher().dispatch(planned, store=store)
    submission = collect_submission(anchor, planned.request, results, artifact_store=store)
    assert submission.integrity == "verified"
    assert submission.scorecard_hash == run(anchor, BaselinePolicy(), seeds=SEEDS).content_hash


def test_collect_flags_a_failed_rollout(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    (planned,) = plan_batch(anchor, [_target(anchor.scenario_id)], seeds=SEEDS).evaluations
    results = list(LocalBatchDispatcher().dispatch(planned, store=store))
    failed = results[0].model_copy(update={"status": "failed", "exit_code": 1, "outputs": {}})
    submission = collect_submission(
        anchor, planned.request, [failed, *results[1:]], artifact_store=store
    )
    assert submission.integrity == "flagged"  # a dropped rollout is visible, not silently ignored


def test_collect_raises_when_no_rollout_succeeds(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    ctx = (
        LocalBatchDispatcher()
        .dispatch(
            plan_batch(anchor, [_target(anchor.scenario_id)], seeds=(1001,)).evaluations[0],
            store=store,
        )[0]
        .run_context
    )
    failed = RunResult(
        status="failed", exit_code=1, run_context_address="sha256:" + "0" * 64, run_context=ctx
    )
    with pytest.raises(ValueError, match="no successful rollouts"):
        collect_submission(
            anchor,
            SubmissionRequest(scenario_id=anchor.scenario_id, policy_ref=BASELINE_REF),
            [failed],
            artifact_store=store,
        )


def test_collect_flags_a_trace_seed_mismatch(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    # metrics for seed 1001, but the raw trace's provenance says 1002 → tamper/mismatch → flagged.
    (planned,) = plan_batch(anchor, [_target(anchor.scenario_id)], seeds=(1001, 1002)).evaluations
    good = LocalBatchDispatcher().dispatch(planned, store=store)
    by_seed = {
        read_metrics_parquet(store.get(r.outputs[METRICS_OUTPUT]))[0]["seed"]: r for r in good
    }
    mixed = by_seed[1001].model_copy(
        update={
            "outputs": {
                METRICS_OUTPUT: by_seed[1001].outputs[METRICS_OUTPUT],
                TRACE_OUTPUT: by_seed[1002].outputs[TRACE_OUTPUT],
            }
        }
    )
    # the mismatched 1001 result is skipped-and-flagged; the good 1002 result still scores.
    submission = collect_submission(
        anchor, planned.request, [mixed, by_seed[1002]], artifact_store=store
    )
    assert submission.integrity == "flagged"


# --- orchestrator (RM-P1-BENCH-11: admission, budgets, events, ranking) --------------------------


def test_run_batch_ingests_and_ranks(anchor: ScenarioSpec, store: FilesystemArtifactStore) -> None:
    lb = InMemoryStore()
    publisher = CollectingPublisher()
    submissions = run_evaluation_batch(
        anchor,
        [
            _target(anchor.scenario_id, policy_ref=BASELINE_REF),
            _target(anchor.scenario_id, policy_ref=IDLE_REF),
        ],
        seeds=SEEDS,
        dispatcher=CloudBatchDispatcher(client=DryRunClient()),
        leaderboard_store=lb,
        artifact_store=store,
        publisher=publisher,
        base_env={"PYTHONPATH": _REPO_ROOT},
    )
    assert len(submissions) == 2
    assert all(lb.get_submission(s.submission_id) is not None for s in submissions)
    entries = rank(lb.list_submissions(anchor.scenario_id))
    assert [e.rank for e in entries] == [1, 2]
    # a completion event per seed per submission, on the Cloud runs subject
    assert len(publisher.events) == 2 * len(SEEDS)
    assert {subject for subject, _ in publisher.events} == {SUBJECT}
    assert {event.status for _, event in publisher.events} == {"completed"}


def test_run_batch_enforces_per_submission_budget(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    lb = InMemoryStore()
    with pytest.raises(BudgetExceeded):
        run_evaluation_batch(
            anchor,
            [_target(anchor.scenario_id, compute_budget=0.001)],
            seeds=SEEDS,
            dispatcher=CloudBatchDispatcher(client=DryRunClient()),
            leaderboard_store=lb,
            artifact_store=store,
            cost_rates=CostRates(cpu_hour=1.0),
            hours_per_seed=1.0,
            spot=False,
        )
    assert lb.list_submissions(anchor.scenario_id) == []  # the sweep halted before ingesting


def test_run_batch_backpressure_denies_over_quota(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    with pytest.raises(AdmissionDenied):
        run_evaluation_batch(
            anchor,
            [_target(anchor.scenario_id, cpu="1")],
            seeds=SEEDS,
            dispatcher=CloudBatchDispatcher(client=DryRunClient()),
            leaderboard_store=InMemoryStore(),
            artifact_store=store,
            quotas={"public": {"cpu": 0.5}},
        )


def test_run_batch_admits_within_quota(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    submissions = run_evaluation_batch(
        anchor,
        [_target(anchor.scenario_id, cpu="1")],
        seeds=SEEDS,
        dispatcher=CloudBatchDispatcher(client=DryRunClient()),
        leaderboard_store=InMemoryStore(),
        artifact_store=store,
        quotas={"public": {"cpu": 2.0}},
    )
    assert submissions[0].integrity == "verified"


def test_run_batch_local_dispatcher_and_gpu(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    submissions = run_evaluation_batch(
        anchor,
        [_target(anchor.scenario_id, gpu=True, mig_profile="1g.10gb")],
        seeds=SEEDS,
        dispatcher=LocalBatchDispatcher(),
        leaderboard_store=InMemoryStore(),
        artifact_store=store,
    )
    assert submissions[0].integrity == "verified"


# --- determinism gate + no-Sim guard (RM-P1-BENCH-11 acceptance) ---------------------------------


def test_assert_batch_reproducible_returns_shared_hash(anchor: ScenarioSpec) -> None:
    card_hash = assert_batch_reproducible(anchor, policy_ref=BASELINE_REF, seeds=SEEDS)
    assert card_hash == run(anchor, BaselinePolicy(), seeds=SEEDS).content_hash


def test_assert_batch_reproducible_raises_on_drift(
    anchor: ScenarioSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Card:
        content_hash = "sha256:" + "f" * 64

    monkeypatch.setattr("astro_mine.bench.baseline.run", lambda *a, **k: _Card())
    with pytest.raises(DeterminismError, match="did not reproduce"):
        assert_batch_reproducible(anchor, policy_ref=BASELINE_REF, seeds=SEEDS)


def test_running_a_batch_never_imports_sim(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    _SIM_MODULES_BEFORE = frozenset(
        n for n in sys.modules if n == "astro_mine.sim" or n.startswith("astro_mine.sim.")
    )
    run_evaluation_batch(
        anchor,
        [_target(anchor.scenario_id)],
        seeds=SEEDS,
        dispatcher=CloudBatchDispatcher(client=DryRunClient()),
        leaderboard_store=InMemoryStore(),
        artifact_store=store,
    )
    # In astro-mine-platform Sim is co-installed, so earlier tests in this process may have
    # loaded it; assert the batch run itself added no Sim modules.
    assert not any(
        name == "astro_mine.sim" or name.startswith("astro_mine.sim.")
        for name in sys.modules
        if name not in _SIM_MODULES_BEFORE
    )


def test_importing_eval_is_dependency_clean() -> None:
    # A fresh interpreter: importing the eval package pulls neither Sim nor Cloud (both are lazy).
    code = (
        "import astro_mine.bench.eval, sys; "
        "assert 'astro_mine.sim' not in sys.modules; "
        "assert 'astro_mine.cloud' not in sys.modules"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# --- runner provenance (bench#64: the scale-out path names what actually rolled the seeds) -------


def _reparquet(rows: list[dict[str, object]], *, seed: int, runner: str | None) -> bytes:
    """Re-encode worker rows with a substituted (or removed) ``runner`` column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = [c for c in METRICS_COLUMNS if runner is not None or c != "runner"]
    data = {
        c: [seed if c == "seed" else (runner if c == "runner" else row[c]) for row in rows]
        for c in columns
    }
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.table(data), buffer)
    return bytes(buffer.getvalue().to_pybytes())


def _restamp(
    results: list[RunResult], store: FilesystemArtifactStore, runners: Sequence[str | None]
) -> list[RunResult]:
    """Rewrite each seed's Parquet so its ``runner`` column reads ``runners[i]``."""
    out: list[RunResult] = []
    for result, runner in zip(results, runners, strict=True):
        rows = read_metrics_parquet(store.get(result.outputs[METRICS_OUTPUT]))
        seed = int(rows[0]["seed"])  # type: ignore[arg-type]
        digest = store.put(_reparquet(rows, seed=seed, runner=runner))
        out.append(
            result.model_copy(update={"outputs": {**result.outputs, METRICS_OUTPUT: digest}})
        )
    return out


def test_worker_stamps_the_runner_it_resolved(anchor: ScenarioSpec, tmp_path: Path) -> None:
    """The fixture is no longer assumed — the worker records the runner it actually resolved."""
    rc = run_worker(
        ["--scenario-id", anchor.scenario_id, "--policy-ref", BASELINE_REF, "--seed", "1001"],
        env={OUTPUTS_ENV: str(tmp_path)},
    )
    assert rc == 0
    rows = read_metrics_parquet((tmp_path / METRICS_OUTPUT).read_bytes())
    assert all(row["runner"] == REFERENCE_EPISODE_RUNNER_ID for row in rows)
    # ... and on the sandbox hand-back channel too, which carries no Parquet at all.
    result = WorkerResult.model_validate_json((tmp_path / WORKER_RESULT).read_text())
    assert result.runner == REFERENCE_EPISODE_RUNNER_ID


@pytest.mark.skip(
    reason="sibling-absent state unreachable in astro-mine-platform: Sim ships in the same "
    "distribution, so the 'sim runner unavailable' structured-failure path cannot occur"
)
def test_worker_reports_an_unavailable_runner_as_data(anchor: ScenarioSpec, tmp_path: Path) -> None:
    """`--runner sim` without astro-mine-sim[bench] is a structured failure, never a traceback."""
    rc = run_worker(
        [
            "--scenario-id",
            anchor.scenario_id,
            "--policy-ref",
            BASELINE_REF,
            "--seed",
            "1001",
            "--runner",
            "sim",
            "--emit",
            "json",
        ],
        env={OUTPUTS_ENV: str(tmp_path)},
    )
    assert rc == 1
    result = WorkerResult.model_validate_json((tmp_path / WORKER_RESULT).read_text())
    assert result.ok is False
    assert "astro-mine-platform[sim-bench]" in (result.error or "")
    assert result.runner is None  # it never got far enough to resolve one


def test_the_plan_names_the_runner_in_the_fanned_out_argv(anchor: ScenarioSpec) -> None:
    (planned,) = plan_batch(
        anchor, [_target(anchor.scenario_id)], seeds=SEEDS, runner="sim"
    ).evaluations
    command = planned.sweep.base.command
    assert command[command.index("--runner") + 1] == "sim"


def test_collect_stamps_the_runner_read_from_the_artifacts(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    """The regression bench#64 is about: a non-fixture rollout must not report `fixture/0.1.0`."""
    (planned,) = plan_batch(anchor, [_target(anchor.scenario_id)], seeds=SEEDS).evaluations
    results = list(LocalBatchDispatcher().dispatch(planned, store=store))
    restamped = _restamp(results, store, ["astro-mine-sim/0.1.0"] * len(results))

    submission = collect_submission(anchor, planned.request, restamped, artifact_store=store)

    # The runner folds into the scorecard's content hash, so matching the hash of a locally-scored
    # card built with that runner id proves the collected card carries exactly it — nothing else
    # about the two cards differs.
    assert (
        submission.scorecard_hash
        == run(anchor, BaselinePolicy(), runner_id="astro-mine-sim/0.1.0", seeds=SEEDS).content_hash
    )
    # And it is emphatically not the fixture scorecard, which is what this path used to claim.
    assert submission.scorecard_hash != run(anchor, BaselinePolicy(), seeds=SEEDS).content_hash


def test_collect_flags_a_seed_that_does_not_name_its_runner(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    """Fail closed: an unattributable seed is dropped and the submission is flagged."""
    (planned,) = plan_batch(anchor, [_target(anchor.scenario_id)], seeds=SEEDS).evaluations
    results = list(LocalBatchDispatcher().dispatch(planned, store=store))
    stripped = _restamp(results[:1], store, [None])

    submission = collect_submission(
        anchor, planned.request, [*stripped, *results[1:]], artifact_store=store
    )

    assert submission.integrity == "flagged"
    # The unattributable seed is dropped, not guessed at: the card is the *remaining* seeds', still
    # correctly attributed to the runner they did name.
    assert submission.scorecard_hash == run(anchor, BaselinePolicy(), seeds=SEEDS[1:]).content_hash


def test_collect_refuses_a_batch_that_mixes_runners(
    anchor: ScenarioSpec, store: FilesystemArtifactStore
) -> None:
    """Seeds rolled by different runners are not one result — refuse rather than pick a winner."""
    (planned,) = plan_batch(anchor, [_target(anchor.scenario_id)], seeds=SEEDS).evaluations
    results = list(LocalBatchDispatcher().dispatch(planned, store=store))
    mixed = _restamp(
        results, store, [REFERENCE_EPISODE_RUNNER_ID, "astro-mine-sim/0.1.0", "other/0.1.0"]
    )

    with pytest.raises(ValueError, match="mixes runners"):
        collect_submission(anchor, planned.request, mixed, artifact_store=store)
