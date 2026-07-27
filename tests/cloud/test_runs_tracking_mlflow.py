"""MlflowTrackingClient against a *real* MLflow tracking store (RM-P1-CLOUD-05; cloud.md §5).

Hermetic and mock-free: MLflow's own local **file store** is a real tracking store, so every call
the client makes -- experiment creation, run creation, params, metrics, artifact-ref tags,
termination -- runs through real MLflow and is read back through ``MlflowClient``. That covers the
production backend in CI with no server, mirroring how ``moto`` covers the S3 path and ``fakeredis``
the Redis path (``mlflow-skinny`` rides the dev group; ``conventions.md`` §11).

The live *tracking-server* (REST store) path is the opt-in ``mlflow``-marked test in
``test_mlflow_integration.py``; the seam's in-memory fake lives in ``test_runs_tracking.py``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest

from astro_mine.cloud.artifacts.runcontext import RunContext
from astro_mine.cloud.runs.tracking import MlflowTrackingClient, RunTracker, TrackingClient

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("mlflow", reason="the dev group ships mlflow-skinny for this")

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

PIN_TAG = "astro-mine.org/run"


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A throwaway MLflow **file store**, the hermetic stand-in for a tracking server."""
    # MLflow 3 puts the file store in maintenance mode behind this opt-out; it is what keeps the
    # backend real *and* serverless in CI. A deployment points at http(s):// (docker-compose.yml).
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    store = tmp_path / "mlruns"
    store.mkdir()
    return store.as_uri()


def _context() -> RunContext:
    return RunContext(
        seed=42,
        image_digest="sha256:" + "a" * 64,
        core_interface_version="0.1.0",
        env_lockfile="sha256:" + "b" * 64,
        source_content_hashes={"scenario": "sha256:" + "c" * 64},
        outputs={"trace.mcap": "sha256:" + "d" * 64, "metrics.json": "sha256:" + "e" * 64},
    )


def _tags(run: Any) -> dict[str, str]:
    """The run's own tags, minus MLflow's internal ``mlflow.*`` bookkeeping."""
    return {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}


def test_real_client_satisfies_the_seam(tracking_uri: str) -> None:
    assert isinstance(MlflowTrackingClient(tracking_uri), TrackingClient)


def test_record_round_trips_the_envelope_and_outputs(tracking_uri: str) -> None:
    """A RunContext + its content-addressed outputs are retrievable from the store afterwards."""
    context = _context()
    tracker = RunTracker(MlflowTrackingClient(tracking_uri), experiment="astro-mine-test")

    stamped = tracker.record(context, params={"engine": "sim"}, metrics={"reward": 1.5, "ice": 0.2})

    assert stamped.run_id is not None
    inspector = MlflowClient(tracking_uri=tracking_uri)
    run = inspector.get_run(stamped.run_id)

    # the reproducibility envelope, as params (cloud.md §5)
    assert run.data.params["schema_version"] == "0.1"
    assert run.data.params["seed"] == "42"
    assert run.data.params["image_digest"] == context.image_digest
    assert run.data.params["core_interface_version"] == "0.1.0"
    assert run.data.params["env_lockfile"] == context.env_lockfile
    assert run.data.params["input.scenario"] == "sha256:" + "c" * 64
    assert run.data.params["engine"] == "sim"  # caller-supplied params merge in
    # metrics
    assert run.data.metrics == {"reward": 1.5, "ice": 0.2}
    # the outputs, as content-addressed refs -- the addresses, not the bytes
    assert _tags(run) == {
        PIN_TAG: context.content_address(),
        "artifact.trace.mcap": "sha256:" + "d" * 64,
        "artifact.metrics.json": "sha256:" + "e" * 64,
    }
    assert run.info.status == "FINISHED"
    assert (
        run.info.experiment_id == inspector.get_experiment_by_name("astro-mine-test").experiment_id
    )


def test_run_is_findable_by_its_reproducibility_pin(tracking_uri: str) -> None:
    """The pin tag is a real query key: a re-run is findable from the recorded envelope."""
    tracker = RunTracker(MlflowTrackingClient(tracking_uri), experiment="pinned")
    context = _context()
    first = tracker.record(context)
    second = tracker.record(context)  # same job, run again -> same pin, new MLflow run

    inspector = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = inspector.get_experiment_by_name("pinned").experiment_id
    found = inspector.search_runs(
        [experiment_id], filter_string=f"tags.`{PIN_TAG}` = '{context.content_address()}'"
    )

    assert first.run_id != second.run_id
    assert {r.info.run_id for r in found} == {first.run_id, second.run_id}


def test_calls_are_addressed_by_run_id_not_a_global_active_run(tracking_uri: str) -> None:
    """Two runs open at once stay separate -- the seam is run-id-addressed, not process-global.

    The fluent ``mlflow.start_run()`` API would attribute both runs' params/metrics/tags to
    whichever run is "active" (and refuse the second ``start_run`` outright); the explicit
    client must not.
    """
    client = MlflowTrackingClient(tracking_uri)
    first = client.start_run(experiment="interleaved", tags={PIN_TAG: "sha256:" + "1" * 64})
    second = client.start_run(experiment="interleaved", tags={PIN_TAG: "sha256:" + "2" * 64})
    assert first != second

    client.log_params(first, {"seed": "1"})
    client.log_params(second, {"seed": "2"})
    client.log_metrics(first, {"reward": 1.0})
    client.log_metrics(second, {"reward": 2.0})
    client.log_artifact_ref(first, "trace.mcap", "sha256:" + "d" * 64)
    client.log_artifact_ref(second, "trace.mcap", "sha256:" + "f" * 64)
    client.end_run(first, "FINISHED")
    client.end_run(second, "FAILED")

    inspector = MlflowClient(tracking_uri=tracking_uri)
    run_one, run_two = inspector.get_run(first), inspector.get_run(second)
    assert run_one.data.params == {"seed": "1"}
    assert run_two.data.params == {"seed": "2"}
    assert run_one.data.metrics == {"reward": 1.0}
    assert run_two.data.metrics == {"reward": 2.0}
    assert _tags(run_one)["artifact.trace.mcap"] == "sha256:" + "d" * 64
    assert _tags(run_two)["artifact.trace.mcap"] == "sha256:" + "f" * 64
    assert run_one.info.status == "FINISHED"
    assert run_two.info.status == "FAILED"  # a failed job is recorded as a failed run


def test_experiment_is_created_once_then_reused(tracking_uri: str) -> None:
    RunTracker(MlflowTrackingClient(tracking_uri), experiment="reused").record(RunContext())
    # a *fresh* client (a second worker) finds the existing experiment rather than re-creating it
    RunTracker(MlflowTrackingClient(tracking_uri), experiment="reused").record(RunContext())

    inspector = MlflowClient(tracking_uri=tracking_uri)
    named = [e for e in inspector.search_experiments() if e.name == "reused"]
    assert len(named) == 1
    assert len(inspector.search_runs([named[0].experiment_id])) == 2


def test_experiment_creation_race_adopts_the_winners_experiment(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two workers race to create one experiment: the loser adopts the winner's, not an error.

    The stale read is simulated (the first lookup returns ``None``, as it would before the winner
    committed); the failure itself is real -- MLflow raises ``RESOURCE_ALREADY_EXISTS`` from a
    genuine duplicate ``create_experiment``.
    """
    loser = MlflowTrackingClient(tracking_uri)
    real_lookup = loser._client.get_experiment_by_name
    lookups = 0

    def stale_first_lookup(name: str) -> Any:
        nonlocal lookups
        lookups += 1
        return None if lookups == 1 else real_lookup(name)  # the pre-race read misses

    monkeypatch.setattr(loser._client, "get_experiment_by_name", stale_first_lookup)

    winner_run = MlflowTrackingClient(tracking_uri).start_run(experiment="raced", tags={})
    loser_run = loser.start_run(experiment="raced", tags={})

    inspector = MlflowClient(tracking_uri=tracking_uri)
    assert inspector.get_run(loser_run).info.experiment_id == (
        inspector.get_run(winner_run).info.experiment_id
    )


def test_a_create_failure_that_is_not_the_race_surfaces(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that genuinely rejects the create must raise -- the race fallback swallows nothing.

    Only a *lost race* is recoverable (the experiment now exists); any other create failure is a
    real error and must reach the caller rather than be retried into a confusing lookup miss.
    """
    client = MlflowTrackingClient(tracking_uri)

    def rejected(name: str, *args: Any, **kwargs: Any) -> str:
        raise MlflowException("store unavailable")

    monkeypatch.setattr(client._client, "create_experiment", rejected)

    with pytest.raises(MlflowException, match="store unavailable"):
        client.start_run(experiment="never-created", tags={})


def test_tracking_uri_defaults_to_mlflows_own_resolution(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``None`` defers to MLflow (``MLFLOW_TRACKING_URI``) instead of pinning a URI."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    run_id = MlflowTrackingClient().start_run(experiment="from-env", tags={})
    assert MlflowClient(tracking_uri=tracking_uri).get_run(run_id).info.run_id == run_id


def test_missing_extra_raises_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", None)
    with pytest.raises(ModuleNotFoundError, match=r"astro-mine-cloud\[mlflow\]"):
        MlflowTrackingClient()
