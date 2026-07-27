"""Opt-in integration test against a real MLflow **tracking server** (RM-P1-CLOUD-05; cloud.md §5).

Exercises :class:`MlflowTrackingClient` over MLflow's REST store -- a live server backed by a
database, the cluster-tier shape (``cloud.md`` §7) -- rather than the local file store the hermetic
``test_runs_tracking_mlflow.py`` runs against in CI. Requires the ``[mlflow]`` extra and a running
server; self-skips otherwise, so CI (which runs no server) never depends on it -- the same opt-in
shape as the MinIO and NATS integration tests. Run it with::

    docker compose up -d mlflow
    MLFLOW_TRACKING_URI=http://localhost:5000 uv run pytest -m mlflow
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient

from astro_mine.cloud.artifacts.runcontext import RunContext
from astro_mine.cloud.runs.tracking import MlflowTrackingClient, RunTracker

pytestmark = pytest.mark.mlflow

PIN_TAG = "astro-mine.org/run"

_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
_skip = pytest.mark.skipif(
    not _TRACKING_URI,
    reason="set MLFLOW_TRACKING_URI (and `docker compose up -d mlflow`) to exercise a real server",
)


@_skip
def test_run_context_and_outputs_round_trip_through_a_live_server() -> None:
    uri = _TRACKING_URI or ""
    experiment = f"astro-mine-integration-{uuid.uuid4().hex[:8]}"
    context = RunContext(
        seed=42,
        image_digest="sha256:" + "a" * 64,
        core_interface_version="0.1.0",
        env_lockfile="sha256:" + "b" * 64,
        source_content_hashes={"scenario": "sha256:" + "c" * 64},
        outputs={"trace.mcap": "sha256:" + "d" * 64},
    )

    tracker = RunTracker(MlflowTrackingClient(uri), experiment=experiment)
    stamped = tracker.record(context, params={"engine": "sim"}, metrics={"reward": 1.5})

    assert stamped.run_id is not None
    inspector = MlflowClient(tracking_uri=uri)
    run = inspector.get_run(stamped.run_id)
    # the reproducibility envelope survives the round trip to the server
    assert run.data.params["seed"] == "42"
    assert run.data.params["image_digest"] == context.image_digest
    assert run.data.params["core_interface_version"] == "0.1.0"
    assert run.data.params["env_lockfile"] == context.env_lockfile
    assert run.data.params["input.scenario"] == "sha256:" + "c" * 64
    assert run.data.params["engine"] == "sim"
    assert run.data.metrics["reward"] == 1.5
    # the content-addressed outputs are retrievable as refs, and the run is keyed by its pin
    assert run.data.tags["artifact.trace.mcap"] == "sha256:" + "d" * 64
    assert run.data.tags[PIN_TAG] == context.content_address()
    assert run.info.status == "FINISHED"

    experiment_id = inspector.get_experiment_by_name(experiment).experiment_id
    found = inspector.search_runs(
        [experiment_id], filter_string=f"tags.`{PIN_TAG}` = '{context.content_address()}'"
    )
    assert [r.info.run_id for r in found] == [stamped.run_id]


@_skip
def test_concurrent_runs_stay_separate_on_a_live_server() -> None:
    """Two runs open at once against one server: no process-global active run, no cross-talk."""
    uri = _TRACKING_URI or ""
    experiment = f"astro-mine-integration-{uuid.uuid4().hex[:8]}"
    client = MlflowTrackingClient(uri)

    first = client.start_run(experiment=experiment, tags={PIN_TAG: "sha256:" + "1" * 64})
    second = client.start_run(experiment=experiment, tags={PIN_TAG: "sha256:" + "2" * 64})
    client.log_params(first, {"seed": "1"})
    client.log_params(second, {"seed": "2"})
    client.log_metrics(second, {"reward": 2.0})
    client.log_artifact_ref(second, "trace.mcap", "sha256:" + "f" * 64)
    client.end_run(first, "FINISHED")
    client.end_run(second, "FAILED")

    inspector = MlflowClient(tracking_uri=uri)
    run_one, run_two = inspector.get_run(first), inspector.get_run(second)
    assert run_one.data.params == {"seed": "1"}
    assert run_two.data.params == {"seed": "2"}
    assert run_one.data.metrics == {}
    assert run_two.data.metrics == {"reward": 2.0}
    assert run_two.data.tags["artifact.trace.mcap"] == "sha256:" + "f" * 64
    assert run_one.info.status == "FINISHED"
    assert run_two.info.status == "FAILED"
