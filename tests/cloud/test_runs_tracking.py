"""RunTracker -- every job is an MLflow run recording its RunContext + artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.cloud.artifacts.runcontext import RunContext
from astro_mine.cloud.runs.tracking import RunTracker, TrackingClient


class FakeTrackingClient:
    """Records tracking calls in memory (stands in for MLflow)."""

    def __init__(self) -> None:
        self.counter = 0
        self.tags: dict[str, dict[str, str]] = {}
        self.params: dict[str, dict[str, str]] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.artifacts: dict[str, dict[str, str]] = {}
        self.ended: dict[str, str] = {}

    def start_run(self, *, experiment: str, tags: Mapping[str, str]) -> str:
        self.counter += 1
        run_id = f"run-{self.counter}"
        self.tags[run_id] = dict(tags)
        return run_id

    def log_params(self, run_id: str, params: Mapping[str, str]) -> None:
        self.params[run_id] = dict(params)

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        self.metrics[run_id] = dict(metrics)

    def log_artifact_ref(self, run_id: str, name: str, address: str) -> None:
        self.artifacts.setdefault(run_id, {})[name] = address

    def end_run(self, run_id: str, status: str) -> None:
        self.ended[run_id] = status


def _context() -> RunContext:
    return RunContext(
        seed=42,
        image_digest="sha256:" + "a" * 64,
        core_interface_version="0.1.0",
        env_lockfile="sha256:" + "b" * 64,
        source_content_hashes={"scenario": "sha256:" + "c" * 64},
        outputs={"trace.mcap": "sha256:" + "d" * 64},
    )


def test_client_satisfies_protocol() -> None:
    assert isinstance(FakeTrackingClient(), TrackingClient)


def test_record_logs_envelope_artifacts_and_stamps_run_id() -> None:
    client = FakeTrackingClient()
    tracker = RunTracker(client)
    context = _context()

    stamped = tracker.record(context, metrics={"reward": 1.5})

    assert stamped.run_id == "run-1"
    assert context.run_id is None  # original is unchanged (a copy is stamped)
    params = client.params["run-1"]
    assert params["seed"] == "42"
    assert params["image_digest"] == context.image_digest
    assert params["core_interface_version"] == "0.1.0"
    assert params["input.scenario"] == "sha256:" + "c" * 64
    assert client.artifacts["run-1"] == {"trace.mcap": "sha256:" + "d" * 64}
    assert client.metrics["run-1"] == {"reward": 1.5}
    assert client.ended["run-1"] == "FINISHED"
    # the run is tagged with the reproducibility pin
    assert client.tags["run-1"]["astro-mine.org/run"] == context.content_address()


def test_rerun_reproduces_from_the_recorded_envelope() -> None:
    client = FakeTrackingClient()
    tracker = RunTracker(client)
    first = tracker.record(_context())
    second = tracker.record(_context())
    # distinct MLflow runs, identical reproducibility pin
    assert first.run_id != second.run_id
    assert first.content_address() == second.content_address()


def test_record_omits_absent_optional_params() -> None:
    client = FakeTrackingClient()
    RunTracker(client).record(RunContext())
    params = client.params["run-1"]
    assert "seed" not in params
    assert "image_digest" not in params
    assert params["schema_version"] == "0.1"
