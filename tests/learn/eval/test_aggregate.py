"""Curve aggregation: Parquet round-trip, content-addressed manifest, sinks (RM-P1-LEARN-06)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from astro_mine.learn.eval.aggregate import (
    CURVE_COLUMNS,
    CURVE_SCHEMA_VERSION,
    CurveRow,
    CurveTable,
    MetricSink,
    MlflowSink,
    ParquetSink,
)


def _rows() -> list[CurveRow]:
    return [
        CurveRow(
            algorithm="ippo",
            policy_id="live:ippo",
            stress_axis="drop",
            stress_value=0.3,
            seed=100,
            episode_return=-0.5,
            delivery_ratio=0.7,
            offered=10,
            delivered=7,
            eval_throughput_steps_per_s=1234.0,
            wall_clock_s=0.01,
            sample_efficiency=None,
            comms_config_hash="sha256:cfg",
        ),
        CurveRow(
            algorithm="ippo",
            policy_id="live:ippo",
            stress_axis="drop",
            stress_value=0.9,
            seed=100,
            episode_return=-0.6,
            delivery_ratio=0.1,
            offered=10,
            delivered=1,
            eval_throughput_steps_per_s=1200.0,
            wall_clock_s=0.01,
            sample_efficiency=2.0,
            comms_config_hash="sha256:cfg2",
        ),
    ]


def _manifest() -> dict[str, object]:
    return {"schema_version": CURVE_SCHEMA_VERSION, "kind": "comms_stress_curve", "seeds": [100]}


def test_curve_table_manifest_hash_is_stable_and_content_addressed() -> None:
    table = CurveTable(_rows(), _manifest())
    assert table.manifest_hash.startswith("sha256:")
    # Same manifest ⇒ same hash; a changed manifest ⇒ a different hash.
    assert CurveTable(_rows(), _manifest()).manifest_hash == table.manifest_hash
    changed = {**_manifest(), "seeds": [999]}
    assert CurveTable(_rows(), changed).manifest_hash != table.manifest_hash


def test_records_carry_all_schema_columns_and_manifest_hash() -> None:
    table = CurveTable(_rows(), _manifest())
    records = table.to_records()
    assert len(records) == 2
    assert set(records[0]) == set(CURVE_COLUMNS)
    assert all(r["manifest_hash"] == table.manifest_hash for r in records)
    assert all(r["schema_version"] == CURVE_SCHEMA_VERSION for r in records)


def test_parquet_round_trip_preserves_schema(tmp_path) -> None:
    table = CurveTable(_rows(), _manifest())
    out = tmp_path / "curve.parquet"
    ParquetSink(out).write(table)
    assert out.exists()
    read = pq.read_table(out)
    assert read.num_rows == 2
    assert list(read.schema.names) == list(CURVE_COLUMNS)
    back = read.to_pylist()
    assert back[0]["algorithm"] == "ippo"
    assert back[0]["delivery_ratio"] == pytest.approx(0.7)
    assert back[0]["sample_efficiency"] is None  # nullable column round-trips
    assert back[1]["sample_efficiency"] == pytest.approx(2.0)
    assert back[0]["manifest_hash"] == table.manifest_hash


def test_parquet_sink_satisfies_the_metric_sink_protocol(tmp_path) -> None:
    assert isinstance(ParquetSink(tmp_path / "c.parquet"), MetricSink)


class _FakeMlflow:
    """A minimal stand-in for the ``mlflow`` module so MlflowSink is exercised without the
    optional [mlflow] extra installed (it stays out of the CI sync).

    Tracks the ``MlflowBackend`` surface the sink now delegates to (learn.md §3's ``track/``
    module ships, so there is exactly one MLflow implementation in the package)."""

    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment: str | None = None
        self.tags: dict[str, str] = {}
        self.dicts: list[tuple[dict, str]] = []
        self.metrics: list[tuple[str, float, int]] = []
        self.runs = 0
        self.ended = 0

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, run_name=None, tags=None):
        self.runs += 1
        self.tags.update(tags or {})
        return SimpleNamespace(run_name=run_name, info=SimpleNamespace(run_id=f"run-{self.runs}"))

    def log_params(self, params: dict) -> None:
        pass

    def log_dict(self, dictionary: dict, artifact_file: str) -> None:
        self.dicts.append((dictionary, artifact_file))

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        self.metrics.extend((key, value, step) for key, value in metrics.items())

    def end_run(self) -> None:
        self.ended += 1


def test_mlflow_sink_logs_manifest_and_metrics(monkeypatch) -> None:
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    sink = MlflowSink(run_name="run-1", tracking_uri="file:///tmp/mlruns", experiment="learn-eval")
    assert isinstance(sink, MetricSink)
    table = CurveTable(_rows(), _manifest())
    sink.write(table)
    assert fake.tracking_uri == "file:///tmp/mlruns"
    assert fake.experiment == "learn-eval"
    assert fake.runs == 1
    assert fake.ended == 1  # the run is closed even though the sink holds no TrackedRun
    assert fake.tags["manifest_hash"] == table.manifest_hash
    assert fake.dicts and fake.dicts[0][1] == "curve_manifest.json"
    # One return + one delivery-ratio metric per row.
    metric_keys = {k for k, _v, _s in fake.metrics}
    assert "ippo.drop.episode_return" in metric_keys
    assert "ippo.drop.delivery_ratio" in metric_keys
    assert len(fake.metrics) == 2 * len(table.rows)


def test_mlflow_sink_fails_loud_without_the_extra(monkeypatch) -> None:
    # With no mlflow installed and none injected, constructing the sink raises loudly.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError):
        MlflowSink()


def test_mlflow_sink_delegates_to_the_single_mlflow_backend(monkeypatch) -> None:
    # There is exactly ONE MLflow implementation now (track/backends.py::MlflowBackend); the sink
    # is a thin adapter over it, not a second copy of the same logic.
    from astro_mine.learn.track.backends import MlflowBackend

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow())
    sink = MlflowSink()
    assert isinstance(sink._backend, MlflowBackend)
