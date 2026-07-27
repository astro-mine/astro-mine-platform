"""Experiment tracking + provenance capture (learn.md §3 ``track/``; §4, §11).

learn.md §11 makes **MLflow the default** tracking store; learn.md §2.4 says what tracking is
*for*: "Every run records its inputs, code version, lockfile, and seeds; results are
content-addressed so Bench can re-derive them."

So the tests are about the **reproducibility record**, not about dashboards:

- the run record captures the TrainConfig, the comms regime, the curriculum, the seeds, the
  toolchain and the lockfile, and **content-addresses** them into ``run_hash``;
- the same inputs hash the same, and *any* changed input changes the hash — otherwise the key is
  a lie and Bench could "re-derive" a different experiment;
- curves and the produced policy's ONNX digests land in that same record (learn.md §4: "runs link
  to Bench results and Hub artifacts by content hash");
- the MLflow backend is exercised **without MLflow installed** (it is a conflicting optional
  extra, deliberately absent from the CI sync) by injecting a fake ``mlflow`` module — the same
  pattern ``tests/eval/test_aggregate.py`` established for the sink.

Torch-free.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.curriculum import comms_ladder
from astro_mine.learn.envs.comms import CommsModelConfig, DropConfig
from astro_mine.learn.eval.aggregate import CurveRow, CurveTable
from astro_mine.learn.track import (
    RUN_RECORD_VERSION,
    InMemoryBackend,
    MlflowBackend,
    TrackedRun,
    TrackingBackend,
    run_provenance,
    tracked_run,
)

_CFG = TrainConfig(seed=13, iterations=3, rollout_steps=16)
_COMMS = CommsModelConfig(drop=DropConfig(probability=0.25))


# --- the provenance record -----------------------------------------------------------


def test_provenance_captures_config_seeds_comms_curriculum_and_toolchain() -> None:
    record = run_provenance(
        _CFG,
        algorithm="mappo",
        comms=_COMMS,
        curriculum=comms_ladder(),
        env_lockfile="sha256:lock",
    )
    assert record["record_version"] == RUN_RECORD_VERSION
    assert record["algorithm"] == "mappo"
    assert record["seed"] == 13
    assert record["train_config"]["rollout_steps"] == 16
    assert record["comms"]["drop"]["probability"] == 0.25
    assert record["curriculum"]["name"] == "comms_ladder"
    assert record["env_lockfile"] == "sha256:lock"
    # The toolchain the run was pinned to (conventions.md §5) — at minimum the package itself.
    # Core ships inside the platform distribution now — one entry covers both.
    assert "astro-mine-platform" in record["toolchain"]


def test_the_run_hash_is_the_reproducibility_key() -> None:
    # Same inputs ⇒ same key. Two runs with this hash ARE the same experiment and must produce
    # the same learning curve — that claim is exactly what the CX-REPRO determinism gate asserts.
    a = TrackedRun(_CFG, algorithm="mappo", comms=_COMMS)
    b = TrackedRun(_CFG, algorithm="mappo", comms=_COMMS)
    assert a.run_hash == b.run_hash
    assert a.run_hash.startswith("sha256:")


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"config": TrainConfig(seed=14, iterations=3, rollout_steps=16)}, "a different seed"),
        ({"config": TrainConfig(seed=13, iterations=3, rollout_steps=32)}, "a different config"),
        ({"algorithm": "ippo"}, "a different algorithm"),
        ({"comms": CommsModelConfig(drop=DropConfig(probability=0.9))}, "a different channel"),
        ({"curriculum": comms_ladder()}, "a curriculum was added"),
        ({"env_lockfile": "sha256:other"}, "a different lockfile"),
    ],
)
def test_any_changed_input_changes_the_run_hash(kwargs: dict[str, Any], why: str) -> None:
    # If a changed input did NOT change the key, Bench could "re-derive" a different experiment
    # and believe it reproduced the first. Every input must be inside the hash.
    base = TrackedRun(_CFG, algorithm="mappo", comms=_COMMS)
    config = kwargs.pop("config", _CFG)
    changed = TrackedRun(config, **{"algorithm": "mappo", "comms": _COMMS, **kwargs})
    assert changed.run_hash != base.run_hash, why


# --- the tracked run -----------------------------------------------------------------


def test_in_memory_backend_needs_no_server_and_records_everything() -> None:
    # learn.md §7 tier 1 "MUST always work": a workstation run gets its full provenance record
    # with no MLflow, no Postgres, and no network.
    backend = InMemoryBackend()
    assert isinstance(backend, TrackingBackend)

    with TrackedRun(_CFG, algorithm="mappo", comms=_COMMS, backend=backend) as run:
        assert run.run_id == "in-memory"
        run.log_iteration({"mean_reward": -0.1, "policy_loss": 0.5})
        run.log_iteration({"mean_reward": -0.05, "policy_loss": 0.4})

    assert backend.ended is True
    assert backend.name == "mappo-13"
    assert backend.tags["run_hash"] == run.run_hash
    assert backend.tags["algorithm"] == "mappo"
    # The immutable inputs are params; the full documents are artifacts.
    assert backend.params["seed"] == 13
    assert backend.params["run_hash"] == run.run_hash
    assert backend.artifacts["run_provenance.json"]["comms"]["drop"]["probability"] == 0.25
    # The learning curve, in order.
    assert backend.curve("mean_reward") == [-0.1, -0.05]
    assert backend.curve("policy_loss") == [0.5, 0.4]


def test_the_run_closes_even_when_training_raises() -> None:
    backend = InMemoryBackend()
    with pytest.raises(RuntimeError, match="boom"), TrackedRun(_CFG, backend=backend):
        raise RuntimeError("boom")
    assert backend.ended is True  # a crashed run is still a closed run


def test_tracked_run_context_helper() -> None:
    backend = InMemoryBackend()
    with tracked_run(_CFG, algorithm="ippo", backend=backend) as run:
        run.log_iteration({"mean_reward": 1.0})
    assert backend.ended and backend.curve("mean_reward") == [1.0]


def test_eval_curves_land_in_the_same_run_as_the_config() -> None:
    # The point of superseding the standalone MlflowSink: the eval numbers and the run that
    # produced them are ONE record, not two that must be joined by hand afterwards.
    backend = InMemoryBackend()
    rows = [
        CurveRow(
            algorithm="mappo",
            policy_id="live:mappo",
            stress_axis="drop",
            stress_value=0.3,
            seed=1,
            episode_return=-0.4,
            delivery_ratio=0.7,
            offered=10,
            delivered=7,
            eval_throughput_steps_per_s=100.0,
            wall_clock_s=0.1,
            sample_efficiency=None,
            comms_config_hash="sha256:cfg",
        )
    ]
    table = CurveTable(rows, {"kind": "comms_stress_curve", "seeds": [1]})
    with TrackedRun(_CFG, algorithm="mappo", backend=backend) as run:
        run.log_curve(table)

    assert backend.artifacts["curve_manifest.json"] == table.manifest
    assert backend.curve("mappo.drop.episode_return") == [-0.4]
    assert backend.curve("mappo.drop.delivery_ratio") == [0.7]


def test_the_produced_policy_is_linked_to_the_run_by_content_hash() -> None:
    # learn.md §4: "Runs link to Bench results and Hub artifacts by content hash." This is that
    # link: the ONNX graph digest Bench scores, recorded against the run that trained it.
    from astro_mine.core.registry.model import Provenance
    from astro_mine.learn.algos import IoSignature, PolicyAssumptions, PolicyExport

    export = PolicyExport(
        algorithm="mappo",
        backend="torch",
        io_signature=IoSignature(agent_ids=("rover",), per_agent={}),
        assumptions=PolicyAssumptions(
            comms_observability={"kind": "comms_model"},
            surrogate_fidelity_caveats=("needs a high-fidelity pass",),
        ),
        provenance=Provenance(seed=13),
        metrics={"mean_reward": -0.05},
    )
    backend = InMemoryBackend()
    with TrackedRun(_CFG, algorithm="mappo", backend=backend) as run:
        run.log_export(export, digests={"rover": "sha256:aaa", "digger": "sha256:bbb"})

    linked = backend.artifacts["policy_export.json"]
    assert linked["onnx_digests"] == {"rover": "sha256:aaa", "digger": "sha256:bbb"}
    assert linked["surrogate_fidelity_caveats"] == ["needs a high-fidelity pass"]
    assert linked["comms_observability"] == {"kind": "comms_model"}
    assert run.artifact_digests == {"rover": "sha256:aaa", "digger": "sha256:bbb"}


# --- the MLflow backend (exercised without MLflow installed) --------------------------


class _FakeMlflow:
    """A minimal stand-in for the ``mlflow`` module, so MlflowBackend is exercised without the
    optional [mlflow] extra (which conflicts with [rllib] and is never in the CI sync)."""

    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment: str | None = None
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[str, float, int]] = []
        self.dicts: dict[str, dict[str, Any]] = {}
        self.tags: dict[str, str] = {}
        self.run_name: str | None = None
        self.started = 0
        self.ended = 0

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, run_name=None, tags=None):
        self.started += 1
        self.run_name = run_name
        self.tags.update(tags or {})

        class _Run:
            info = type("Info", (), {"run_id": "mlflow-run-1"})()

        return _Run()

    def log_params(self, params) -> None:
        self.params.update(params)

    def log_metrics(self, metrics, step: int = 0) -> None:
        self.metrics.extend((k, v, step) for k, v in metrics.items())

    def log_dict(self, dictionary, artifact_file: str) -> None:
        self.dicts[artifact_file] = dictionary

    def end_run(self) -> None:
        self.ended += 1


def test_mlflow_backend_is_the_default_store(monkeypatch) -> None:
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    backend = MlflowBackend(tracking_uri="file:///tmp/mlruns", experiment="learn")
    assert isinstance(backend, TrackingBackend)
    assert fake.tracking_uri == "file:///tmp/mlruns"
    assert fake.experiment == "learn"

    with TrackedRun(_CFG, algorithm="mappo", comms=_COMMS, backend=backend) as run:
        assert run.run_id == "mlflow-run-1"  # the backend-assigned id flows back
        run.log_iteration({"mean_reward": -0.2})

    # The run is held OPEN across the loop, so the curve lands in the SAME run as the config.
    assert fake.started == 1 and fake.ended == 1
    assert fake.tags["run_hash"] == run.run_hash
    assert fake.params["seed"] == 13
    assert fake.dicts["run_provenance.json"]["algorithm"] == "mappo"
    assert ("mean_reward", -0.2, 0) in fake.metrics


def test_mlflow_backend_fails_loud_without_the_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError):
        MlflowBackend()
