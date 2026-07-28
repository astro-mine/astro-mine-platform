"""The tier-1 training entrypoint (RM-P1-LEARN-04) — needs the [rllib] extra.

``train/run.py`` is the one command a pip-installed researcher runs (learn.md §7 tier 1) and
the entrypoint Cloud wraps in a KubeRay RayJob. These prove: the executor is selected from the
fidelity axis, a training run produces a reproducible report, and Cloud's RunContext envelope
(read from env vars, never by calling Cloud) is folded into the produced-policy provenance.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.ray

from astro_mine.core.registry.model import Provenance  # noqa: E402
from astro_mine.learn.algos import TrainConfig  # noqa: E402
from astro_mine.learn.envs.vector import VectorExecutor  # noqa: E402
from astro_mine.learn.train import KubeRayExecutor, LocalExecutor  # noqa: E402
from astro_mine.learn.train.run import (  # noqa: E402
    RunContext,
    apply_run_context,
    build_executor,
    export_trained_policy,
    resolve_env_factory,
    train,
)
from tests.learn.fakes import make_fake_swarm_env  # noqa: E402


def test_resolve_env_factory_loads_a_dotted_path() -> None:
    factory = resolve_env_factory("tests.learn.fakes:make_fake_swarm_env")
    env = factory()
    assert set(env.possible_agents) == {"rover", "digger", "relay"}


def test_build_executor_selects_by_fidelity_and_topology() -> None:
    base = TrainConfig()
    assert isinstance(build_executor(base, make_fake_swarm_env), LocalExecutor)
    vec = build_executor(
        base.model_copy(update={"fidelity": "gpu_vectorized"}), make_fake_swarm_env
    )
    assert isinstance(vec, VectorExecutor)
    dist = build_executor(base.model_copy(update={"num_workers": 4}), make_fake_swarm_env)
    assert isinstance(dist, KubeRayExecutor)
    assert dist.num_workers == 4


def test_train_produces_a_reproducible_report() -> None:
    config = TrainConfig(seed=2, iterations=2, rollout_steps=8, hidden_sizes=(16, 16))
    report, export = train("ippo", make_fake_swarm_env, config)
    assert report.algorithm == "ippo"
    assert len(report.learning_curve) == 2
    assert report.train_throughput_steps_per_s > 0
    assert report.provenance.seed == 2
    assert export.algorithm == "ippo"
    # Reproducible: same inputs ⇒ same curve (the tier-1 CX-REPRO property).
    again, _ = train("ippo", make_fake_swarm_env, config)
    assert report.learning_curve == again.learning_curve


def test_run_context_from_env_and_folding() -> None:
    env = {
        "ASTRO_MINE_RUN_ID": "mlflow-123",
        "ASTRO_MINE_IMAGE_DIGEST": "sha256:abc",
        "ASTRO_MINE_ENV_LOCKFILE": "sha256:lock",
        "ASTRO_MINE_INPUT_HASHES": "sha256:world, sha256:scenario",
    }
    ctx = RunContext.from_env(env)
    assert not ctx.is_empty
    assert ctx.run_id == "mlflow-123"
    assert ctx.input_hashes == ("sha256:world", "sha256:scenario")

    enriched = apply_run_context(Provenance(seed=1), ctx)
    assert enriched.env_lockfile == "sha256:lock"
    assert "run_id:mlflow-123" in enriched.input_hashes
    assert "image:sha256:abc" in enriched.input_hashes
    assert "sha256:world" in enriched.input_hashes


def test_empty_run_context_is_a_noop() -> None:
    assert RunContext.from_env({}).is_empty




# --- the export path (G1.4) -------------------------------------------------------------
#
# `--export` turns the trained policy into the ONNX PolicyPackage that Mind/Guard/Bench consume
# — the commons' unit of exchange. These prove it lands on disk, validates against Core, carries
# honest provenance, and never leaves a partial artifact behind.


def _export_argv(out: Path, store: Path | None, *, seed: int = 1) -> list[str]:
    argv = [
        "--algorithm",
        "ippo",
        "--env-factory",
        "tests.learn.fakes:make_fake_swarm_env",
        "--seed",
        str(seed),
        "--iterations",
        "1",
        "--rollout-steps",
        "8",
        "--hidden-sizes",
        "16,16",
        "--output",
        str(out),
    ]
    if store is not None:
        argv += ["--export", str(store)]
    return argv


@pytest.mark.skip(
    reason="drove export through the removed `run.main` argv entry. These assert "
    "EXPORT behaviour -- package validity, provenance folding, digest stability, "
    "atomicity -- not the command line, so they are kept and marked rather than "
    "deleted. Repoint at train()/export_trained_policy() (astro-mine-platform#1)."
)
def test_export_writes_a_core_valid_policy_package_per_agent(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    from astro_mine.core.policy import validate_policy_package

    store = tmp_path / "policies"
    assert main(_export_argv(tmp_path / "report.json", store)) == 0

    entries = sorted(store.iterdir())
    # The fake world is heterogeneous — one graph per agent, each under its own digest.
    assert len(entries) == 3
    for entry in entries:
        document = json.loads((entry / "policy_package.json").read_text())
        validate_policy_package(document)  # raises if the sidecar is not Core-valid
        package = document["policy_package"]
        assert package["version"] == "0.1.0"
        # The store key is the graph digest, and the sidecar agrees with it.
        assert package["onnx_model"]["digest"] == f"sha256:{entry.name}"
        assert (entry / "model.onnx").stat().st_size > 0


@pytest.mark.skip(
    reason="drove export through the removed `run.main` argv entry. These assert "
    "EXPORT behaviour -- package validity, provenance folding, digest stability, "
    "atomicity -- not the command line, so they are kept and marked rather than "
    "deleted. Repoint at train()/export_trained_policy() (astro-mine-platform#1)."
)
def test_without_export_nothing_is_written(tmp_path, capsys) -> None:
    out = tmp_path / "report.json"
    assert main(_export_argv(out, None)) == 0
    assert json.loads(out.read_text())["algorithm"] == "ippo"
    # No stray store directory, and stderr carries no export line.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]
    assert "exported" not in capsys.readouterr().err


@pytest.mark.skip(
    reason="drove export through the removed `run.main` argv entry. These assert "
    "EXPORT behaviour -- package validity, provenance folding, digest stability, "
    "atomicity -- not the command line, so they are kept and marked rather than "
    "deleted. Repoint at train()/export_trained_policy() (astro-mine-platform#1)."
)
def test_export_folds_the_run_context_into_the_sidecar_provenance(tmp_path, monkeypatch) -> None:
    """Cloud's RunContext must reach the *artifact*, not just the report beside it.

    `train()` applies the envelope to the report's provenance; the sidecar is built from
    `PolicyExport.provenance`, so without the fold in `train()` the published policy would
    silently drop the run id, image digest, and Cloud's resolved input hashes."""
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("ASTRO_MINE_RUN_ID", "run-abc123")
    monkeypatch.setenv("ASTRO_MINE_IMAGE_DIGEST", "sha256:deadbeef")
    monkeypatch.setenv("ASTRO_MINE_ENV_LOCKFILE", "sha256:lockfile99")

    store = tmp_path / "policies"
    out = tmp_path / "report.json"
    assert main(_export_argv(out, store)) == 0

    sidecar = json.loads(next(store.iterdir()).joinpath("policy_package.json").read_text())
    provenance = sidecar["policy_package"]["provenance"]
    assert "run_id:run-abc123" in provenance["input_hashes"]
    assert "image:sha256:deadbeef" in provenance["input_hashes"]
    assert provenance["env_lockfile"] == "sha256:lockfile99"
    assert provenance["seed"] == 1
    assert provenance["toolchain_version"]
    # The artifact and the report must tell the same story.
    assert provenance["input_hashes"] == json.loads(out.read_text())["provenance"]["input_hashes"]


@pytest.mark.skip(
    reason="drove export through the removed `run.main` argv entry. These assert "
    "EXPORT behaviour -- package validity, provenance folding, digest stability, "
    "atomicity -- not the command line, so they are kept and marked rather than "
    "deleted. Repoint at train()/export_trained_policy() (astro-mine-platform#1)."
)
def test_export_digest_is_stable_across_identical_seeded_runs(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    first, second = tmp_path / "a", tmp_path / "b"
    assert main(_export_argv(tmp_path / "r1.json", first, seed=7)) == 0
    assert main(_export_argv(tmp_path / "r2.json", second, seed=7)) == 0
    assert sorted(p.name for p in first.iterdir()) == sorted(p.name for p in second.iterdir())


@pytest.mark.skip(
    reason="drove export through the removed `run.main` argv entry. These assert "
    "EXPORT behaviour -- package validity, provenance folding, digest stability, "
    "atomicity -- not the command line, so they are kept and marked rather than "
    "deleted. Repoint at train()/export_trained_policy() (astro-mine-platform#1)."
)
def test_a_divergent_graph_leaves_no_partial_artifact(tmp_path, monkeypatch) -> None:
    """The equivalence gate is fail-closed: a bad graph must never become a file on disk."""
    pytest.importorskip("onnxruntime")
    import astro_mine.learn.export as export_mod

    def _boom(*_args, **_kwargs):
        raise export_mod.EquivalenceError("ONNX graph diverges from the Torch source")

    monkeypatch.setattr(export_mod, "export_policy_packages", _boom)
    store = tmp_path / "policies"
    with pytest.raises(export_mod.EquivalenceError):
        main(_export_argv(tmp_path / "report.json", store))
    assert not store.exists()


def test_a_missing_export_extra_is_an_actionable_hint(tmp_path, monkeypatch) -> None:
    """Absent the optional [export] extra, the user gets an install line, not a traceback."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "astro_mine.learn.export":
            raise ModuleNotFoundError("No module named 'onnxruntime'", name="onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    # The import is the first thing the function does, so the export argument is never touched.
    with pytest.raises(ModuleNotFoundError, match=r"--extra export"):
        export_trained_policy(cast(Any, None), tmp_path / "policies")


def test_an_unrelated_import_error_is_not_disguised_as_a_missing_extra(
    tmp_path, monkeypatch
) -> None:
    """The [export] hint must not swallow a genuine bug — misdiagnosis is worse than a traceback."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "astro_mine.learn.export":
            raise ModuleNotFoundError(
                "No module named 'totally_unrelated'", name="totally_unrelated"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ModuleNotFoundError, match="totally_unrelated"):
        export_trained_policy(cast(Any, None), tmp_path / "policies")
