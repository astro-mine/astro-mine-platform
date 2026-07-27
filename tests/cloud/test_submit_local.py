"""LocalBackend + submit(): a real subprocess workload, provenance, determinism."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from astro_mine.cloud import submit
from astro_mine.cloud.artifacts import RunContext, content_address
from astro_mine.cloud.artifacts.store import DEFAULT_ROOT_ENV, FilesystemArtifactStore
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import JobSpec

IMAGE = ImageRef.parse("ghcr.io/astro-mine/astro-mine-bench@sha256:" + "ab" * 32)

# Deterministic workload: y = x * seed, reading/writing via the ASTRO_MINE_* contract.
_WORKLOAD = (
    "import os, pathlib;"
    "i=pathlib.Path(os.environ['ASTRO_MINE_INPUTS']);"
    "o=pathlib.Path(os.environ['ASTRO_MINE_OUTPUTS']);"
    "s=int(os.environ['ASTRO_MINE_SEED']);"
    "x=int((i/'x.txt').read_text());"
    "(o/'y.txt').write_text(str(x*s))"
)


def _job(store: FilesystemArtifactStore, *, seed: int = 7) -> tuple[JobSpec, str]:
    address = store.put(b"6")
    job = JobSpec(
        image=IMAGE,
        command=[sys.executable, "-c", _WORKLOAD],
        inputs={"x.txt": address},
        outputs=["y.txt"],
        seed=seed,
    )
    return job, address


def test_local_run_captures_outputs_and_provenance(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job, input_address = _job(store, seed=7)

    result = submit(job, store=store)

    assert result.ok and result.exit_code == 0
    assert result.outputs["y.txt"] == content_address(b"42")
    assert store.get(result.outputs["y.txt"]) == b"42"

    ctx = result.run_context
    assert ctx.image_digest == job.image.reference
    assert ctx.seed == 7
    assert ctx.source_content_hashes == {"x.txt": input_address}
    assert ctx.outputs == result.outputs
    # the envelope itself is stored and reloads to the same object
    assert RunContext.load(store, result.run_context_address) == ctx


def test_local_run_is_deterministic(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job, _ = _job(store, seed=7)
    first = submit(job, store=store)
    second = submit(job, store=store)
    assert first.outputs == second.outputs
    assert first.run_context_address == second.run_context_address
    assert first.run_context.content_address() == second.run_context.content_address()


def test_run_records_env_lockfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FilesystemArtifactStore(tmp_path)
    lockfile = tmp_path / "uv.lock"
    lockfile.write_bytes(b"version = 1\n")
    monkeypatch.setenv("ASTRO_MINE_ENV_LOCKFILE", str(lockfile))
    job, _ = _job(store, seed=7)

    result = submit(job, store=store)

    # the environment pin is recorded and the lockfile itself is content-addressed + stored
    assert result.run_context.env_lockfile == content_address(b"version = 1\n")
    assert store.get(result.run_context.env_lockfile) == b"version = 1\n"


def test_run_env_lockfile_none_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FilesystemArtifactStore(tmp_path)
    # an explicit override pointing at a missing file records no pin (never a wrong one)
    monkeypatch.setenv("ASTRO_MINE_ENV_LOCKFILE", str(tmp_path / "missing.lock"))
    job, _ = _job(store, seed=7)

    result = submit(job, store=store)

    assert result.run_context.env_lockfile is None


def test_run_env_lockfile_none_when_undiscoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No override and a cwd with no uv.lock up the tree → the pin is left unset.
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    store = FilesystemArtifactStore(store_dir)
    monkeypatch.delenv("ASTRO_MINE_ENV_LOCKFILE", raising=False)
    monkeypatch.chdir(tmp_path)
    job, _ = _job(store, seed=7)

    result = submit(job, store=store)

    assert result.run_context.env_lockfile is None


def test_seed_changes_the_result(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job7, _ = _job(store, seed=7)
    job9, _ = _job(store, seed=9)
    assert submit(job7, store=store).outputs != submit(job9, store=store).outputs


def test_failing_command_reports_failure(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job = JobSpec(image=IMAGE, command=[sys.executable, "-c", "import sys; sys.exit(3)"])
    result = submit(job, store=store)
    assert not result.ok
    assert result.status == "failed"
    assert result.exit_code == 3
    assert result.outputs == {}


def test_missing_declared_output_fails_loud(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job = JobSpec(image=IMAGE, command=[sys.executable, "-c", "pass"], outputs=["y.txt"])
    with pytest.raises(FileNotFoundError, match=r"y\.txt"):
        submit(job, store=store)


def test_empty_command_is_rejected(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    with pytest.raises(ValueError, match=r"non-empty job\.command"):
        submit(JobSpec(image=IMAGE), store=store)


def test_default_store_is_local_and_accountless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DEFAULT_ROOT_ENV, str(tmp_path))
    store = FilesystemArtifactStore(tmp_path)
    job, _ = _job(store, seed=7)
    result = submit(job)  # no store -> default FilesystemArtifactStore honoring the env
    assert result.ok
    assert FilesystemArtifactStore(tmp_path).get(result.outputs["y.txt"]) == b"42"


def test_unknown_backend_raises(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job, _ = _job(store)
    with pytest.raises(ValueError, match="unknown backend 'nope'"):
        submit(job, backend="nope", store=store)
