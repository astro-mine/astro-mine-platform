"""The in-container run harness -- hermetic (this is the code that runs *inside* a pod).

No cluster, no container: the harness is just "read a JobSpec from the env, run it through the
shared `execute()`, print two sentinels". Every one of those steps is exercised here, so what a
live cluster adds is only the pod around it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from astro_mine.cloud.artifacts.s3 import S3ArtifactStore
from astro_mine.cloud.artifacts.store import DEFAULT_ROOT_ENV, FilesystemArtifactStore
from astro_mine.cloud.engines.base import workload_env
from astro_mine.cloud.k8s import ENV_JOBSPEC
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import JobSpec, submit
from astro_mine.cloud.submission.harness import (
    EXIT_CODE_SENTINEL,
    RUN_CONTEXT_SENTINEL,
    S3_BUCKET_VAR,
    S3_ENDPOINT_VAR,
    build_store,
    main,
    parse_sentinels,
    run,
)

IMAGE = ImageRef.parse("ghcr.io/astro-mine/astro-mine-bench@sha256:" + "cd" * 32)

# The same deterministic workload the local-backend tests use: y = x * seed, over the
# ASTRO_MINE_{INPUTS,OUTPUTS,SEED} contract -- which the harness sets up exactly as the local
# backend does, because it *is* the local backend's launcher.
_WORKLOAD = (
    "import os, pathlib;"
    "i=pathlib.Path(os.environ['ASTRO_MINE_INPUTS']);"
    "o=pathlib.Path(os.environ['ASTRO_MINE_OUTPUTS']);"
    "s=int(os.environ['ASTRO_MINE_SEED']);"
    "x=int((i/'x.txt').read_text());"
    "(o/'y.txt').write_text(str(x*s))"
)


def _job(store: FilesystemArtifactStore) -> JobSpec:
    return JobSpec(
        image=IMAGE,
        command=[sys.executable, "-c", _WORKLOAD],
        inputs={"x.txt": store.put(b"6")},
        outputs=["y.txt"],
        seed=7,
    )


# --- the sentinel protocol -----------------------------------------------------------------


def test_parse_sentinels_reads_the_address_and_exit_code() -> None:
    logs = f"noise\n{RUN_CONTEXT_SENTINEL}sha256:{'ab' * 32}\n{EXIT_CODE_SENTINEL}0\nmore noise\n"
    assert parse_sentinels(logs) == (f"sha256:{'ab' * 32}", 0)


def test_parse_sentinels_takes_the_last_occurrence() -> None:
    """A container whose attempts streamed into one log reports the state it ended in."""
    logs = (
        f"{RUN_CONTEXT_SENTINEL}sha256:{'11' * 32}\n{EXIT_CODE_SENTINEL}1\n"
        f"{RUN_CONTEXT_SENTINEL}sha256:{'22' * 32}\n{EXIT_CODE_SENTINEL}0\n"
    )
    assert parse_sentinels(logs) == (f"sha256:{'22' * 32}", 0)


def test_parse_sentinels_reports_a_nonzero_exit_code() -> None:
    logs = f"{RUN_CONTEXT_SENTINEL}sha256:{'ab' * 32}\n{EXIT_CODE_SENTINEL}3\n"
    assert parse_sentinels(logs) == (f"sha256:{'ab' * 32}", 3)


@pytest.mark.parametrize(
    "logs",
    [
        pytest.param("", id="empty (a pod killed before it printed anything)"),
        pytest.param("kubectl: no such pod\n", id="not the harness at all"),
        pytest.param(f"{RUN_CONTEXT_SENTINEL}sha256:{'ab' * 32}\n", id="no exit code"),
        pytest.param(f"{EXIT_CODE_SENTINEL}0\n", id="no run context"),
        pytest.param(f"{RUN_CONTEXT_SENTINEL}\n{EXIT_CODE_SENTINEL}0\n", id="empty address"),
        pytest.param(
            f"{RUN_CONTEXT_SENTINEL}sha256:{'ab' * 32}\n{EXIT_CODE_SENTINEL}oops\n",
            id="unparsable exit code",
        ),
    ],
)
def test_parse_sentinels_is_none_when_the_run_did_not_report(logs: str) -> None:
    assert parse_sentinels(logs) is None


# --- the store the pod writes to -----------------------------------------------------------


def test_build_store_defaults_to_the_filesystem(tmp_path: Path) -> None:
    store = build_store({DEFAULT_ROOT_ENV: str(tmp_path)})
    assert isinstance(store, FilesystemArtifactStore)
    assert store.root == tmp_path


def test_build_store_selects_s3_when_a_bucket_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pod points at MinIO/S3 purely through job.env -- no engine change to wire it up.

    boto3 picks the credentials/region up from the ambient ``AWS_*`` environment, which
    ``workload_env()`` already forwards from ``job.env`` into the container.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    store = build_store(
        {
            S3_BUCKET_VAR: "astro-mine",
            S3_ENDPOINT_VAR: "http://minio.astro-mine-system.svc.cluster.local:9000",
        }
    )
    assert isinstance(store, S3ArtifactStore)
    assert store.bucket == "astro-mine"


# --- the run itself --------------------------------------------------------------------------


def test_harness_run_is_byte_identical_to_the_local_backend(tmp_path: Path) -> None:
    """The determinism gate, minus the cluster.

    The harness calls the *same* ``execute()`` the local backend calls, so its outputs and its
    RunContext content-address are equal to a local run's **by construction**. A live cluster
    changes only where the process runs -- which is exactly why the cluster equivalence test is
    meaningful rather than tautological.
    """
    store = FilesystemArtifactStore(tmp_path)
    job = _job(store)

    local = submit(job, store=store)
    in_pod = run(job, store)

    assert in_pod.outputs == local.outputs
    assert in_pod.run_context.content_address() == local.run_context.content_address()
    assert in_pod.run_context_address == local.run_context_address


def test_main_runs_the_jobspec_from_the_env_and_prints_the_sentinels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job = _job(store)
    monkeypatch.setenv(DEFAULT_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(ENV_JOBSPEC, job.model_dump_json())

    assert main() == 0

    collected = parse_sentinels(capsys.readouterr().out)
    assert collected is not None
    address, exit_code = collected
    assert exit_code == 0
    # the envelope the pod printed the address of is really in the shared store
    assert store.exists(address)
    assert json.loads(store.get(address))["outputs"]["y.txt"] == store.put(b"42")


def test_main_consumes_exactly_what_the_engines_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The compile side and the runtime side agree: ``workload_env()`` is the pod's whole env."""
    store = FilesystemArtifactStore(tmp_path)
    job = _job(store)
    monkeypatch.setenv(DEFAULT_ROOT_ENV, str(tmp_path))
    for key, value in workload_env(job).items():
        monkeypatch.setenv(key, value)

    assert main() == 0
    assert parse_sentinels(capsys.readouterr().out) is not None


def test_main_exits_with_the_workloads_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pod's failure is the job's failure -- that is what a Job's Failed condition means."""
    job = JobSpec(image=IMAGE, command=[sys.executable, "-c", "import sys; sys.exit(3)"])
    monkeypatch.setenv(DEFAULT_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(ENV_JOBSPEC, job.model_dump_json())

    assert main() == 3

    collected = parse_sentinels(capsys.readouterr().out)
    assert collected is not None
    assert collected[1] == 3


def test_main_without_a_jobspec_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(ENV_JOBSPEC, raising=False)
    assert main() == 2
    assert ENV_JOBSPEC in capsys.readouterr().err
