"""The backend-equivalence contract: local and docker produce identical results.

This is the heart of RM-P0-CLOUD-02 -- the same JobSpec, run by different backends, yields
identical content-addressed outputs and an identical run-context. Proven here local vs a
simulated container; the real-Docker equivalence is opt-in in ``test_submit_docker.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import JobSpec, registered_backends
from astro_mine.cloud.submission.docker import DockerBackend
from astro_mine.cloud.submission.local import LocalBackend

IMAGE = ImageRef.parse("ghcr.io/astro-mine/astro-mine-bench@sha256:" + "ab" * 32)
_WORKLOAD = (
    "import os, pathlib;"
    "i=pathlib.Path(os.environ['ASTRO_MINE_INPUTS']);"
    "o=pathlib.Path(os.environ['ASTRO_MINE_OUTPUTS']);"
    "s=int(os.environ['ASTRO_MINE_SEED']);"
    "x=int((i/'x.txt').read_text());"
    "(o/'y.txt').write_text(str(x*s))"
)


def test_local_and_docker_backends_are_equivalent(
    tmp_path: Path, docker_simulator: Callable[[Sequence[str]], int]
) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job = JobSpec(
        image=IMAGE,
        command=[sys.executable, "-c", _WORKLOAD],
        inputs={"x.txt": store.put(b"6")},
        outputs=["y.txt"],
        seed=7,
    )

    local = LocalBackend().run(job, store=store)
    docker = DockerBackend(runner=docker_simulator).run(job, store=store)

    assert local.outputs == docker.outputs
    assert local.run_context.content_address() == docker.run_context.content_address()
    assert local.run_context_address == docker.run_context_address


def test_builtin_backends_are_registered() -> None:
    assert {"local", "docker"} <= set(registered_backends())


def test_register_backend_rejects_duplicates_and_allows_replace() -> None:
    from astro_mine.cloud.submission.backend import get_backend, register_backend

    backend = LocalBackend()
    register_backend("equivalence-dup", backend)
    with pytest.raises(ValueError, match="already registered"):
        register_backend("equivalence-dup", backend)
    register_backend("equivalence-dup", backend, replace=True)
    assert get_backend("equivalence-dup") is backend
