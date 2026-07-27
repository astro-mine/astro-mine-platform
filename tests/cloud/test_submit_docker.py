"""DockerBackend: argv construction, simulated run, and an opt-in real-Docker test."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from astro_mine.cloud.artifacts import content_address
from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import JobSpec
from astro_mine.cloud.submission.docker import DockerBackend, build_docker_argv

IMAGE = ImageRef.parse("ghcr.io/astro-mine/astro-mine-bench@sha256:" + "ab" * 32)
_WORKLOAD = (
    "import os, pathlib;"
    "i=pathlib.Path(os.environ['ASTRO_MINE_INPUTS']);"
    "o=pathlib.Path(os.environ['ASTRO_MINE_OUTPUTS']);"
    "s=int(os.environ['ASTRO_MINE_SEED']);"
    "x=int((i/'x.txt').read_text());"
    "(o/'y.txt').write_text(str(x*s))"
)


def test_argv_pins_image_mounts_and_seed(tmp_path: Path) -> None:
    job = JobSpec(image=IMAGE, command=["run", "--flag"], seed=7)
    argv = build_docker_argv(job, tmp_path / "in", tmp_path / "out")

    assert argv[:4] == ["docker", "run", "--rm", "--network=none"]
    assert f"{tmp_path / 'in'}:/inputs:ro" in argv
    assert f"{tmp_path / 'out'}:/outputs" in argv
    assert "ASTRO_MINE_SEED=7" in argv
    # the digest-pinned reference precedes the command, which comes last
    idx = argv.index(IMAGE.reference)
    assert argv[idx + 1 :] == ["run", "--flag"]


def test_docker_backend_runs_via_simulated_runner(
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
    result = DockerBackend(runner=docker_simulator).run(job, store=store)
    assert result.ok
    assert result.outputs["y.txt"] == content_address(b"42")


def test_docker_backend_rejects_empty_command(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    backend = DockerBackend(runner=lambda _argv: 0)
    with pytest.raises(ValueError, match=r"non-empty job\.command"):
        backend.run(JobSpec(image=IMAGE), store=store)


@pytest.mark.docker
@pytest.mark.skipif(
    not (os.environ.get("ASTRO_MINE_DOCKER_IMAGE") and shutil.which("docker")),
    reason="set ASTRO_MINE_DOCKER_IMAGE (a digest-pinned image with /bin/sh) and install Docker",
)
def test_real_docker_round_trip(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job = JobSpec(
        image=ImageRef.parse(os.environ["ASTRO_MINE_DOCKER_IMAGE"]),
        command=["/bin/sh", "-c", 'printf hi > "$ASTRO_MINE_OUTPUTS/y.txt"'],
        outputs=["y.txt"],
    )
    result = DockerBackend().run(job, store=store)
    assert result.ok
    assert store.get(result.outputs["y.txt"]) == b"hi"
