"""Shared test fixtures."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence

import pytest


@pytest.fixture
def docker_simulator() -> Callable[[Sequence[str]], int]:
    """A DockerBackend runner that *simulates* a container without Docker.

    It parses the ``docker run`` argv the backend built, translates the in-container
    ``/inputs`` and ``/outputs`` paths back to the host bind-mount directories, and runs
    the inner command locally -- so the docker code path (argv + mounts + env) is
    exercised in CI while producing real outputs for equivalence checks.
    """

    def run(argv: Sequence[str]) -> int:
        mounts: dict[str, str] = {}
        env: dict[str, str] = {}
        image_index: int | None = None
        i = 0
        while i < len(argv):
            token = argv[i]
            if token == "-v":
                host, _, rest = argv[i + 1].partition(":")
                mounts[rest.split(":")[0]] = host
                i += 2
            elif token == "-e":
                key, _, value = argv[i + 1].partition("=")
                env[key] = value
                i += 2
            elif token in {"docker", "run", "--rm", "--network=none"}:
                i += 1
            else:
                image_index = i
                break
        assert image_index is not None, "no image reference found in docker argv"

        host_env = dict(os.environ)
        for key, value in env.items():
            host_env[key] = mounts.get(value, value)  # /inputs -> host, seed stays as-is
        inner = list(argv[image_index + 1 :])
        return subprocess.run(inner, env=host_env, check=False).returncode

    return run
