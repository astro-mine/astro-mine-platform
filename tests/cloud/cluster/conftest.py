"""Fixtures for the opt-in live-cluster tests.

Every module in this package is marked ``cluster`` **and** self-skips when
``ASTRO_MINE_CLUSTER_KUBECONFIG`` is unset, mirroring how the ``minio`` / ``nats`` / ``mlflow``
tests gate on their own endpoint variables. There is no default ``-m`` deselection in
``pyproject.toml``, so a laptop's ``uv run pytest`` must *skip* these, never error.

Stand the environment up with::

    ./platform/kind/up.sh                 # kind + local registry + MinIO + the umbrella chart
    set -a && . ./platform/kind/harness.env && set +a
    uv run pytest -m cluster

and tear it down with ``./platform/kind/down.sh``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest

from astro_mine.cloud.artifacts.s3 import S3ArtifactStore
from astro_mine.cloud.packaging import ImageRef

#: The kubeconfig `up.sh` writes. Its presence is what "a cluster is available" means.
KUBECONFIG_VAR = "ASTRO_MINE_CLUSTER_KUBECONFIG"
#: The digest-pinned workload image `up.sh` built and pushed -- entrypoint: the run harness.
IMAGE_VAR = "ASTRO_MINE_WORKLOAD_IMAGE"
#: MinIO as the *host* reaches it (a kind node port) vs as a *pod* reaches it (cluster DNS).
HOST_ENDPOINT_VAR = "ASTRO_MINE_S3_ENDPOINT_HOST"
POD_ENDPOINT_VAR = "ASTRO_MINE_S3_ENDPOINT"
BUCKET_VAR = "ASTRO_MINE_S3_BUCKET"

#: Namespace the dispatch tests submit into (created by `up.sh`).
NAMESPACE = "astro-mine-runs"

#: Applied to every module in this package.
CLUSTER_MARKS = [
    pytest.mark.cluster,
    pytest.mark.skipif(
        not os.environ.get(KUBECONFIG_VAR),
        reason=f"set {KUBECONFIG_VAR} (run ./platform/kind/up.sh) to exercise a live cluster",
    ),
]


def requires(*binaries: str) -> pytest.MarkDecorator:
    """Skip unless every one of *binaries* is on PATH (cosign / opa are not always installed)."""
    missing = [b for b in binaries if shutil.which(b) is None]
    return pytest.mark.skipif(bool(missing), reason=f"needs {', '.join(missing)} on PATH")


class Kubectl:
    """A thin ``kubectl`` wrapper bound to the harness kubeconfig."""

    def __init__(self, kubeconfig: str) -> None:
        self._env = {**os.environ, "KUBECONFIG": kubeconfig}

    def __call__(
        self, *args: str, stdin: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["kubectl", *args],
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
            env=self._env,
            timeout=300,
        )
        if check and result.returncode != 0:
            # Raise with kubectl's *own words*. A bare CalledProcessError says only "exit status 1"
            # and hides stderr, where the admission rejection actually is -- so "the API server
            # refused this pod, and here is the policy that refused it" arrives as an opaque
            # traceback about subprocess.py. The message is the entire point of the test.
            raise AssertionError(
                f"kubectl {' '.join(args)} failed (exit {result.returncode}):\n"
                f"{(result.stderr or result.stdout).strip()}"
            )
        return result

    def json(self, *args: str) -> dict:  # type: ignore[type-arg]
        """``kubectl get -o json`` as a dict."""
        result: dict = json.loads(self(*args, "-o", "json").stdout)  # type: ignore[type-arg]
        return result

    def apply(self, manifest: str) -> None:
        self("apply", "-f", "-", stdin=manifest)

    def wait_until(
        self, predicate: Callable[[], bool], *, timeout: float = 300.0, interval: float = 3.0
    ) -> None:
        """Poll *predicate* until true, or fail the test with a timeout."""
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                pytest.fail(f"condition not met within {timeout}s")
            time.sleep(interval)


@pytest.fixture(scope="session")
def kubeconfig() -> str:
    return os.environ[KUBECONFIG_VAR]


@pytest.fixture(scope="session")
def kubectl(kubeconfig: str) -> Kubectl:
    return Kubectl(kubeconfig)


@pytest.fixture(scope="session")
def workload_image() -> ImageRef:
    """The digest-pinned workload image. Digest-pinned is not optional: an unpinned tag would
    make ``RunContext.image_digest`` -- part of the content address -- meaningless."""
    return ImageRef.parse(os.environ[IMAGE_VAR])


@pytest.fixture(scope="session")
def bucket() -> str:
    return os.environ.get(BUCKET_VAR, "astro-mine")


@pytest.fixture(scope="session")
def pod_store_env() -> dict[str, str]:
    """The ``job.env`` that points a *pod* at the shared store (cluster-internal endpoint).

    ``workload_env()`` forwards ``job.env`` into the container verbatim, and the harness builds
    its store from exactly these variables -- so no engine change is needed to wire a pod up to
    MinIO (``cloud.md`` §5).
    """
    return {
        BUCKET_VAR: os.environ.get(BUCKET_VAR, "astro-mine"),
        POD_ENDPOINT_VAR: os.environ[POD_ENDPOINT_VAR],
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }


@pytest.fixture(scope="session")
def store(bucket: str) -> S3ArtifactStore:
    """The same store the pods use, reached from the host through the node port.

    One store, two endpoints -- that is the *point*: the host writes the job's inputs into it and
    reads the pod's outputs and provenance back out of it.
    """
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ[HOST_ENDPOINT_VAR],
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    return S3ArtifactStore(bucket, client=client)


@pytest.fixture(scope="session")
def repo_lockfile() -> Path:
    """The repo's ``uv.lock`` -- the environment pin the workload image ships the same bytes of.

    ``env_lockfile`` is inside the RunContext content address (``conventions.md`` §5), so the
    host half of an equivalence test must pin the *same* lockfile the image does. Pointing at it
    explicitly beats relying on a CWD walk-up that a pytest rootdir change would silently break.
    """
    return Path(__file__).resolve().parents[3] / "uv.lock"


@pytest.fixture
def temp_namespace(kubectl: Kubectl) -> Iterator[str]:
    """A throwaway namespace, deleted afterwards."""
    name = f"amc-test-{uuid.uuid4().hex[:8]}"
    kubectl("create", "namespace", name)
    try:
        yield name
    finally:
        kubectl("delete", "namespace", name, "--wait=false", check=False)


def run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Run a host command (docker / cosign / opa), capturing output."""
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=600,
        **kwargs,  # type: ignore[arg-type]
    )
