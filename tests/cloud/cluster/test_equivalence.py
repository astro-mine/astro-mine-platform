"""The determinism gate: a cluster run reproduces the laptop run, byte for byte.

This is the Phase-1 exit criterion the gap analysis called empirically undemonstrated, and the
one assertion in this repo that could not be faked. Everything else -- the manifest builders, the
in-process quota model, ``DryRunClient`` -- either never leaves the workstation or never runs at
all. Here the *same* ``JobSpec`` goes through ``submit()`` twice: once as a subprocess on this
machine, once as a pod on a real Kubernetes cluster, in a container, staging its inputs from an
object store. The two must agree on:

  - ``outputs``            -- the content addresses of the bytes the workload produced, and
  - ``run_context.content_address()`` -- the whole reproducibility envelope
    (``conventions.md`` §5): source hashes, seed, code version, env lockfile, image digest.

They agree by *construction*, not by coincidence: the in-pod harness calls the same ``execute()``
the local backend does. Which is exactly why this test is worth running -- it fails loudly if the
construction is ever broken.

Two fields in that envelope are hostages to the *image build*, and both are pinned deliberately:

  - ``code_version`` is the installed wheel version, which hatch-vcs derives from ``git
    describe``. The image is built with ``SETUPTOOLS_SCM_PRETEND_VERSION`` set to the host's
    version (``platform/kind/up.sh``), so the two match.
  - ``env_lockfile`` is the content address of the active ``uv.lock``. The image COPYs the repo's
    lockfile to its WORKDIR; the host half pins the same file explicitly below.

Neither is excluded from the content address to make this pass -- excluding them would gut the
reproducibility contract. A test that is sensitive to them is the harness working correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.cloud.artifacts.s3 import S3ArtifactStore
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import ClusterBackend, JobSpec, submit
from astro_mine.cloud.submission._run import ENV_LOCKFILE_VAR
from astro_mine.cloud.submission.backend import register_backend
from tests.cloud.cluster import workloads
from tests.cloud.cluster.conftest import CLUSTER_MARKS, NAMESPACE

pytestmark = CLUSTER_MARKS

SEED = 7
X = 6


@pytest.fixture(scope="module", autouse=True)
def _cluster_backend() -> None:
    register_backend("live", ClusterBackend(namespace=NAMESPACE), replace=True)


def test_a_cluster_run_reproduces_the_laptop_run(
    workload_image: ImageRef,
    store: S3ArtifactStore,
    pod_store_env: dict[str, str],
    repo_lockfile: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin the host's env lockfile to the very bytes the image ships, so `env_lockfile` compares
    # like for like rather than depending on where pytest happened to be invoked from.
    monkeypatch.setenv(ENV_LOCKFILE_VAR, str(repo_lockfile))

    job = JobSpec(
        image=workload_image,
        command=["python", "-c", workloads.DETERMINISTIC],
        env=pod_store_env,
        inputs={"x.txt": store.put(str(X).encode())},
        outputs=["y.txt"],
        seed=SEED,
    )

    local = submit(job, store=store)  # a subprocess, here
    cluster = submit(job, backend="live", store=store)  # a pod, over there

    assert local.ok and cluster.ok

    # 1. The bytes. Same content address => the workload produced identical output.
    assert cluster.outputs == local.outputs
    assert store.get(cluster.outputs["y.txt"]) == workloads.deterministic_output(X, SEED)

    # 2. The provenance. Same content address => the whole reproducibility envelope agrees.
    assert cluster.run_context.content_address() == local.run_context.content_address()

    # 3. ...and the agreement is field-for-field across the whole deterministic core, not just a
    #    matching digest.
    #
    #    Note what is NOT asserted: that the stored envelopes are byte-identical. They are not, and
    #    must not be. RunContext carries an `environment` stamp (python, platform) and an MLflow
    #    `run_id`, both deliberately *outside* the determinism set -- `content_address()` excludes
    #    them by name, and EnvironmentFingerprint calls itself "recorded, but outside the
    #    determinism set". A pod and a laptop legitimately differ there. Requiring
    #    `run_context_address` to match (as this once did) would assert the run had never left the
    #    workstation -- the opposite of what this test exists to prove.
    assert cluster.run_context.model_dump(
        mode="json", exclude={"environment", "run_id"}
    ) == local.run_context.model_dump(mode="json", exclude={"environment", "run_id"})

    # If (2) ever fails, these narrow it down to the field at fault instead of leaving a bare
    # digest mismatch to debug.
    assert cluster.run_context.code_version == local.run_context.code_version
    assert cluster.run_context.env_lockfile == local.run_context.env_lockfile
    assert cluster.run_context.image_digest == local.run_context.image_digest
    assert cluster.run_context.source_content_hashes == local.run_context.source_content_hashes
    assert cluster.run_context.seed == local.run_context.seed


def test_the_gate_is_sensitive_to_the_seed(
    workload_image: ImageRef,
    store: S3ArtifactStore,
    pod_store_env: dict[str, str],
    repo_lockfile: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A control: equal addresses above must mean something, so a *different* run must differ.

    Without this, an equivalence assertion that compared two constants would pass just as happily.
    """
    monkeypatch.setenv(ENV_LOCKFILE_VAR, str(repo_lockfile))

    def job(seed: int) -> JobSpec:
        return JobSpec(
            image=workload_image,
            command=["python", "-c", workloads.DETERMINISTIC],
            env=pod_store_env,
            inputs={"x.txt": store.put(str(X).encode())},
            outputs=["y.txt"],
            seed=seed,
        )

    seven = submit(job(SEED), backend="live", store=store)
    nine = submit(job(9), backend="live", store=store)

    assert seven.outputs != nine.outputs
    assert seven.run_context.content_address() != nine.run_context.content_address()
