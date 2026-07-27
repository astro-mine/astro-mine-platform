"""STUDIO-03 — CloudDispatcher: the design-loop fan-out over ``cloud.submit()``.

Mirrors ``test_jobs.py`` scenario-for-scenario against ``CloudDispatcher`` — the same
durable/cancelable/resumable assertions, the same cache assertions — because the acceptance
criterion is that ``CloudDispatcher`` satisfies the *same* ``JobDispatcher`` contract, not a
weaker one. Every job here is a real ``cloud.submit()`` through a real Cloud backend, running a
real subprocess: only the container engine is simulated (in the docker lane), and only because
CI has no daemon.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from astro_mine.core.objective import ObjectiveDocument
from astro_mine.studio.models import AssetSelection, DesignCandidate
from astro_mine.studio.orchestrate import (
    CollectingEventSink,
    GuardRejection,
    InMemoryJobStore,
    InMemoryResultCache,
    JobDispatcher,
    JobRecord,
    JobStatus,
    LocalDispatcher,
    SiblingClients,
    cache_key,
    job_id_for,
    run_batch,
)
from astro_mine.studio.orchestrate.cloud import (
    LOCAL_IMAGE_DIGEST,
    WORKER_MODULE,
    CloudDispatcher,
    CloudEvaluationError,
    JobEventPublisher,
    MissingCloudExtra,
    _default_command,
)

pytest.importorskip("astro_mine.cloud", reason="the [cloud] extra is not installed")

from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.runs import CompletionEvent
from astro_mine.cloud.submission import DockerBackend
from astro_mine.cloud.submission.backend import get_backend, register_backend

# A digest-pinned image. The docker lane never pulls it (the engine is simulated), but JobSpec
# and DockerBackend both insist on a pinned reference, which is the point.
_IMAGE = ImageRef.parse("ghcr.io/astro-mine/astro-mine-studio@sha256:" + "ab" * 32)


def _candidates(n: int) -> list[DesignCandidate]:
    return [
        DesignCandidate(id=f"c{i}", swarm=[AssetSelection(sadf_ref="rover", count=i + 1)])
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path: Path) -> FilesystemArtifactStore:
    """A per-test artifact store — never the default `./.astro-mine/artifacts` in the repo."""
    return FilesystemArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def dispatcher(store: FilesystemArtifactStore) -> CloudDispatcher:
    return CloudDispatcher(backend="local", store=store)


@pytest.fixture
def docker_backend() -> Iterator[Callable[[Sequence[str]], int]]:
    """Register a *simulated* docker engine as Cloud's `docker` backend for the test.

    Cloud's DockerBackend takes an injectable runner precisely so the docker code path — the
    `docker run` argv, the /inputs//outputs bind mounts, the -e env — can be exercised without a
    daemon. The runner below translates the container paths the backend emitted back to the host
    bind-mount dirs and runs the inner command, so `submit(backend="docker")` really does drive
    Cloud's docker path end to end; only `docker` itself is stubbed out.
    """

    def run(argv: Sequence[str]) -> int:
        mounts: dict[str, str] = {}
        env: dict[str, str] = {}
        index = 0
        image_index: int | None = None
        while index < len(argv):
            token = argv[index]
            if token == "-v":
                host, _, rest = argv[index + 1].partition(":")
                mounts[rest.split(":")[0]] = host
                index += 2
            elif token == "-e":
                key, _, value = argv[index + 1].partition("=")
                env[key] = value
                index += 2
            elif token in {"docker", "run", "--rm", "--network=none"}:
                index += 1
            else:
                image_index = index
                break
        assert image_index is not None, "no image reference in the docker argv"

        host_env = dict(os.environ)
        for key, value in env.items():
            host_env[key] = mounts.get(value, value)  # /inputs -> host dir; the seed stays put
        inner = list(argv[image_index + 1 :])
        return subprocess.run(inner, env=host_env, check=False).returncode

    previous = get_backend("docker")
    register_backend("docker", DockerBackend(runner=run), replace=True)
    try:
        yield run
    finally:
        register_backend("docker", previous, replace=True)


# ---- construction & wiring ------------------------------------------------ #


def test_is_a_job_dispatcher(dispatcher: CloudDispatcher) -> None:
    assert isinstance(dispatcher, JobDispatcher)


def test_local_backend_falls_back_to_the_placeholder_image(
    dispatcher: CloudDispatcher,
) -> None:
    image = dispatcher._image_ref()
    assert image.digest == LOCAL_IMAGE_DIGEST  # honest placeholder, not a borrowed digest
    assert image.reference.endswith(LOCAL_IMAGE_DIGEST)


@pytest.mark.parametrize("backend", ["docker", "cluster"])
def test_an_executing_backend_demands_a_real_image(
    backend: str, store: FilesystemArtifactStore
) -> None:
    """docker/cluster actually run the image, so the placeholder would be a lying provenance."""
    with pytest.raises(ValueError, match="digest-pinned"):
        CloudDispatcher(backend=backend, store=store)


def test_accepts_a_pinned_image_string(store: FilesystemArtifactStore) -> None:
    ref = CloudDispatcher(backend="docker", image=_IMAGE.reference, store=store)._image_ref()
    assert ref == _IMAGE


def test_rejects_an_unpinned_image_string(store: FilesystemArtifactStore) -> None:
    unpinned = CloudDispatcher(backend="docker", image="ghcr.io/astro-mine/studio:latest")
    with pytest.raises(ValueError, match="unpinned image reference"):
        unpinned._image_ref()


def test_the_worker_argv_targets_this_interpreter_locally_and_the_image_python_in_a_container() -> (
    None
):
    assert _default_command("local") == [sys.executable, "-m", WORKER_MODULE]
    assert _default_command("docker") == ["python", "-m", WORKER_MODULE]


# ---- the design loop, end to end through cloud.submit() ------------------- #


def test_cloud_evaluation_is_byte_identical_to_the_local_one(
    dispatcher: CloudDispatcher,
    objective_doc: ObjectiveDocument,
    clients: SiblingClients,
    candidate: DesignCandidate,
) -> None:
    """The whole justification for the drop-in claim: same loop, same seed, same result."""
    remote = asyncio.run(dispatcher.evaluate(candidate, objective_doc, seed=3, max_steps=4))
    local = asyncio.run(
        LocalDispatcher(clients).evaluate(candidate, objective_doc, seed=3, max_steps=4)
    )
    assert remote.digest() == local.digest()


def test_run_batch_over_cloud_succeeds_with_events(
    dispatcher: CloudDispatcher, objective_doc: ObjectiveDocument
) -> None:
    candidates = _candidates(2)
    events = CollectingEventSink()
    records = asyncio.run(
        run_batch(
            candidates,
            objective_doc,
            dispatcher=dispatcher,
            seeds=(1,),
            events=events,
            concurrency=2,
            max_steps=2,
        )
    )
    assert len(records) == 2
    assert all(r.status is JobStatus.SUCCEEDED for r in records)
    assert all(r.result_digest is not None for r in records)
    emitted = {(e.job_id, e.status) for e in events.events}
    assert (job_id_for(candidates[0], 1), JobStatus.SUCCEEDED) in emitted


def test_cloud_run_lifecycle_reaches_the_studio_event_sink(
    store: FilesystemArtifactStore,
    objective_doc: ObjectiveDocument,
    candidate: DesignCandidate,
) -> None:
    """Cloud's CompletionEvents re-enter Studio as JobEvents — one event vocabulary, not two."""
    events = CollectingEventSink()
    dispatcher = CloudDispatcher(backend="local", store=store, events=events)
    asyncio.run(dispatcher.evaluate(candidate, objective_doc, seed=1, max_steps=2))

    job_id = job_id_for(candidate, 1)
    seen = [(e.job_id, e.status) for e in events.events]
    assert (job_id, JobStatus.PENDING) in seen  # cloud "submitted"
    assert (job_id, JobStatus.RUNNING) in seen  # cloud "started"
    assert (job_id, JobStatus.SUCCEEDED) in seen  # cloud "completed"


def test_job_event_publisher_maps_the_cloud_vocabulary() -> None:
    events = CollectingEventSink()
    publisher = JobEventPublisher(events, "c0@7")
    for status in ("submitted", "started", "completed", "failed"):
        publisher.publish("astro-mine.cloud.runs", CompletionEvent(run_address="r", status=status))
    assert [e.status for e in events.events] == [
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
    ]
    assert {e.job_id for e in events.events} == {"c0@7"}


# ---- the three guarantees (mirrors test_jobs.py) -------------------------- #


def test_run_batch_over_cloud_resumes_from_the_store(
    objective_doc: ObjectiveDocument, store: FilesystemArtifactStore
) -> None:
    """Durable: a job already SUCCEEDED in the store is not re-submitted."""
    candidates = _candidates(2)
    job_store = InMemoryJobStore()
    dispatcher = _CountingCloudDispatcher(store)

    asyncio.run(
        run_batch(
            candidates,
            objective_doc,
            dispatcher=dispatcher,
            seeds=(1,),
            store=job_store,
            max_steps=2,
        )
    )
    assert dispatcher.calls == 2

    asyncio.run(
        run_batch(
            candidates,
            objective_doc,
            dispatcher=dispatcher,
            seeds=(1,),
            store=job_store,
            max_steps=2,
        )
    )
    assert dispatcher.calls == 2  # resumed: no candidate re-submitted to Cloud


def test_result_cache_skips_an_already_evaluated_candidate(
    objective_doc: ObjectiveDocument, store: FilesystemArtifactStore
) -> None:
    """Resumable: the content-addressed cache short-circuits *before* a job is submitted."""
    candidates = _candidates(2)
    cache = InMemoryResultCache()
    dispatcher = _CountingCloudDispatcher(store, cache=cache)

    asyncio.run(
        run_batch(
            candidates, objective_doc, dispatcher=dispatcher, seeds=(1,), cache=cache, max_steps=2
        )
    )
    assert dispatcher.calls == 2

    # A fresh job store, a warm cache -> every job resolves from the content-addressed checkpoint.
    records = asyncio.run(
        run_batch(
            candidates,
            objective_doc,
            dispatcher=dispatcher,
            seeds=(1,),
            store=InMemoryJobStore(),
            cache=cache,
            max_steps=2,
        )
    )
    assert dispatcher.calls == 2  # no second Cloud job
    assert all(r.status is JobStatus.SUCCEEDED for r in records)


def test_dispatcher_level_cache_hit_submits_no_job(
    objective_doc: ObjectiveDocument,
    store: FilesystemArtifactStore,
    candidate: DesignCandidate,
) -> None:
    """The cache is checked in the dispatcher too, so a direct `evaluate` never reaches Cloud."""
    cache = InMemoryResultCache()
    dispatcher = _CountingCloudDispatcher(store, cache=cache)

    first = asyncio.run(dispatcher.evaluate(candidate, objective_doc, seed=5, max_steps=2))
    assert dispatcher.calls == 1
    assert cache.has(cache_key(candidate, objective_doc, 5))

    second = asyncio.run(dispatcher.evaluate(candidate, objective_doc, seed=5, max_steps=2))
    assert dispatcher.calls == 1  # skip-if-already-evaluated
    assert second.digest() == first.digest()


def test_run_batch_over_cloud_cancels_a_flagged_job(
    dispatcher: CloudDispatcher, objective_doc: ObjectiveDocument
) -> None:
    """Cancelable: the cooperative pre-dispatch checkpoint, identical to the local path."""
    candidates = _candidates(1)
    job_store = InMemoryJobStore()
    job_id = job_id_for(candidates[0], 9)
    job_store.create(
        JobRecord(
            job_id=job_id,
            candidate_id=candidates[0].id,
            seed=9,
            cache_key=cache_key(candidates[0], objective_doc, 9),
            cancel_requested=True,
        )
    )
    records = asyncio.run(
        run_batch(
            candidates,
            objective_doc,
            dispatcher=dispatcher,
            seeds=(9,),
            store=job_store,
            max_steps=2,
        )
    )
    assert records[0].status is JobStatus.CANCELED
    assert records[0].result_digest is None


def test_guard_rejection_survives_the_job_boundary(
    dispatcher: CloudDispatcher, objective_doc: ObjectiveDocument
) -> None:
    """An infeasible candidate is a *result*, not a crashed job — and Guard's reason comes back.

    The identical assertion `test_jobs.py::test_run_batch_records_guard_failure` makes locally.
    """
    unsafe = DesignCandidate(
        id="unsafe",
        swarm=[AssetSelection(sadf_ref="rover", count=1)],
        decision_vector={"unsafe": 1.0},
    )
    records = asyncio.run(run_batch([unsafe], objective_doc, dispatcher=dispatcher, max_steps=2))
    assert records[0].status is JobStatus.FAILED
    assert records[0].error is not None and "certification" in records[0].error


def test_guard_rejection_raises_guard_rejection_not_a_cloud_error(
    dispatcher: CloudDispatcher, objective_doc: ObjectiveDocument
) -> None:
    unsafe = DesignCandidate(
        id="unsafe",
        swarm=[AssetSelection(sadf_ref="rover", count=1)],
        decision_vector={"unsafe": 1.0},
    )
    with pytest.raises(GuardRejection, match="certification"):
        asyncio.run(dispatcher.evaluate(unsafe, objective_doc, seed=0, max_steps=2))


def test_a_broken_worker_raises_a_cloud_evaluation_error(
    store: FilesystemArtifactStore,
    objective_doc: ObjectiveDocument,
    candidate: DesignCandidate,
) -> None:
    """A non-zero exit means the *job* broke (bad image, missing dep) — distinct from infeasible."""
    broken = CloudDispatcher(
        backend="local",
        store=store,
        command=[sys.executable, "-c", "import sys; sys.exit(3)"],
    )
    with pytest.raises(CloudEvaluationError, match="exit code 3"):
        asyncio.run(broken.evaluate(candidate, objective_doc, seed=0, max_steps=2))


# ---- the docker backend --------------------------------------------------- #


def test_docker_backend_round_trips_a_candidate(
    docker_backend: Callable[[Sequence[str]], int],
    store: FilesystemArtifactStore,
    objective_doc: ObjectiveDocument,
    clients: SiblingClients,
    candidate: DesignCandidate,
) -> None:
    """`submit(backend="docker")` through Cloud's real docker path (simulated engine only).

    The container's python is `sys.executable` here because the simulated engine runs the inner
    command on the host; in a real container it is the image's own interpreter (`_default_command`).
    """
    dispatcher = CloudDispatcher(
        backend="docker",
        image=_IMAGE,
        store=store,
        command=[sys.executable, "-m", WORKER_MODULE],
    )
    evaluated = asyncio.run(dispatcher.evaluate(candidate, objective_doc, seed=3, max_steps=4))
    local = asyncio.run(
        LocalDispatcher(clients).evaluate(candidate, objective_doc, seed=3, max_steps=4)
    )
    # cloud.md §2 principle 2: a container run is byte-for-byte equivalent to the local run.
    assert evaluated.digest() == local.digest()


def test_docker_backend_runs_a_batch(
    docker_backend: Callable[[Sequence[str]], int],
    store: FilesystemArtifactStore,
    objective_doc: ObjectiveDocument,
) -> None:
    dispatcher = CloudDispatcher(
        backend="docker",
        image=_IMAGE,
        store=store,
        command=[sys.executable, "-m", WORKER_MODULE],
    )
    records = asyncio.run(
        run_batch(_candidates(2), objective_doc, dispatcher=dispatcher, seeds=(1,), max_steps=2)
    )
    assert all(r.status is JobStatus.SUCCEEDED for r in records)


@pytest.mark.skipif(
    not (os.environ.get("ASTRO_MINE_STUDIO_DOCKER_IMAGE") and shutil.which("docker")),
    reason=(
        "set ASTRO_MINE_STUDIO_DOCKER_IMAGE (a digest-pinned image with astro-mine-studio "
        "installed) and install Docker; no such image is published during private incubation"
    ),
)
def test_real_docker_round_trip(
    store: FilesystemArtifactStore,
    objective_doc: ObjectiveDocument,
    clients: SiblingClients,
    candidate: DesignCandidate,
) -> None:
    """Opt-in: a real `docker run` of the real worker, against a real daemon.

    Skipped in CI — Studio publishes no image yet (that is Cloud's container-first packaging,
    RM-P1-CLOUD-01). The code path is the same one `test_docker_backend_round_trips_a_candidate`
    covers; what this adds is a live daemon and a real image.
    """
    dispatcher = CloudDispatcher(
        backend="docker",
        image=os.environ["ASTRO_MINE_STUDIO_DOCKER_IMAGE"],
        store=store,
    )
    evaluated = asyncio.run(dispatcher.evaluate(candidate, objective_doc, seed=3, max_steps=4))
    local = asyncio.run(
        LocalDispatcher(clients).evaluate(candidate, objective_doc, seed=3, max_steps=4)
    )
    assert evaluated.digest() == local.digest()


# ---- the optional extra ---------------------------------------------------- #


def test_missing_cloud_extra_is_raised_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast and actionably, rather than deep inside a batch. (`None` in sys.modules makes
    the import raise, which is exactly what an uninstalled extra does.)"""
    monkeypatch.setitem(sys.modules, "astro_mine.cloud", None)
    with pytest.raises(MissingCloudExtra, match=r"--extra cloud"):
        CloudDispatcher(backend="local")


def test_the_default_store_is_memoized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no store injected, Cloud's own default is used — unwrapped."""
    monkeypatch.setenv("ASTRO_MINE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    dispatcher = CloudDispatcher(backend="local")

    store = dispatcher._artifact_store()
    assert isinstance(store, FilesystemArtifactStore)  # no Studio-side wrapper
    assert dispatcher._artifact_store() is store  # memoized
    assert store.get(store.put(b"hello")) == b"hello"
    assert (tmp_path / "artifacts").exists()  # the injected root, not the repo's cwd


# ---- concurrent writes under the fan-out ---------------------------------- #


def test_the_raw_store_survives_the_batch_fan_outs_identical_writes(tmp_path: Path) -> None:
    """Studio's fan-out must survive concurrent identical writes — now with no wrapper.

    ``run_batch`` fans jobs out across threads, and every candidate in a batch ships the *same*
    objective bytes, so identical concurrent writes are the common path, not a corner. Cloud's
    ``FilesystemArtifactStore.put`` used to name its scratch file after the **process** id, which
    threads share: one writer renamed the temp file away and the other died with ENOENT. Studio
    carried a ``SerializedArtifactStore`` lock to work around it.

    Fixed upstream in astro-mine-cloud#25: the scratch name is unique per *writer*, so this holds
    against the raw store. The guarantee moved into Cloud; it did not go away, and Studio is still
    the component whose fan-out depends on it — so the assertion stays here, pointed at the store
    the dispatcher actually uses now.
    """
    payload = b"the identical bytes two candidates in one batch both store"
    store = FilesystemArtifactStore(tmp_path / "artifacts")

    with ThreadPoolExecutor(max_workers=16) as pool:
        addresses = list(pool.map(lambda _: store.put(payload), range(64)))

    assert len(set(addresses)) == 1  # content-addressed: one payload, one address
    assert store.get(addresses[0]) == payload
    assert store.exists(addresses[0])


class _CountingCloudDispatcher(CloudDispatcher):
    """A CloudDispatcher that counts the jobs it actually submits (to prove skip/resume)."""

    def __init__(self, store: FilesystemArtifactStore, **kwargs: object) -> None:
        super().__init__(backend="local", store=store, **kwargs)  # type: ignore[arg-type]
        self.calls = 0

    def _submit(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return super()._submit(*args, **kwargs)  # type: ignore[arg-type]
