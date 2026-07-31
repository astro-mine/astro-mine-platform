"""The Cloud fan-out for the design loop (RM-P1-STUDIO-03; studio.md §3, §6, §12; cloud.md §3).

studio.md §6 →Cloud: "large fan-out (hundreds of candidates x many seeds, or
training-in-the-loop) is submitted to Cloud's K8s/Ray scale-out; **Studio tracks jobs, not
workers**." :class:`CloudDispatcher` is that submission path — a second
:class:`~.jobs.JobDispatcher` that runs the same job loop over ``cloud.submit()`` instead of
in-process ``asyncio``. ``LocalDispatcher`` is the other implementation of the same Protocol;
:func:`~.jobs.run_batch` does not know or care which one it holds.

Same contract, not a weaker one
-------------------------------
The three STUDIO-03 guarantees are *structural* — they live in ``run_batch`` and the
:class:`~.jobs.JobStore`/:class:`~.cache.ResultCache` seams, above the dispatcher — so they hold
for any ``JobDispatcher``:

- **durable** — job state is written to the ``JobStore`` by ``run_batch``, before and after each
  dispatch. A killed run resumes from the store regardless of transport.
- **cancelable** — ``run_batch``'s cooperative pre-dispatch checkpoint. Note honestly what this
  does *not* buy: Cloud's ``submit()`` is a blocking call that returns no handle and exposes no
  cancel API, so a job already in flight cannot be killed. Cancellation is therefore exactly as
  cooperative as it is locally — a job canceled before it claims a concurrency slot is skipped —
  and no stronger claim is made. In-flight cancellation needs a Cloud-side handle, which
  ``cloud.md`` §3 does not define.
- **resumable** — the content-addressed :class:`~.cache.ResultCache`. ``CloudDispatcher`` checks
  it *before* submitting, so a cached candidate costs no job at all, exactly as
  ``LocalDispatcher``'s :func:`~.loop.evaluate_candidate` short-circuits in-process.

And the evaluation itself is the same: the job runs
:mod:`astro_mine.studio.orchestrate.worker`, which calls the same
:func:`~.loop.evaluate_candidate` with the same seed. A Cloud-evaluated candidate and a
locally-evaluated one are byte-identical for the same ``(candidate, objective, seed)`` — asserted
in ``tests/test_cloud_dispatch.py``.

Why per-candidate ``JobSpec`` and not ``SweepSpec``
--------------------------------------------------
Cloud's ``SweepSpec`` varies **environment parameters** over one base job and derives each
variant's seed positionally (``base.seed + index``). A design-loop batch is the wrong shape for
that: candidates differ by their *content-addressed input payload*, not by an env string, and
their seeds are explicit (``run_batch(seeds=…)``), not positional. So each candidate/seed is one
``JobSpec`` pinned to its own input digests, and the concurrency ``SweepSpec.max_parallel`` would
otherwise provide is already owned by ``run_batch``'s semaphore — together with the durability
and cancellation a bare sweep has no notion of.

Out of scope
------------
**gRPC sibling fan-out** — talking to Sim/Learn/Mind/Allocate/Guard/Bench as *services* over
gRPC, rather than through the in-process ``SiblingClients`` Protocols — is a separate
service-level integration, tracked in ``astro-mine-sim`` as "[gap] gRPC ``EnvironmentService`` +
Ray-actor service skin". It is orthogonal to this module: it changes what a worker binds its
clients to (via ``ASTRO_MINE_STUDIO_CLIENTS``, see :mod:`.worker`), not how batches are fanned
out. Cloud's ``cluster`` backend is likewise only usable once a ``ClusterClient`` is supplied —
its registered default raises ``NotImplementedError`` — hence "local and docker first".
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from astro_mine.core.compat import CORE_INTERFACE_VERSIONS
from astro_mine.core.objective import ObjectiveDocument

from ..models import DesignCandidate, EvaluatedCandidate
from .cache import ResultCache, cache_key
from .clients import GuardRejection
from .jobs import EventSink, JobEvent, JobStatus, job_id_for
from .worker import (
    CLIENTS_ENV,
    DEFAULT_CLIENTS_FACTORY,
    OBJECTIVE_INPUT,
    OUTCOME_OUTPUT,
    REQUEST_INPUT,
    EvaluationOutcome,
    EvaluationRequest,
    encode_objective,
)

if TYPE_CHECKING:  # pragma: no cover - the [cloud] extra is not in the base wheel
    from astro_mine.cloud.packaging import ImageRef
    from astro_mine.cloud.runs import CompletionEvent
    from astro_mine.core.artifacts import ArtifactStore

__all__ = [
    "LOCAL_IMAGE_DIGEST",
    "LOCAL_IMAGE_REPOSITORY",
    "WORKER_MODULE",
    "CloudDispatcher",
    "CloudEvaluationError",
    "JobEventPublisher",
    "MissingCloudExtra",
]

#: The argv Cloud runs (see :mod:`.worker`).
WORKER_MODULE = "astro_mine.studio.orchestrate.worker"

#: ``JobSpec.image`` is required and must be digest-pinned — but the **local** backend never
#: pulls or executes it (it subprocesses ``job.command`` directly), so for local runs the field
#: is pure provenance. Rather than borrow a real image's digest and record a run that never
#: happened in it, local runs are pinned to this self-describing constant: the digest is the hash
#: of the string below, so it is honest about being a placeholder and stable across runs. The
#: docker and cluster backends *do* execute the image, so they refuse to start without a real one.
LOCAL_IMAGE_REPOSITORY = "local/astro-mine-studio"
LOCAL_IMAGE_DIGEST = f"sha256:{hashlib.sha256(b'astro-mine-studio:local-dispatcher').hexdigest()}"

#: Core holds every published interface in lockstep (VERSIONING.md §4); ``objective`` is the Core
#: contract the dispatch payload actually carries (the ``ObjectiveDocument`` wire form), so it is
#: the version this job declares. Cloud admits it against Core's own compatibility rule.
_CORE_INTERFACE_VERSION = CORE_INTERFACE_VERSIONS["objective"]

#: Cloud's run lifecycle -> Studio's job lifecycle. Cloud has no `canceled` (it exposes no cancel),
#: so the mapping is total in this direction only.
_STATUS: Mapping[str, JobStatus] = {
    "submitted": JobStatus.PENDING,
    "started": JobStatus.RUNNING,
    "completed": JobStatus.SUCCEEDED,
    "failed": JobStatus.FAILED,
}


class MissingCloudExtra(ImportError):
    """:class:`CloudDispatcher` was constructed without the optional ``[cloud]`` extra."""

    def __init__(self) -> None:
        super().__init__(
            "CloudDispatcher requires the optional [cloud] extra; install it with "
            "`uv sync --extra cloud` (or `pip install astro-mine-platform[studio-cloud]`). "
            "LocalDispatcher needs no extra and remains the working local tier."
        )


class CloudEvaluationError(RuntimeError):
    """A Cloud job failed to *run* (bad image, missing dependency, crashed worker).

    Distinct from a :class:`~.clients.GuardRejection`, which is a successful run reporting an
    infeasible candidate — that is re-raised as ``GuardRejection`` so a Cloud-fanned batch records
    the same ``FAILED`` job, with Guard's own reason, that a local batch would.
    """


class JobEventPublisher:
    """Adapts Cloud's ``EventPublisher`` to Studio's :class:`~.jobs.EventSink`.

    The two shapes already mirror each other (jobs.py: "``JobEvent`` mirrors Cloud's
    ``CompletionEvent``"), so a Cloud run's lifecycle re-enters Studio as ordinary ``JobEvent``s —
    no second event vocabulary. The Cloud run is bound to the Studio ``job_id`` it was submitted
    for, because Cloud addresses runs by content (``run_address``) and knows nothing of candidates.
    """

    def __init__(self, sink: EventSink, job_id: str) -> None:
        self._sink = sink
        self._job_id = job_id

    def publish(self, subject: str, event: CompletionEvent) -> None:
        status = _STATUS.get(event.status)
        if status is not None:
            self._sink.emit(JobEvent(job_id=self._job_id, status=status))


class CloudDispatcher:
    """A :class:`~.jobs.JobDispatcher` that evaluates candidates through ``cloud.submit()``.

    ``backend`` names a Cloud backend (``"local"`` — subprocess, no daemon; ``"docker"`` — the
    digest-pinned image; ``"cluster"`` — K8s/Ray, once a ``ClusterClient`` is registered).
    ``image`` is required for every backend that actually executes it, and may be an ``ImageRef``
    or a digest-pinned ``"repo@sha256:…"`` string. ``clients_factory`` (``"module:factory"``)
    selects the ``SiblingClients`` bundle the *worker* binds — the seam a real deployment points
    at its live siblings.

    ``cache`` gives the content-addressed skip-if-already-evaluated behavior in the dispatcher
    itself, so a cache hit costs no Cloud job. ``events`` receives the Cloud run's lifecycle,
    adapted into Studio ``JobEvent``s (note ``run_batch`` separately emits the *job's* lifecycle
    to its own sink; pass the same sink to see both, or leave this unset).
    """

    def __init__(
        self,
        *,
        backend: str = "local",
        image: ImageRef | str | None = None,
        clients_factory: str = DEFAULT_CLIENTS_FACTORY,
        store: ArtifactStore | None = None,
        cache: ResultCache | None = None,
        events: EventSink | None = None,
        command: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        _require_cloud()
        if image is None and backend != "local":
            raise ValueError(
                f"the {backend!r} Cloud backend executes the image, so it needs a real "
                "digest-pinned `image`; only the local backend (which subprocesses the command "
                "directly) may fall back to the placeholder"
            )
        self._backend = backend
        self._image = image
        self._clients_factory = clients_factory
        self._store: ArtifactStore | None = store
        self._cache = cache
        self._events = events
        self._command = list(command) if command is not None else _default_command(backend)
        self._env = dict(env) if env is not None else {}

    async def evaluate(
        self,
        candidate: DesignCandidate,
        objective: ObjectiveDocument,
        *,
        seed: int,
        max_steps: int,
    ) -> EvaluatedCandidate:
        key = cache_key(candidate, objective, seed)
        if self._cache is not None and self._cache.has(key):
            return self._cache.get(key)  # resumable: a cache hit costs no Cloud job

        # `cloud.submit()` is synchronous and blocking (it waits on the job), so it goes off the
        # event loop onto a worker thread — the same shape LocalDispatcher uses.
        evaluated = await asyncio.to_thread(self._submit, candidate, objective, seed, max_steps)

        if self._cache is not None:
            self._cache.put(key, evaluated)
        return evaluated

    # -- internals ---------------------------------------------------------- #

    def _submit(
        self,
        candidate: DesignCandidate,
        objective: ObjectiveDocument,
        seed: int,
        max_steps: int,
    ) -> EvaluatedCandidate:
        from astro_mine.cloud import submit
        from astro_mine.cloud.submission import JobSpec

        store = self._artifact_store()
        request = EvaluationRequest(candidate=candidate, seed=seed, max_steps=max_steps)
        job = JobSpec(
            image=self._image_ref(),
            command=list(self._command),
            env={CLIENTS_ENV: self._clients_factory, **self._env},
            # Both payloads are content-addressed, so an identical candidate/objective submitted
            # twice pins the identical inputs — Cloud's provenance and Studio's cache key agree.
            inputs={
                REQUEST_INPUT: store.put(request.model_dump_json().encode("utf-8")),
                OBJECTIVE_INPUT: store.put(encode_objective(objective)),
            },
            outputs=[OUTCOME_OUTPUT],
            seed=seed,
            core_interface_version=_CORE_INTERFACE_VERSION,
        )

        sink = self._events
        publisher = None if sink is None else JobEventPublisher(sink, job_id_for(candidate, seed))
        result = submit(job, backend=self._backend, store=store, publisher=publisher)

        if not result.ok:
            raise CloudEvaluationError(
                f"cloud job for candidate {candidate.id!r} (seed {seed}) failed on the "
                f"{self._backend!r} backend with exit code {result.exit_code}"
            )

        outcome = EvaluationOutcome.model_validate_json(store.get(result.outputs[OUTCOME_OUTPUT]))
        if outcome.evaluated is None:
            # The job ran; the candidate is infeasible. Raise what the local path raises, so
            # `run_batch` records the identical FAILED job carrying Guard's own reason.
            raise GuardRejection(outcome.error or "candidate rejected without a reason")
        return outcome.evaluated

    def _artifact_store(self) -> ArtifactStore:
        if self._store is None:
            from astro_mine.cloud.artifacts.store import FilesystemArtifactStore

            self._store = FilesystemArtifactStore()
        return self._store

    def _image_ref(self) -> ImageRef:
        from astro_mine.cloud.packaging import ImageRef

        if self._image is None:
            return ImageRef(repository=LOCAL_IMAGE_REPOSITORY, digest=LOCAL_IMAGE_DIGEST)
        if isinstance(self._image, str):
            return ImageRef.parse(self._image)  # rejects an unpinned reference
        return self._image


def _default_command(backend: str) -> list[str]:
    """The worker argv. The local backend subprocesses on *this* host, so it must be *this*
    interpreter (the environment Studio is installed in); a container runs its own."""
    python = sys.executable if backend == "local" else "python"
    return [python, "-m", WORKER_MODULE]


def _require_cloud() -> None:
    try:
        import astro_mine.cloud  # noqa: F401  (the [cloud] extra backs the dispatcher)
    except ImportError as exc:
        raise MissingCloudExtra() from exc
