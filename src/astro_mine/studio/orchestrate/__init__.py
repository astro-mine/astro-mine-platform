# SPDX-License-Identifier: Apache-2.0
"""Design-loop orchestration (RM-P1-STUDIO-03).

The ``orchestrate/`` layer turns one ``DesignCandidate`` into a scored evaluation by
fanning it across the autonomy stack as durable, cancelable, resumable async jobs
(studio.md §6). Studio only *sequences* Core-contract calls on content-addressed
artifacts — it computes nothing and imports no sibling package.

The **transport is a seam**: :class:`~.jobs.JobDispatcher` has two implementations —
:class:`~.jobs.LocalDispatcher` (in-process ``asyncio``, the local tier that MUST work,
conventions.md §7) and :class:`~.cloud.CloudDispatcher` (fan-out via ``cloud.submit()``, behind
the optional ``[cloud]`` extra). ``run_batch`` holds either one and its durable/cancelable/
resumable guarantees are unchanged. Real NATS/JetStream eventing (RM-P1-CLOUD-06) and gRPC
sibling *services* remain seams behind the :class:`~.jobs.EventSink` and
:class:`~.clients.SiblingClients` Protocols.

Importing this package never imports ``astro_mine.cloud``: :mod:`.cloud` defers it to
construction time, so the base wheel still imports only Core (+ FastAPI).
"""

from __future__ import annotations

from .cache import InMemoryResultCache, ResultCache, cache_key
from .clients import (
    LOCAL_STAND_IN_EVALUATOR_ID,
    EpisodeResult,
    GuardRejection,
    SiblingClients,
    local_clients,
    objective_content_hash,
)
from .cloud import (
    CloudDispatcher,
    CloudEvaluationError,
    JobEventPublisher,
    MissingCloudExtra,
)
from .jobs import (
    CollectingEventSink,
    EventSink,
    InMemoryJobStore,
    JobDispatcher,
    JobEvent,
    JobRecord,
    JobStatus,
    JobStore,
    LocalDispatcher,
    NullEventSink,
    job_id_for,
    run_batch,
)
from .loop import evaluate_candidate
from .worker import EvaluationOutcome, EvaluationRequest, load_clients, run_request

__all__ = [
    "LOCAL_STAND_IN_EVALUATOR_ID",
    "CloudDispatcher",
    "CloudEvaluationError",
    "CollectingEventSink",
    "EpisodeResult",
    "EvaluationOutcome",
    "EvaluationRequest",
    "EventSink",
    "GuardRejection",
    "InMemoryJobStore",
    "InMemoryResultCache",
    "JobDispatcher",
    "JobEvent",
    "JobEventPublisher",
    "JobRecord",
    "JobStatus",
    "JobStore",
    "LocalDispatcher",
    "MissingCloudExtra",
    "NullEventSink",
    "ResultCache",
    "SiblingClients",
    "cache_key",
    "evaluate_candidate",
    "job_id_for",
    "load_clients",
    "local_clients",
    "objective_content_hash",
    "run_batch",
    "run_request",
]
