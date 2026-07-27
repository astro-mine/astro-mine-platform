"""Lifecycle events flow end-to-end through submit() (RM-P1-CLOUD-06).

Drives the real local run path with a CollectingPublisher + in-memory status store and asserts the
``submitted -> started -> completed/failed`` sequence, a single shared run identity, and the status
read model -- the CI-covered proof that the seam is wired into the run lifecycle. No broker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from astro_mine.cloud import submit
from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.runs import CollectingPublisher, InMemoryJobStatusStore
from astro_mine.cloud.submission import JobSpec

IMAGE = ImageRef.parse("ghcr.io/astro-mine/astro-mine-bench@sha256:" + "ab" * 32)


def _ok_job() -> JobSpec:
    return JobSpec(
        image=IMAGE, command=[sys.executable, "-c", "print('ok')"], seed=7, tenant="acme"
    )


def _statuses(publisher: CollectingPublisher) -> list[str]:
    return [event.status for _, event in publisher.events]


def test_successful_run_emits_submitted_started_completed(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    publisher, status_store = CollectingPublisher(), InMemoryJobStatusStore()

    result = submit(_ok_job(), store=store, publisher=publisher, status_store=status_store)

    assert result.ok
    assert _statuses(publisher) == ["submitted", "started", "completed"]
    # All three events correlate on one run identity (run_pin), and every event is on SUBJECT.
    assert len({event.run_address for _, event in publisher.events}) == 1
    assert {subject for subject, _ in publisher.events} == {"astro-mine.cloud.runs"}
    # The tenant rides every event.
    assert all(event.tenant == "acme" for _, event in publisher.events)
    # The status read model lands on the terminal state.
    final = status_store.get_status(publisher.events[-1][1].run_address)
    assert final is not None and final.status == "completed" and final.tenant == "acme"


def test_failing_run_emits_failed(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    publisher, status_store = CollectingPublisher(), InMemoryJobStatusStore()
    job = JobSpec(image=IMAGE, command=[sys.executable, "-c", "import sys; sys.exit(3)"], seed=1)

    result = submit(job, store=store, publisher=publisher, status_store=status_store)

    assert not result.ok
    assert _statuses(publisher) == ["submitted", "started", "failed"]
    assert status_store.list_statuses()[0].status == "failed"


def test_exception_during_run_emits_failed(tmp_path: Path) -> None:
    # A declared output that never appears raises -- but a `failed` event is emitted first.
    store = FilesystemArtifactStore(tmp_path)
    publisher = CollectingPublisher()
    job = JobSpec(image=IMAGE, command=[sys.executable, "-c", "pass"], outputs=["y.txt"])

    with pytest.raises(FileNotFoundError, match=r"y\.txt"):
        submit(job, store=store, publisher=publisher)

    assert _statuses(publisher) == ["submitted", "started", "failed"]


def test_local_submit_needs_no_publisher(tmp_path: Path) -> None:
    # The sacred local tier: submit() with no publisher/status store runs broker-free.
    store = FilesystemArtifactStore(tmp_path)
    result = submit(_ok_job(), store=store)
    assert result.ok


def test_cluster_dryrun_emits_the_full_lifecycle(tmp_path: Path) -> None:
    # The DryRun cluster path runs through the same harness, so events flow identically.
    from astro_mine.cloud.submission.backend import register_backend
    from astro_mine.cloud.submission.cluster import ClusterBackend, DryRunClient

    register_backend("cluster-dryrun", ClusterBackend(client=DryRunClient()), replace=True)
    store = FilesystemArtifactStore(tmp_path)
    publisher = CollectingPublisher()

    result = submit(_ok_job(), backend="cluster-dryrun", store=store, publisher=publisher)

    assert result.ok
    assert _statuses(publisher) == ["submitted", "started", "completed"]
