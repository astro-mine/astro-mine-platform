"""The ephemeral job-status store -- in-memory default + Redis backend (RM-P1-CLOUD-06).

The Redis backend's logic is covered hermetically with ``fakeredis`` (an injected client), mirroring
how ``moto`` covers the S3 path -- no server, deterministic, CI-safe. An opt-in ``redis``-marked
test against a real server is a deployment concern, not required here.
"""

from __future__ import annotations

import pytest

from astro_mine.cloud.runs.status import (
    InMemoryJobStatusStore,
    JobStatus,
    JobStatusStore,
    RedisJobStatusStore,
)

fakeredis = pytest.importorskip("fakeredis")


def _status(address: str, status: str = "submitted", *, tenant: str | None = None) -> JobStatus:
    return JobStatus(run_address=address, status=status, tenant=tenant, run_id=f"run-{address[-1]}")


@pytest.fixture(params=["memory", "redis"])
def store(request: pytest.FixtureRequest) -> JobStatusStore:
    if request.param == "memory":
        return InMemoryJobStatusStore()
    return RedisJobStatusStore(fakeredis.FakeStrictRedis())


def test_stores_satisfy_protocol(store: JobStatusStore) -> None:
    assert isinstance(store, JobStatusStore)


def test_set_then_get(store: JobStatusStore) -> None:
    store.set_status(_status("sha256:a", tenant="acme"))
    got = store.get_status("sha256:a")
    assert got is not None
    assert got.status == "submitted"
    assert got.tenant == "acme"


def test_get_unknown_is_none(store: JobStatusStore) -> None:
    assert store.get_status("sha256:missing") is None


def test_last_write_wins(store: JobStatusStore) -> None:
    store.set_status(_status("sha256:a", "submitted"))
    store.set_status(_status("sha256:a", "completed"))
    got = store.get_status("sha256:a")
    assert got is not None and got.status == "completed"


def test_list_filters_by_tenant(store: JobStatusStore) -> None:
    store.set_status(_status("sha256:a", tenant="acme"))
    store.set_status(_status("sha256:b", tenant="beta"))
    store.set_status(_status("sha256:c", tenant="acme"))
    acme = store.list_statuses(tenant="acme")
    assert [s.run_address for s in acme] == ["sha256:a", "sha256:c"]
    assert len(store.list_statuses()) == 3


def test_redis_status_survives_a_restart() -> None:
    # The data lives in Redis, not the process: a fresh store over the same client reads it back
    # -- the "survives a publisher/consumer restart" guarantee.
    client = fakeredis.FakeStrictRedis()
    RedisJobStatusStore(client).set_status(_status("sha256:z", "started", tenant="acme"))
    reopened = RedisJobStatusStore(client)
    got = reopened.get_status("sha256:z")
    assert got is not None and got.status == "started"
