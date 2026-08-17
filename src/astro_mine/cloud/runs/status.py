# SPDX-License-Identifier: Apache-2.0
"""Ephemeral job-status store -- the fast submit/track/cancel read model (RM-P1-CLOUD-06).

``conventions.md`` §5 scopes **Redis** to "queues, locks, ephemeral cache ... short-lived
coordination"; authoritative scheduling/lifecycle state lives in PostgreSQL/etcd. So this store is
deliberately the *ephemeral* index a caller reads to answer "what is this run doing right now?"
without hitting the durable event log or the scheduler -- it is a cache, not the source of truth.

The seam mirrors the rest of Cloud (a Protocol + a dependency-free in-memory default + an
extra-gated real backend): :class:`JobStatusStore` is the contract, :class:`InMemoryJobStatusStore`
is the local-tier default, and :class:`RedisJobStatusStore` is the cluster backend behind the
``[redis]`` extra. The Redis store takes an **injected** client so its logic is covered in CI with
``fakeredis`` (mirroring how ``moto`` covers the S3 path); only :meth:`RedisJobStatusStore.from_url`
imports ``redis``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from astro_mine.cloud.runs.events import RunStatus

__all__ = [
    "InMemoryJobStatusStore",
    "JobStatus",
    "JobStatusStore",
    "RedisJobStatusStore",
]

#: Redis key prefix for the ephemeral status index (``conventions.md`` §5 short-lived coordination).
_KEY_PREFIX = "astro-mine:cloud:status"
#: The index set of known run addresses, so a tenant-wide track read needs no key scan.
_INDEX_KEY = f"{_KEY_PREFIX}:index"
#: Default TTL: ephemeral by construction -- a stale entry expires rather than lingering as truth.
_DEFAULT_TTL_SECONDS = 24 * 3600


class JobStatus(BaseModel):
    """The current lifecycle status of a run, keyed by its stable ``run_pin`` address."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_address: str
    status: RunStatus
    tenant: str | None = None
    run_id: str | None = None


@runtime_checkable
class JobStatusStore(Protocol):
    """A fast, ephemeral read model for run status: set on each transition, get/list to track."""

    def set_status(self, status: JobStatus) -> None:
        """Upsert the current status for ``status.run_address`` (last write wins)."""
        ...

    def get_status(self, run_address: str) -> JobStatus | None:
        """Return the current status for ``run_address``, or ``None`` if unknown/expired."""
        ...

    def list_statuses(self, *, tenant: str | None = None) -> list[JobStatus]:
        """Every known status, optionally filtered to ``tenant``, in ``run_address`` order."""
        ...


class InMemoryJobStatusStore:
    """The process-local default -- the local tier tracks status with no Redis (``cloud.md`` §2)."""

    def __init__(self) -> None:
        self._by_address: dict[str, JobStatus] = {}

    def set_status(self, status: JobStatus) -> None:
        self._by_address[status.run_address] = status

    def get_status(self, run_address: str) -> JobStatus | None:
        return self._by_address.get(run_address)

    def list_statuses(self, *, tenant: str | None = None) -> list[JobStatus]:
        values = [s for s in self._by_address.values() if tenant is None or s.tenant == tenant]
        return sorted(values, key=lambda s: s.run_address)


def _as_text(value: Any) -> str:
    """Decode a redis-py return (``bytes`` by default, ``str`` with ``decode_responses``)."""
    return value.decode() if isinstance(value, bytes) else str(value)


class RedisJobStatusStore:
    """A Redis-backed :class:`JobStatusStore` -- the cluster tier's ephemeral status index.

    Takes an **injected** redis-py-compatible ``client`` (so the logic is exercised in CI against
    ``fakeredis``; a deployment passes a real ``redis.Redis``). Each status is a TTL'd key, so the
    index is self-expiring -- it caches live status, never becomes authoritative state. State
    written by one process is read by another sharing the same Redis, so status **survives a
    publisher/consumer restart** (the AC). Use :meth:`from_url` to build one from a Redis URL.
    """

    def __init__(self, client: Any, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._client = client
        self._ttl = ttl_seconds

    @classmethod
    def from_url(cls, url: str, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> RedisJobStatusStore:
        """Build a store from a Redis ``url`` (requires the ``[redis]`` extra)."""
        try:
            import redis  # pragma: no cover - requires the [redis] extra
        except ModuleNotFoundError as exc:  # pragma: no cover - requires the [redis] extra
            raise ModuleNotFoundError(
                "RedisJobStatusStore.from_url needs the 'redis' extra: "
                "pip install 'astro-mine-platform[cloud-redis]'"
            ) from exc
        return cls(redis.Redis.from_url(url), ttl_seconds=ttl_seconds)  # pragma: no cover

    def _key(self, run_address: str) -> str:
        return f"{_KEY_PREFIX}:{run_address}"

    def set_status(self, status: JobStatus) -> None:
        self._client.set(self._key(status.run_address), status.model_dump_json(), ex=self._ttl)
        self._client.sadd(_INDEX_KEY, status.run_address)

    def get_status(self, run_address: str) -> JobStatus | None:
        raw = self._client.get(self._key(run_address))
        if raw is None:
            return None
        return JobStatus.model_validate_json(_as_text(raw))

    def list_statuses(self, *, tenant: str | None = None) -> list[JobStatus]:
        addresses = sorted(_as_text(member) for member in self._client.smembers(_INDEX_KEY))
        statuses: list[JobStatus] = []
        for address in addresses:
            status = self.get_status(address)
            if status is None:
                # The key TTL'd out; drop the dangling index member so the index self-heals.
                self._client.srem(_INDEX_KEY, address)
                continue
            if tenant is None or status.tenant == tenant:
                statuses.append(status)
        return statuses
