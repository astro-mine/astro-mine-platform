# SPDX-License-Identifier: Apache-2.0
"""Content-addressed object store for traces + provenance bundles (RM-P1-BENCH-10; bench.md §7).

The hosted leaderboard keeps its bulky, immutable artifacts — episode traces and the
:class:`~astro_mine.bench.leaderboard._provenance.ProvenanceBundle` behind every entry — in an
**S3-compatible object store**, separately from the relational catalog (Postgres) and the job
state (Redis). This module is that store as a *contract* plus two dependency-clean backends:

- :class:`InMemoryObjectStore` — the process-local default (tests + the local tier);
- :class:`FileObjectStore` — a digest-sharded on-disk layout (a laptop, or the object store's
  local-filesystem gateway).

A real deployment slots an S3/MinIO backend behind the same :class:`ObjectStore` protocol — the
hosted service is a *deployment of this same code*, not a second path (bench.md §2.6). Every put
is keyed by the ``sha256:`` digest of its bytes and every get **re-verifies** that digest
(fail-closed on a content-address mismatch), so a corrupted or swapped object can never be served
as authentic — the storage-layer half of the leaderboard's integrity posture (bench.md §9).

Backlog: RM-P1-BENCH-10 — astro-mine-bench#18
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

__all__ = [
    "FileObjectStore",
    "InMemoryObjectStore",
    "ObjectIntegrityError",
    "ObjectStore",
    "blob_digest",
]


class ObjectIntegrityError(Exception):
    """Raised when stored bytes do not hash to the digest they are addressed by (bench.md §9)."""


def blob_digest(data: bytes) -> str:
    """The ``sha256:<hex>`` content address of raw ``data`` (the object store's key)."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class ObjectStore(Protocol):
    """A content-addressed blob store: put returns the digest, get re-verifies it.

    Puts are idempotent — identical bytes address the same digest, so re-storing is a no-op.
    """

    def put(self, data: bytes) -> str:
        """Store ``data`` and return its ``sha256:`` content address."""
        ...

    def get(self, digest: str) -> bytes | None:
        """Return the bytes at ``digest`` (re-verifying the address), or ``None`` if absent."""
        ...

    def contains(self, digest: str) -> bool:
        """Whether an object at ``digest`` is present."""
        ...


def _verify(digest: str, data: bytes) -> bytes:
    """Assert ``data`` hashes to ``digest`` — content-addressing enforced on every read."""
    actual = blob_digest(data)
    if actual != digest:
        raise ObjectIntegrityError(
            f"object {digest} content-address mismatch (bytes hash {actual})"
        )
    return data


class InMemoryObjectStore:
    """A process-local :class:`ObjectStore` — the dependency-clean default backend."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def put(self, data: bytes) -> str:
        digest = blob_digest(data)
        self._blobs[digest] = data
        return digest

    def get(self, digest: str) -> bytes | None:
        data = self._blobs.get(digest)
        return None if data is None else _verify(digest, data)

    def contains(self, digest: str) -> bool:
        return digest in self._blobs


class FileObjectStore:
    """An on-disk :class:`ObjectStore` with an OCI-style ``sha256/<aa>/<hex>`` shard layout.

    The local-filesystem realization of the object store — a laptop deployment, or the gateway an
    S3/MinIO backend replaces at scale. Same digest keys, same fail-closed verify-on-read.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        algorithm, _, hexpart = digest.partition(":")
        if algorithm != "sha256" or len(hexpart) != 64:
            raise ValueError(f"not a sha256 object digest: {digest!r}")
        return self._root / "sha256" / hexpart[:2] / hexpart

    def put(self, data: bytes) -> str:
        digest = blob_digest(data)
        path = self._path(digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return digest

    def get(self, digest: str) -> bytes | None:
        path = self._path(digest)
        if not path.is_file():
            return None
        return _verify(digest, path.read_bytes())

    def contains(self, digest: str) -> bool:
        return self._path(digest).is_file()
