"""Range-readable object stores -- one S3 client, plus a dep-free local store.

The data plane needs **byte-range reads** so a worker fetches a single chunk instead of a
whole object (``cloud.md`` §5, §8). :class:`S3ObjectStore` gives that over any S3-compatible
backend (MinIO on-prem, native S3/GCS/Azure in cloud) through one boto3 client -- storage
portable, ``[s3]`` extra. :class:`FilesystemObjectStore` is the dependency-free local store
(range reads via ``seek``), so the data layer works on a workstation with no cloud.

Backlog: RM-P1-CLOUD-04 -- astro-mine-cloud#15
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["FilesystemObjectStore", "ObjectStore", "S3ObjectStore"]


@runtime_checkable
class ObjectStore(Protocol):
    """A path-keyed object store supporting whole and byte-range reads."""

    def get(self, key: str) -> bytes:
        """Return the whole object at *key*; raise ``KeyError`` if absent."""
        ...

    def get_range(self, key: str, start: int, length: int) -> bytes:
        """Return *length* bytes of *key* starting at *start* (a chunk-range read)."""
        ...

    def size(self, key: str) -> int:
        """Return the object's size in bytes."""
        ...


class FilesystemObjectStore:
    """A local, dependency-free object store rooted at a directory (range reads via seek)."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def put(self, key: str, data: bytes) -> None:
        """Write *data* at *key* (a test/pre-stage helper; the store is otherwise read-only)."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None

    def get_range(self, key: str, start: int, length: int) -> bytes:
        if start < 0 or length < 0:
            raise ValueError("start and length must be non-negative")
        try:
            with self._path(key).open("rb") as handle:
                handle.seek(start)
                return handle.read(length)
        except FileNotFoundError:
            raise KeyError(key) from None

    def size(self, key: str) -> int:
        try:
            return self._path(key).stat().st_size
        except FileNotFoundError:
            raise KeyError(key) from None


def _load_boto3() -> Any:
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - only without the [s3] extra
        raise ModuleNotFoundError(
            "S3ObjectStore needs the 's3' extra: pip install 'astro-mine-platform[cloud-s3]'"
        ) from exc
    return boto3


class S3ObjectStore:
    """A range-readable object store over an S3-compatible bucket (MinIO / S3 / GCS / Azure).

    Pass an existing boto3 *client* (or moto's, in tests), or let the store build one from
    *endpoint_url* / *client_kwargs* (the ``s3`` extra). Mirrors
    :class:`~astro_mine.cloud.artifacts.s3.S3ArtifactStore`'s injectable-client seam.
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        client: Any | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.bucket = bucket
        if client is None:
            boto3 = _load_boto3()
            client = boto3.client("s3", endpoint_url=endpoint_url, **dict(client_kwargs or {}))
        self._client = client

    def get(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        data: bytes = response["Body"].read()
        return data

    def get_range(self, key: str, start: int, length: int) -> bytes:
        if length <= 0:
            raise ValueError("length must be positive for a range read")
        # HTTP range is inclusive on both ends: bytes=start-(start+length-1).
        header = f"bytes={start}-{start + length - 1}"
        response = self._client.get_object(Bucket=self.bucket, Key=key, Range=header)
        data: bytes = response["Body"].read()
        return data

    def size(self, key: str) -> int:
        size: int = self._client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        return size
