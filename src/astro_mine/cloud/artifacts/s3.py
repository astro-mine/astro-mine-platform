# SPDX-License-Identifier: Apache-2.0
"""S3-compatible artifact store (MinIO local, or any S3 backend).

Implements the :class:`~astro_mine.core.artifacts.ArtifactStore` contract
against an S3-compatible object store through a single boto3 client, so the same
content-addressed I/O runs against local MinIO (``docker compose up minio``) or a cloud
object store without a code change (``cloud.md`` §5, §11). boto3 is an *optional*
dependency -- install ``astro-mine-platform[cloud-s3]``; the base package and the sacred
filesystem tier never import it.

Keys mirror the filesystem layout: ``sha256/<h[:2]>/<h>``.

Backlog: RM-P0-CLOUD-03 -- astro-mine-cloud#3
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astro_mine.cloud.artifacts.addressing import content_address, parse_address

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["S3ArtifactStore"]

_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


def _load_boto3() -> Any:
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - only without the [s3] extra
        raise ModuleNotFoundError(
            "S3ArtifactStore needs the 's3' extra: pip install 'astro-mine-platform[cloud-s3]'"
        ) from exc
    return boto3


def _is_missing(exc: BaseException) -> bool:
    """Return whether a boto3 ``ClientError`` denotes an absent key (vs a real fault)."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str(response.get("Error", {}).get("Code", ""))
    return code in _MISSING_CODES


class S3ArtifactStore:
    """Content-addressed store backed by an S3-compatible bucket.

    Pass an existing boto3 *client*, or let the store build one from *endpoint_url* /
    *client_kwargs* (the latter requires the ``s3`` extra). The bucket must already
    exist -- bucket lifecycle is an operator concern, not this store's.
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

    def _key(self, address: str) -> str:
        _, hexdigest = parse_address(address)
        return f"sha256/{hexdigest[:2]}/{hexdigest}"

    def put(self, data: bytes) -> str:
        address = content_address(data)
        key = self._key(address)
        if not self._object_exists(key):
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return address

    def get(self, address: str) -> bytes:
        key = self._key(address)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # botocore.exceptions.ClientError; kept boto3-import-free
            if _is_missing(exc):
                raise KeyError(address) from None
            raise
        data: bytes = response["Body"].read()
        return data

    def exists(self, address: str) -> bool:
        return self._object_exists(self._key(address))

    def _object_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # botocore.exceptions.ClientError
            if _is_missing(exc):
                return False
            raise
        return True
