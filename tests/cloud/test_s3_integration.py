"""Opt-in integration test against a real local MinIO.

Skipped in CI and by default. To run it::

    docker compose up -d minio
    MINIO_ENDPOINT=http://localhost:9000 uv run pytest -m minio

Credentials default to the ``docker-compose.yml`` values and can be overridden with
``MINIO_ACCESS_KEY`` / ``MINIO_SECRET_KEY``.
"""

from __future__ import annotations

import os

import pytest

from astro_mine.cloud.artifacts import addressing
from astro_mine.cloud.artifacts.s3 import S3ArtifactStore

BUCKET = "astro-mine-integration"

pytestmark = pytest.mark.minio


@pytest.mark.skipif(
    not os.environ.get("MINIO_ENDPOINT"),
    reason="set MINIO_ENDPOINT (and run `docker compose up -d minio`) to exercise real MinIO",
)
def test_minio_round_trip() -> None:
    import boto3

    endpoint = os.environ["MINIO_ENDPOINT"]
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        region_name="us-east-1",
    )
    existing = {b["Name"] for b in client.list_buckets()["Buckets"]}
    if BUCKET not in existing:
        client.create_bucket(Bucket=BUCKET)

    store = S3ArtifactStore(BUCKET, client=client)
    address = store.put(b"real-minio-payload")
    assert address == addressing.content_address(b"real-minio-payload")
    assert store.get(address) == b"real-minio-payload"
    assert store.exists(address)
