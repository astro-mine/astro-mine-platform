"""Range-readable object stores -- filesystem (dep-free) and S3 (moto)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from astro_mine.cloud.data.objectstore import (
    FilesystemObjectStore,
    ObjectStore,
    S3ObjectStore,
)

PAYLOAD = bytes(range(32))
BUCKET = "astro-mine-data"
REGION = "us-east-1"


# --- filesystem ----------------------------------------------------------------------


def test_filesystem_round_trip_and_range(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path)
    store.put("a/b.bin", PAYLOAD)
    assert isinstance(store, ObjectStore)
    assert store.get("a/b.bin") == PAYLOAD
    assert store.get_range("a/b.bin", 4, 8) == PAYLOAD[4:12]
    assert store.size("a/b.bin") == 32


def test_filesystem_missing_raises_keyerror(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path)
    with pytest.raises(KeyError):
        store.get("nope")
    with pytest.raises(KeyError):
        store.get_range("nope", 0, 1)
    with pytest.raises(KeyError):
        store.size("nope")


def test_filesystem_rejects_negative_range(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path)
    store.put("x", PAYLOAD)
    with pytest.raises(ValueError, match="non-negative"):
        store.get_range("x", -1, 4)


# --- S3 (moto) -----------------------------------------------------------------------


@pytest.fixture
def s3_store() -> Iterator[S3ObjectStore]:
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        client.put_object(Bucket=BUCKET, Key="data.bin", Body=PAYLOAD)
        yield S3ObjectStore(BUCKET, client=client)


def test_s3_whole_and_range_read(s3_store: S3ObjectStore) -> None:
    assert s3_store.get("data.bin") == PAYLOAD
    assert s3_store.get_range("data.bin", 8, 4) == PAYLOAD[8:12]
    assert s3_store.size("data.bin") == 32


def test_s3_rejects_nonpositive_range(s3_store: S3ObjectStore) -> None:
    with pytest.raises(ValueError, match="length must be positive"):
        s3_store.get_range("data.bin", 0, 0)


def test_s3_builds_its_own_client_when_none_given() -> None:
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
        store = S3ObjectStore(BUCKET, client_kwargs={"region_name": REGION})
        store._client.put_object(Bucket=BUCKET, Key="k", Body=b"hi")
        assert store.get("k") == b"hi"
