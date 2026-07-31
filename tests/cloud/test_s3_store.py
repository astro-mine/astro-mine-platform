"""S3ArtifactStore against a hermetic in-process mock (moto) -- no server, no account.

The real-MinIO path is exercised by the opt-in ``tests/test_s3_integration.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import boto3
import pytest
from moto import mock_aws

from astro_mine.cloud.artifacts import addressing
from astro_mine.cloud.artifacts.s3 import S3ArtifactStore
from astro_mine.core.artifacts import ArtifactStore

BUCKET = "astro-mine-test"
REGION = "us-east-1"


@pytest.fixture
def store() -> Iterator[S3ArtifactStore]:
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        yield S3ArtifactStore(BUCKET, client=client)


def test_put_get_round_trip(store: S3ArtifactStore) -> None:
    address = store.put(b"payload")
    assert address == addressing.content_address(b"payload")
    assert store.get(address) == b"payload"


def test_key_layout_mirrors_filesystem(store: S3ArtifactStore) -> None:
    address = store.put(b"payload")
    hexdigest = addressing.hex_of(address)
    assert store._key(address) == f"sha256/{hexdigest[:2]}/{hexdigest}"


def test_put_is_idempotent(store: S3ArtifactStore) -> None:
    first = store.put(b"same")
    second = store.put(b"same")  # exercises the already-exists fast path
    assert first == second
    assert store.exists(first)


def test_get_missing_raises_keyerror(store: S3ArtifactStore) -> None:
    missing = addressing.content_address(b"never-stored")
    assert not store.exists(missing)
    with pytest.raises(KeyError):
        store.get(missing)


def test_satisfies_protocol(store: S3ArtifactStore) -> None:
    assert isinstance(store, ArtifactStore)


def test_builds_its_own_client_when_none_given() -> None:
    # Exercises the _load_boto3() + boto3.client(...) construction path.
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
        built = S3ArtifactStore(BUCKET, client_kwargs={"region_name": REGION})
        address = built.put(b"hi")
        assert built.get(address) == b"hi"


class _S3Fault(Exception):
    """A boto3-shaped ClientError whose code is a genuine fault, not a missing key."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _FaultyClient:
    def head_object(self, **_: object) -> object:
        raise _S3Fault("AccessDenied")

    def get_object(self, **_: object) -> object:
        raise _S3Fault("AccessDenied")


def test_real_faults_propagate_and_are_not_swallowed() -> None:
    store = S3ArtifactStore(BUCKET, client=_FaultyClient())
    address = addressing.content_address(b"x")
    with pytest.raises(_S3Fault):
        store.exists(address)  # _object_exists re-raises a non-missing fault
    with pytest.raises(_S3Fault):
        store.get(address)  # get() re-raises a non-missing fault


def test_is_missing_ignores_non_client_errors() -> None:
    from astro_mine.cloud.artifacts.s3 import _is_missing

    assert _is_missing(RuntimeError("boom")) is False
