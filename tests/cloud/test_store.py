"""FilesystemArtifactStore: content-addressed, idempotent, KeyError on miss."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from astro_mine.cloud.artifacts import addressing
from astro_mine.cloud.artifacts.store import (
    DEFAULT_ROOT,
    DEFAULT_ROOT_ENV,
    ArtifactStore,
    FilesystemArtifactStore,
)


def test_put_get_round_trip(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    address = store.put(b"payload")
    assert address == addressing.content_address(b"payload")
    assert store.get(address) == b"payload"


def test_put_is_content_addressed_layout(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    address = store.put(b"payload")
    hexdigest = addressing.hex_of(address)
    assert (tmp_path / "sha256" / hexdigest[:2] / hexdigest).read_bytes() == b"payload"


def test_put_is_idempotent(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    first = store.put(b"same")
    second = store.put(b"same")  # exercises the already-exists fast path
    assert first == second
    assert store.exists(first)


def test_get_missing_raises_keyerror(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    missing = addressing.content_address(b"never-stored")
    assert not store.exists(missing)
    with pytest.raises(KeyError):
        store.get(missing)


def test_root_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DEFAULT_ROOT_ENV, str(tmp_path / "envroot"))
    store = FilesystemArtifactStore()
    assert store.root == tmp_path / "envroot"


def test_root_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEFAULT_ROOT_ENV, raising=False)
    assert FilesystemArtifactStore().root == Path(DEFAULT_ROOT)


def test_satisfies_protocol(tmp_path: Path) -> None:
    assert isinstance(FilesystemArtifactStore(tmp_path), ArtifactStore)


def test_concurrent_identical_puts_all_succeed(tmp_path: Path) -> None:
    """Threads storing identical bytes must all succeed (cloud#25).

    The scratch file used to be named after the *process* id, and threads share one. Two threads
    storing identical bytes therefore chose the same scratch path: one renamed it over the target
    and the other died with ENOENT. Identical bytes are the common path, not a corner -- a submit
    re-stores the same ``uv.lock`` on every run, and Cloud's own sync ``POST /jobs`` handler runs in
    Starlette's threadpool, so two concurrent submits collide without any external fan-out at all.

    Fails reliably against the pid-named implementation.
    """
    store = FilesystemArtifactStore(tmp_path)
    payload = b"the identical bytes every job in a fan-out re-stores"
    workers = 16
    # Release every thread into put() at once, so they all miss the already-exists fast path and
    # contend on the scratch file rather than trickling through one at a time.
    barrier = threading.Barrier(workers, timeout=30)

    def put() -> str:
        barrier.wait()
        return store.put(payload)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        addresses = [future.result() for future in [pool.submit(put) for _ in range(workers)]]

    assert set(addresses) == {addressing.content_address(payload)}
    assert store.get(addresses[0]) == payload
    assert not list(tmp_path.rglob("*.tmp"))  # no scratch files left behind
