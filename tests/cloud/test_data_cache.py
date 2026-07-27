"""Pull-through / locality cache -- repeated reads served from local scratch."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from astro_mine.cloud.data.cache import PullThroughCache
from astro_mine.cloud.data.chunks import ChunkRef


class CountingStore:
    """An ObjectStore that counts remote fetches, to prove the cache absorbs repeats."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects
        self.remote_fetches = 0

    def get(self, key: str) -> bytes:
        self.remote_fetches += 1
        return self._objects[key]

    def get_range(self, key: str, start: int, length: int) -> bytes:
        self.remote_fetches += 1
        return self._objects[key][start : start + length]

    def size(self, key: str) -> int:
        return len(self._objects[key])


def test_second_read_is_served_locally(tmp_path: Path) -> None:
    remote = CountingStore({"chunk": b"payload"})
    cache = PullThroughCache(remote, tmp_path)
    ref = ChunkRef(key="chunk")

    assert cache.read_chunk(ref) == b"payload"  # miss -> remote
    assert cache.read_chunk(ref) == b"payload"  # hit -> local

    assert (cache.misses, cache.hits) == (1, 1)
    assert remote.remote_fetches == 1  # remote touched exactly once
    assert cache.warm(ref) is True


def test_distinct_ranges_are_cached_separately(tmp_path: Path) -> None:
    remote = CountingStore({"blob": bytes(range(16))})
    cache = PullThroughCache(remote, tmp_path)
    a = ChunkRef(key="blob", offset=0, length=4)
    b = ChunkRef(key="blob", offset=4, length=4)

    assert cache.read_chunk(a) == bytes(range(0, 4))
    assert cache.read_chunk(b) == bytes(range(4, 8))
    assert cache.read_chunk(a) == bytes(range(0, 4))  # a is warm now

    assert remote.remote_fetches == 2  # a and b each fetched once
    assert cache.misses == 2 and cache.hits == 1


def test_cold_ref_is_not_warm(tmp_path: Path) -> None:
    cache = PullThroughCache(CountingStore({"k": b"v"}), tmp_path)
    assert cache.warm(ChunkRef(key="k")) is False


def test_concurrent_misses_on_one_chunk_all_succeed(tmp_path: Path) -> None:
    """Threads racing on the *same* cold chunk must all succeed (cloud#25).

    The scratch file was named after the process id, which threads share -- so two threads missing
    on one chunk picked the same scratch path and the loser of the rename died with ENOENT. Many
    readers converging on one hot chunk is what a locality cache is *for* (``cloud.md`` §5), so this
    is the cache's design case, not an edge.

    Fails reliably against the pid-named implementation.
    """
    remote = CountingStore({"chunk": b"the hot slice a whole sweep re-reads"})
    cache = PullThroughCache(remote, tmp_path)
    ref = ChunkRef(key="chunk")
    workers = 16
    barrier = threading.Barrier(workers, timeout=30)

    def read() -> bytes:
        barrier.wait()  # all threads miss together, then contend on the scratch file
        return cache.read_chunk(ref)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        payloads = [future.result() for future in [pool.submit(read) for _ in range(workers)]]

    assert set(payloads) == {b"the hot slice a whole sweep re-reads"}
    assert cache.warm(ref) is True
    assert not list(tmp_path.rglob("*.tmp"))  # no scratch files left behind
