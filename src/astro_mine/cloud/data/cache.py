# SPDX-License-Identifier: Apache-2.0
"""Pull-through / locality cache -- keep hot chunks on local scratch.

For a sweep that re-reads the same slices across thousands of jobs, fetching every chunk from
the object store each time is the I/O fan-in bottleneck (``cloud.md`` §8). A
:class:`PullThroughCache` fronts a remote :class:`~astro_mine.cloud.data.objectstore.ObjectStore`
with a local scratch dir: the first read of a chunk fetches it from remote and writes it to
scratch; repeated reads are served locally. It records hit/miss counts so a test can prove the
cache serves repeats without touching remote -- the locality guarantee (``cloud.md`` §2
principle 7, §5).

Backlog: RM-P1-CLOUD-04 -- astro-mine-cloud#15
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from astro_mine.cloud.data.chunks import read_chunk

if TYPE_CHECKING:
    from astro_mine.cloud.data.chunks import ChunkRef
    from astro_mine.cloud.data.objectstore import ObjectStore

__all__ = ["PullThroughCache"]


class PullThroughCache:
    """A local read cache in front of a remote object store, keyed by chunk identity."""

    def __init__(self, remote: ObjectStore, scratch: str | os.PathLike[str]) -> None:
        self._remote = remote
        self._scratch = Path(scratch)
        self.hits = 0
        self.misses = 0

    def _cache_path(self, ref: ChunkRef) -> Path:
        # A stable name per chunk identity (key + byte range) so distinct chunks never collide.
        identity = f"{ref.key}:{ref.offset}:{ref.length}".encode()
        return self._scratch / hashlib.sha256(identity).hexdigest()

    def read_chunk(self, ref: ChunkRef) -> bytes:
        """Return the chunk for *ref*, serving from scratch when warm, else pulling through."""
        path = self._cache_path(ref)
        if path.exists():
            self.hits += 1
            return path.read_bytes()
        self.misses += 1
        data = read_chunk(self._remote, ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique per writer, not per process: two threads missing on the *same* chunk -- which is
        # exactly what a locality cache exists to serve -- would otherwise pick the same pid-named
        # scratch path, and the loser of the rename would die on ENOENT. The chunk is keyed by its
        # identity, so both writers store identical bytes and last-writer-wins is a no-op.
        tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)  # atomic: a concurrent reader never sees a half-written chunk
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return data

    def warm(self, ref: ChunkRef) -> bool:
        """Whether *ref* is already cached on local scratch."""
        return self._cache_path(ref).exists()
