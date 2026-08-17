# SPDX-License-Identifier: Apache-2.0
"""Data locality -- bring data to compute, lazily.

Thousands of workers must stream only the slices they need from object storage, never
bulk-copy a multi-terabyte dataset to every node (``cloud.md`` §2 principle 7, §5, §8):

- :mod:`.objectstore` is one S3-compatible client (MinIO / S3 / GCS / Azure) plus a dep-free
  filesystem store, both supporting **byte-range reads** so a chunk fetch touches only its
  bytes;
- :mod:`.chunks` reads **Zarr / COG / Parquet** chunks lazily -- a Zarr chunk object or a
  byte-range tile/row-group -- with no bulk pre-copy;
- :mod:`.cache` is a **pull-through / locality cache** keeping hot chunks on local scratch, so
  repeated reads across a sweep are served locally;
- :mod:`.locality` pre-warms a shared cache and emits **co-schedule / zone** affinity hints.

Backlog: RM-P1-CLOUD-04 -- astro-mine-cloud#15
"""

from __future__ import annotations

from astro_mine.cloud.data.cache import PullThroughCache
from astro_mine.cloud.data.chunks import ChunkRef, RangeDataset, ZarrArray, read_chunk
from astro_mine.cloud.data.locality import co_schedule_affinity, prewarm, zone_affinity
from astro_mine.cloud.data.objectstore import FilesystemObjectStore, ObjectStore, S3ObjectStore

__all__ = [
    "ChunkRef",
    "FilesystemObjectStore",
    "ObjectStore",
    "PullThroughCache",
    "RangeDataset",
    "S3ObjectStore",
    "ZarrArray",
    "co_schedule_affinity",
    "prewarm",
    "read_chunk",
    "zone_affinity",
]
