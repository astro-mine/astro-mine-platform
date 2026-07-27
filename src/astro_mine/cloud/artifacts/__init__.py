"""Content-addressed artifact I/O + RunContext provenance envelope.

S3-compatible (MinIO local) content-addressed artifact I/O plus a ``RunContext``
provenance envelope, so a future scaled run reproduces a laptop run. The default
:class:`FilesystemArtifactStore` is dependency-free and keeps the local tier sacred
(CX-LOCAL); :class:`~astro_mine.cloud.artifacts.s3.S3ArtifactStore` (the optional
``s3`` extra) runs the same content-addressed I/O against MinIO or any S3 backend.

Backlog: RM-P0-CLOUD-03 -- https://github.com/astro-mine/astro-mine-cloud/issues/3
"""

from __future__ import annotations

from astro_mine.cloud.artifacts.addressing import (
    ALGORITHM,
    content_address,
    format_address,
    hex_of,
    parse_address,
)
from astro_mine.cloud.artifacts.runcontext import (
    EnvironmentFingerprint,
    RunContext,
    code_version,
)
from astro_mine.cloud.artifacts.s3 import S3ArtifactStore
from astro_mine.cloud.artifacts.store import (
    DEFAULT_ROOT,
    DEFAULT_ROOT_ENV,
    ArtifactStore,
    FilesystemArtifactStore,
)

__all__ = [
    "ALGORITHM",
    "DEFAULT_ROOT",
    "DEFAULT_ROOT_ENV",
    "ArtifactStore",
    "EnvironmentFingerprint",
    "FilesystemArtifactStore",
    "RunContext",
    "S3ArtifactStore",
    "code_version",
    "content_address",
    "format_address",
    "hex_of",
    "parse_address",
]
