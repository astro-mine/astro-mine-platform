"""Content-addressed artifact stores — Cloud's backends for the Core contract.

The :class:`~astro_mine.core.artifacts.ArtifactStore` *protocol* is Core's (conventions.md
§3.3): Bench types against it too, and a shape two components share belongs at the waist. What
is here is every part of it that needs a filesystem or a network — the backends, the address
scheme, and the root convention.

A store reads and writes opaque byte payloads keyed by their content
address (:func:`~astro_mine.cloud.artifacts.addressing.content_address`).
:class:`FilesystemArtifactStore` is the default, dependency-free local backend -- it
keeps the sacred single-workstation tier working with no daemon, no cloud, and no
account (CX-LOCAL). The S3-compatible backend lives in
:mod:`astro_mine.cloud.artifacts.s3`.

Layout: ``<root>/sha256/<h[:2]>/<h>`` -- an OCI/git-style two-char fanout keyed by the
hex digest (no key layout is mandated by the docs; this is the chosen convention).
Writes are atomic and idempotent: storing identical bytes twice is a no-op that yields
the same address, and concurrent writers of identical bytes all succeed.

Backlog: RM-P0-CLOUD-03 -- astro-mine-cloud#3
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from astro_mine.cloud.artifacts.addressing import content_address, parse_address

__all__ = [
    "DEFAULT_ROOT",
    "DEFAULT_ROOT_ENV",
    "FilesystemArtifactStore",
]

#: Environment variable that overrides the default filesystem-store root, so an
#: in-process run and a ``docker compose`` container can share one store by pointing
#: at the same path -- the CLOUD-02 backend-equivalence contract.
DEFAULT_ROOT_ENV = "ASTRO_MINE_ARTIFACT_ROOT"
DEFAULT_ROOT = ".astro-mine/artifacts"


class FilesystemArtifactStore:
    """Content-addressed store rooted at a local directory.

    The root is *root* if given, else the ``ASTRO_MINE_ARTIFACT_ROOT`` environment
    variable, else ``./.astro-mine/artifacts``.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        if root is None:
            root = os.environ.get(DEFAULT_ROOT_ENV, DEFAULT_ROOT)
        self.root = Path(root)

    def _path(self, address: str) -> Path:
        _, hexdigest = parse_address(address)
        return self.root / "sha256" / hexdigest[:2] / hexdigest

    def put(self, data: bytes) -> str:
        address = content_address(data)
        path = self._path(address)
        if path.exists():
            return address
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp name then atomically rename, so a concurrent reader never observes a
        # half-written object at the content-addressed path. The name is unique per *writer*, not
        # per process: threads share a pid, so naming the scratch file after it made two threads
        # storing identical bytes pick the same path -- one renamed it away and the other died on
        # ENOENT. Identical bytes are the common case (a lockfile re-stored on every run), not a
        # corner. Both writers now rename their own file over the same content-addressed target;
        # the payloads are identical by construction, so last-writer-wins is a no-op.
        tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return address

    def get(self, address: str) -> bytes:
        try:
            return self._path(address).read_bytes()
        except FileNotFoundError:
            raise KeyError(address) from None

    def exists(self, address: str) -> bool:
        return self._path(address).exists()
