"""Content-addressed byte storage — the contract, not a backend.

An :class:`ArtifactStore` is three methods over opaque bytes keyed by their own digest: ``put``
returns the address, ``get`` round-trips, ``exists`` asks. Nothing about it is specific to a
filesystem, to S3, or to the component that first happened to implement it.

**Why this is at the waist.** It was declared in ``astro_mine.cloud.artifacts.store``, and Bench
imported it from there to type the store its evaluator is handed — an import Bench made under
``TYPE_CHECKING`` precisely because it did not want the dependency. conventions.md §3.3 names that
shape exactly, and names this very import: "``from astro_mine.cloud.artifacts.store import
ArtifactStore`` is a Core Protocol wearing a component's address." Two components share it, so it
is Core's by the same rule that governs schemas (§3.1), and Cloud keeps
what is genuinely Cloud's — :class:`~astro_mine.cloud.artifacts.FilesystemArtifactStore`, the
S3 backend, the addressing scheme, and the default root convention.

Core declares the shape and implements none of it. There is no store in this module and there will
not be one: a backend needs a filesystem layout or a network client, and the waist holds neither
(core.md §2 principle 3).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["ArtifactStore"]


@runtime_checkable
class ArtifactStore(Protocol):
    """A content-addressed byte store: ``put`` returns the address, ``get`` round-trips.

    An *address* is ``sha256:<hex>`` over the stored bytes, which is what makes ``put`` idempotent
    — storing identical bytes twice is a no-op yielding the same address — and what lets a
    consumer verify what it received without trusting where it came from.
    """

    def put(self, data: bytes) -> str:
        """Store *data* and return its ``sha256:<hex>`` content address."""
        ...

    def get(self, address: str) -> bytes:
        """Return the bytes stored at *address*; raise ``KeyError`` if absent."""
        ...

    def exists(self, address: str) -> bool:
        """Return whether an object is stored at *address*."""
        ...
