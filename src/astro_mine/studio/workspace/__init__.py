"""Workspace — Studio's owned design state + audit log (studio.md §5).

Studio owns projects/designs/audit; it does **not** own world, asset, policy, or
result bytes (those belong to siblings and are referenced by content hash). Produced
artifacts are stored **content-addressed and fail-closed** — a payload whose bytes do
not match its claimed digest is rejected, never stored. The audit log records the
observational facts kept out of the (deterministic) provenance envelope: who authored
what, and which LLM model/version (if any) drafted a spec.

Phase-1 ships the in-memory tier-1 store so the library runs on a laptop with no
database (conventions.md §7). The production PostgreSQL(+pgvector) backend
(studio.md §5, §11) implements the same :class:`WorkspaceStore` Protocol and is a
deployment concern, deferred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import Field

from .._base import FrozenStudioModel
from ..hashing import content_hash


class WorkspaceError(RuntimeError):
    """A workspace integrity failure (digest mismatch, or a missing artifact)."""


class AuditEntry(FrozenStudioModel):
    """One audit-log record. ``seq`` is a monotonic in-workspace ordinal (not a
    wall-clock time) so the log is deterministic in tests and reproducible runs."""

    seq: int
    digest: str
    artifact_type: str
    author: str
    action: str = "created"
    model: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class WorkspaceStore(Protocol):
    """The persistence seam Studio artifacts are frozen into."""

    def put(
        self,
        artifact_type: str,
        digest: str,
        payload: bytes,
        *,
        author: str,
        action: str = "created",
        model: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None: ...

    def get(self, digest: str) -> bytes: ...

    def has(self, digest: str) -> bool: ...

    def audit(self) -> Sequence[AuditEntry]: ...


class InMemoryWorkspace:
    """Tier-1 in-process :class:`WorkspaceStore` — content-addressed, fail-closed."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}
        self._audit: list[AuditEntry] = []

    def put(
        self,
        artifact_type: str,
        digest: str,
        payload: bytes,
        *,
        author: str,
        action: str = "created",
        model: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        actual = content_hash(payload)
        if actual != digest:
            raise WorkspaceError(
                f"digest mismatch for {artifact_type}: claimed {digest}, got {actual}"
            )
        self._blobs.setdefault(digest, payload)
        self._audit.append(
            AuditEntry(
                seq=len(self._audit),
                digest=digest,
                artifact_type=artifact_type,
                author=author,
                action=action,
                model=model,
                metadata=dict(metadata or {}),
            )
        )

    def get(self, digest: str) -> bytes:
        try:
            return self._blobs[digest]
        except KeyError as exc:
            raise WorkspaceError(f"no artifact for digest {digest}") from exc

    def has(self, digest: str) -> bool:
        return digest in self._blobs

    def audit(self) -> Sequence[AuditEntry]:
        return tuple(self._audit)


__all__ = [
    "AuditEntry",
    "InMemoryWorkspace",
    "WorkspaceError",
    "WorkspaceStore",
]
