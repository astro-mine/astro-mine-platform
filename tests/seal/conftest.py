"""Shared fixtures — an in-memory :class:`~astro_mine.seal.AttestationStore` and a keypair.

:class:`MemoryStore` is a complete, content-addressed store that knows nothing about OCI, Hub, or
any registry: it is a dict. That it can drive ``attest`` / ``verify`` unchanged is the proof that
those orchestrators are **registry-agnostic** (RM-P1-SEAL-03) — the same port Hub binds its OCI
``Registry`` to, and the same port Bench binds its submission store to.

It also raises its **own** error type (:class:`StoreIntegrityError`, which Seal has never heard of)
and can be told to :meth:`~MemoryStore.tamper` with stored bytes or to fail outright — so the
fail-closed tests exercise a genuinely foreign, genuinely hostile store.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from astro_mine.core.hashing import content_hash
from astro_mine.seal import generate_keypair


class StoreIntegrityError(Exception):
    """The store's own tamper error — a type Seal cannot import and must still fail closed on."""


class MemoryStore:
    """A content-addressed attestation store in a dict. Implements Seal's ``AttestationStore``."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.media: dict[str, str] = {}
        #: subject digest -> [(artifact_type, attestation digest)]
        self.attachments: dict[str, list[tuple[str, str]]] = {}
        #: when set, every port method raises this (a store that is down / hostile)
        self.fail_with: Exception | None = None

    # -- test-side helpers (not part of the port) ---------------------------------------------

    def put_artifact(self, data: bytes) -> str:
        """Store ``data`` at its content address and return the digest."""
        digest = content_hash(data)
        self.blobs[digest] = data
        return digest

    def tamper(self, digest: str, data: bytes) -> None:
        """Replace the bytes at ``digest`` *without* changing its address — a compromised store."""
        self.blobs[digest] = data

    def _guard(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    # -- the AttestationStore port -------------------------------------------------------------

    def attach_attestation(
        self, *, subject: str, artifact_type: str, media_type: str, payload: bytes
    ) -> str:
        self._guard()
        if subject not in self.blobs:
            raise KeyError(f"unknown subject {subject}")
        digest = content_hash(payload)
        self.blobs[digest] = payload
        self.media[digest] = media_type
        self.attachments.setdefault(subject, []).append((artifact_type, digest))
        return digest

    def attestation_digests(self, subject: str, *, artifact_type: str) -> Sequence[str]:
        self._guard()
        return [d for kind, d in self.attachments.get(subject, []) if kind == artifact_type]

    def read_attestation(self, digest: str) -> bytes:
        self._guard()
        self.verify_integrity(digest)
        return self.blobs[digest]

    def verify_integrity(self, subject: str) -> None:
        self._guard()
        if subject not in self.blobs:
            raise StoreIntegrityError(f"unknown digest {subject}")
        if content_hash(self.blobs[subject]) != subject:
            raise StoreIntegrityError(f"{subject} does not hash to its content address")


@pytest.fixture
def store() -> MemoryStore:
    """A fresh in-memory attestation store."""
    return MemoryStore()


@pytest.fixture
def keypair() -> tuple[bytes, bytes]:
    """A fresh ``(private_pem, public_pem)`` ECDSA P-256 keypair."""
    return generate_keypair()
