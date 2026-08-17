# SPDX-License-Identifier: Apache-2.0
"""Canonical content hashing — the platform's one content-address primitive (issue #19).

Core ships byte-stable serializers (``to_wire(deterministic=True)``) but no content-hash
function, so Bench (``ScenarioSpec`` resolver, RM-P0-BENCH-01), Worlds (``WorldSpec``
bundle, RM-P0-WORLDS-07), Fleet (OCI packaging) and Cloud (``RunContext``,
RM-P0-CLOUD-03) each otherwise roll their own hashing — risking "laptop hash != cluster
hash != Bench task identity". This module is the shared primitive they call so a content
address means the same thing everywhere.

- :func:`canonical_json` fixes the **canonical form** for structured data: the UTF-8 bytes
  of ``json.dumps`` with sorted keys, no inter-token whitespace, and non-ASCII preserved.
  It is the same canonical form already used platform-wide for content addressing (the
  schema-bundle ``schema_digest``; the determinism-trace digest in ``env/conformance``).
- :func:`content_hash` is the digest itself — ``"sha256:<hex>"`` over raw bytes — the same
  ``sha256:<hex>`` form the schema bundle's ``schema_digest`` and the ``Provenance``
  ``digest`` / ``input_hashes`` carriers already use, so the algorithm travels with the
  value.
- :func:`content_hash_json` composes the two for the common "hash this spec/dict" case.

No IO, no heavy dependencies (only ``hashlib`` + ``json``); additive (core.md §2.3).
Callers hash *canonical* bytes — raw file bytes, a byte-stable wire form, or
:func:`canonical_json` output — never ad-hoc ``str(obj)``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

__all__ = ["canonical_json", "content_hash", "content_hash_json", "manifest_digest"]

#: The digest algorithm. A content address is ``"sha256:" + hexdigest`` so the algorithm
#: travels with the value (matches the schema bundle and the ``Provenance`` carriers).
HASH_ALGORITHM = "sha256"


def canonical_json(obj: Any) -> bytes:
    """Serialize ``obj`` to the platform's canonical JSON byte form.

    Deterministic and stable across platforms: object keys sorted, no inter-token
    whitespace, non-ASCII characters preserved (UTF-8, ``ensure_ascii=False``). The same
    value always yields the same bytes, so a hash of these bytes is a portable content
    address. Raises ``TypeError`` on a non-JSON-serializable input (fail loud — never a
    silent or partial hash)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_hash(data: bytes) -> str:
    """Return the canonical content address of ``data``: ``"sha256:<hex>"``.

    The one digest form used platform-wide — the Core schema bundle's ``schema_digest``
    and the ``Provenance`` ``digest`` / ``input_hashes`` carriers all speak it. Hash
    *canonical* bytes (raw file bytes, a byte-stable ``to_wire`` form, or
    :func:`canonical_json` output) so the address is reproducible."""
    return f"{HASH_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def content_hash_json(obj: Any) -> str:
    """Content address of a JSON-serializable object: ``content_hash(canonical_json(obj))``.

    The convenience for the common case — pinning a ``ScenarioSpec`` / ``WorldSpec`` /
    config mapping by content — so every producer canonicalizes identically before
    hashing."""
    return content_hash(canonical_json(obj))


def manifest_digest(entries: Iterable[tuple[str, bytes]]) -> str:
    """Content address of a *set* of named blobs: ``(relpath, contents)`` pairs.

    The digest of a ``sha256sum``-style manifest — one ``"<sha256>  <relpath>\\n"`` line per
    entry, path-sorted — hashed as a whole. Order-independent and name-sensitive: the same
    files under the same names always yield the same digest, and renaming or dropping one
    changes it.

    This is the form the Core schema bundle's ``schema_digest`` speaks
    (:data:`astro_mine.core.SCHEMA_DIGEST`; ``VERSIONING.md`` §4.2), and it lives here — in
    the package — so ``scripts/build_schema_bundle.py`` and the shipped constant are the
    *same* computation rather than two hand-maintained ones that must agree byte-for-byte.
    """
    # Sorted by *relpath*, not by the formatted line — the manifest is a stable listing of
    # paths, and ordering it by hash instead would make the digest depend on content in a
    # second, hidden way.
    pairs = sorted((rel, hashlib.sha256(data).hexdigest()) for rel, data in entries)
    manifest = "".join(f"{sha}  {rel}\n" for rel, sha in pairs)
    return content_hash(manifest.encode("utf-8"))
