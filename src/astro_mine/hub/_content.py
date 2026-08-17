# SPDX-License-Identifier: Apache-2.0
"""Canonical content hashing — the platform content-address form (mirrors ``core.hashing``).

Core's shared hashing primitive (``content_hash`` / ``canonical_json``, Core issue #19) landed on
Core ``main`` *after* the frozen ``v0.1.0`` tag Hub pins, so it is not importable from the pinned
dependency yet. This module reproduces that **exact** canonical form — the UTF-8 bytes of
``json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`` and a ``"sha256:<hex>"``
digest — so a Hub content address equals the platform address every other component computes
(``astro-mine-fleet`` rolls its own for the same reason, pinned to the same tag). Swap this module
for ``astro_mine.core.hashing`` once the pinned Core tag includes it (the API matches 1:1).

No IO, no heavy dependencies — only ``hashlib`` + ``json``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["HASH_ALGORITHM", "canonical_json", "content_hash", "content_hash_json"]

#: A content address is ``"sha256:" + hexdigest`` so the algorithm travels with the value.
HASH_ALGORITHM = "sha256"


def canonical_json(obj: Any) -> bytes:
    """Serialize ``obj`` to the platform canonical JSON byte form (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_hash(data: bytes) -> str:
    """The canonical content address of ``data``: ``"sha256:<hex>"``."""
    return f"{HASH_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def content_hash_json(obj: Any) -> str:
    """Content address of a JSON-serializable object: ``content_hash(canonical_json(obj))``."""
    return content_hash(canonical_json(obj))
