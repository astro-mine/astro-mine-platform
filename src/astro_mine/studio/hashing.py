# SPDX-License-Identifier: Apache-2.0
"""Content-addressing primitive — canonical JSON + SHA-256.

The frozen Core ``v0.1.0`` interface Studio pins does **not** ship a content-hash
helper (it arrives in a later Core minor), so Studio carries its own, deliberately
identical to the platform convention (``sha256:<hex>`` over sorted-key, whitespace-
free UTF-8 JSON; conventions.md §5) so a digest computed here matches one computed by
Hub/Cloud and stays valid when Core later absorbs the helper. No I/O, no heavy deps.

For an ``ObjectiveDocument`` prefer hashing its byte-stable Protobuf wire form
(``astro_mine.core.objective.to_wire``) via :func:`content_hash` — that is the
canonical cross-component identity; :func:`content_hash_json` is for Studio-local
models that have no wire form.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

HASH_ALGORITHM = "sha256"


def canonical_json(obj: Any) -> bytes:
    """Deterministic UTF-8 JSON bytes: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_hash(data: bytes) -> str:
    """``sha256:<hex>`` of raw bytes (a file, or a byte-stable wire form)."""
    return f"{HASH_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def content_hash_json(obj: Any) -> str:
    """``sha256:<hex>`` of the canonical JSON encoding of ``obj``."""
    return content_hash(canonical_json(obj))
