"""Canonical content-hashing for scenario artifacts (conventions.md §5).

The platform content-addresses every reproducible artifact with a ``sha256`` over a
*canonical* serialization — key-sorted, compact JSON — so the digest is independent of
formatting or field order (worlds.md §5; ``astro_mine.worlds.spec`` uses the identical
formulation). Bench applies it twice — once for a ``ScenarioSpec``'s own ``spec_hash`` and
once for the resolved scenario identity — so the formula lives here.

Backlog: RM-P0-BENCH-01 — https://github.com/astro-mine/astro-mine-bench/issues/1
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["HASH_PATTERN", "canonical_json", "content_hash", "normalize_sha256"]

#: The platform digest form: ``sha256:`` followed by a 64-character lowercase hex digest
#: (worlds.md §5; fleet packaging). Matched by :data:`HASH_PATTERN`.
HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"

_BARE_SHA256 = r"[0-9A-Fa-f]{64}"


def canonical_json(payload: Any) -> str:
    """The canonical, key-sorted, compact JSON encoding used for content-addressing.

    Separators drop insignificant whitespace and ``sort_keys`` fixes field order, so the
    encoding — and therefore the digest — is stable across authoring and Python runs.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(payload: Any) -> str:
    """A deterministic ``sha256:<hex>`` digest over the canonical JSON of ``payload``."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def normalize_sha256(value: str) -> str:
    """Normalize a content-hash reference to the canonical ``sha256:<lowercase-hex>`` form.

    Accepts either the prefixed platform form (``sha256:<hex>``, as Worlds/Fleet emit) or a
    bare 64-char hex digest (as Prospect emits), so a ScenarioSpec can reference content from
    any producer. Fails loudly on anything else — a malformed pin must never resolve silently.
    """
    import re

    text = value.strip()
    candidate = text[len("sha256:") :] if text.lower().startswith("sha256:") else text
    if not re.fullmatch(_BARE_SHA256, candidate):
        raise ValueError(f"not a sha256 content hash: {value!r}")
    return f"sha256:{candidate.lower()}"
