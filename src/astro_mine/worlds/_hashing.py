"""Where the line falls between *content* and *provenance* in a Worlds content hash.

Every Worlds product hashes the same way: the raw arrays, then the canonical JSON of the manifest
that describes them. This module owns the one question that shape leaves open — **which manifest
keys a hash covers**.

A content hash answers "are these the same bytes, produced from the same declared inputs?". So it
covers the arrays and the parameters that determine them. The **toolchain** that produced them is
provenance: recorded in every manifest, and deliberately *not* hashed.

Folding the toolchain in looked like extra rigour and was the opposite. A toolchain that changes the
output already changes the hash **through the bytes** — every Worlds hash covers its arrays, and the
bundle folds each field store's own content hash into ``world_hash`` — so the version strings
carried no signal a rebuild could not already see. What they did carry was churn:
``astro-mine-worlds`` is versioned by hatch-vcs, so its version tracks git commit distance
(``0.1.dev18`` -> ``0.1.dev24``), and a **bit-identical** world rebuilt one commit later minted a
different ``terrain_hash``, ``regolith_hash``, ``illumination_hash`` and ``world_hash``.
Content-addressing had quietly become commit-addressing: nobody could rebuild a published world from
its recorded recipe and land on its digest, which is exactly the reproducibility ``LUNAR-TR-004``
and ``conventions.md §11`` promise.

The symptom was already visible in the test suite before it was diagnosed — ``test_validation_psr``
declines to assert ``illumination_hash`` because "it folds in the toolchain". That assertion is now
possible (astro-mine-worlds#46).
"""

from __future__ import annotations

import json
from typing import Any

#: Manifest keys that are **recorded but never hashed** — provenance, not content.
#:
#: Kept as a set rather than inlined so the content/provenance split is stated once and every hash
#: in the package answers to the same rule.
PROVENANCE_KEYS = frozenset({"toolchain"})


def content_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """``meta`` minus the provenance keys — the part of a manifest a content hash covers."""
    return {key: value for key, value in meta.items() if key not in PROVENANCE_KEYS}


def canonical_meta_bytes(meta: dict[str, Any]) -> bytes:
    """The canonical JSON encoding of ``meta``'s **content** half, ready to feed a hash."""
    return json.dumps(content_meta(meta), sort_keys=True, ensure_ascii=False).encode("utf-8")
