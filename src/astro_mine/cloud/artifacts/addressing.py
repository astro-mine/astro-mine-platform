"""Content addressing for Cloud artifacts.

Reuses the platform's de-facto content-hash convention -- canonical JSON + ``sha256``,
matching ``astro_mine.sim.runtime.provenance.content_digest`` -- so a structured
payload hashes identically across Sim and Cloud, and formats the result as an
OCI/Hub-aligned ``sha256:<hex>`` digest string (``hub.md`` §2.1). This is a
*Cloud-local* stopgap: it swaps for Core's shared content-hash helper (core#19) when
that ships, exactly as Sim documents. Cloud writes content-addressed bytes; it does
not own the registry/index (that is Hub's, ``cloud.md`` §5).

Backlog: RM-P0-CLOUD-03 -- https://github.com/astro-mine/astro-mine-cloud/issues/3
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "ALGORITHM",
    "content_address",
    "format_address",
    "hex_of",
    "parse_address",
]

ALGORITHM = "sha256"
_PREFIX = f"{ALGORITHM}:"
_HEX_LEN = 64
_HEX_CHARS = frozenset("0123456789abcdef")


def _canonical_json(payload: Any) -> bytes:
    """Encode *payload* as canonical JSON: sorted keys, no whitespace, ``str`` fallback.

    Byte-identical to Sim's ``content_digest`` canonical form, so equal content yields
    an equal digest regardless of mapping key order.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def content_address(payload: bytes | bytearray | Any) -> str:
    """Return the ``sha256:<hex>`` content address of *payload*.

    Raw ``bytes`` are hashed as-is (opaque artifact blobs); any other value is encoded
    as canonical JSON first (matching Sim), so the address is stable across runs and
    independent of mapping key order.
    """
    data = bytes(payload) if isinstance(payload, bytes | bytearray) else _canonical_json(payload)
    return format_address(hashlib.sha256(data).hexdigest())


def format_address(hexdigest: str) -> str:
    """Format a bare hex digest as an ``sha256:<hex>`` address."""
    _validate(ALGORITHM, hexdigest)
    return f"{_PREFIX}{hexdigest}"


def parse_address(address: str) -> tuple[str, str]:
    """Split an address into ``(algorithm, hexdigest)``.

    Accepts the canonical ``sha256:<hex>`` form and a bare 64-char hex string (Sim's
    form, assumed ``sha256``). Raises ``ValueError`` on an unknown algorithm or a
    malformed digest.
    """
    algorithm, _, hexdigest = address.partition(":") if ":" in address else (ALGORITHM, "", address)
    _validate(algorithm, hexdigest)
    return algorithm, hexdigest


def hex_of(address: str) -> str:
    """Return the bare hex digest of *address* (interop with Sim's bare-hex form)."""
    return parse_address(address)[1]


def _validate(algorithm: str, hexdigest: str) -> None:
    if algorithm != ALGORITHM:
        raise ValueError(f"unsupported digest algorithm {algorithm!r}; expected {ALGORITHM!r}")
    if len(hexdigest) != _HEX_LEN or not _HEX_CHARS.issuperset(hexdigest):
        raise ValueError(f"malformed {ALGORITHM} hex digest: {hexdigest!r}")
