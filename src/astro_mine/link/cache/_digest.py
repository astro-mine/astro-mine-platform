"""Content addressing: the digests + cache key Link's reproducibility is built on (LINK-05).

Determinism is a hard requirement (conventions.md §1.5): same SPICE kernels + same terrain +
same node set + same epoch window + same config ⇒ identical connectivity products. This
module makes that concrete — a stable content hash over the pinned inputs
(:class:`CacheKey`, keyed on ``{kernels, terrain, nodes, epoch, config, link_version}`` per
link.md §5) and the content address of a product itself (:func:`plan_digest`, over Core's
byte-stable ``contact_plan_to_wire``). Kernel/DEM files are hashed by **content**
(:func:`hash_file`), so a wrong or truncated kernel silently changing visibility changes the
key — a correctness-and-trust guard, not just a cache miss (link.md §9).

Backlog: RM-P0-LINK-05 -- astro-mine-link#5
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from astro_mine.core.messages import ContactPlan, contact_plan_to_wire
from astro_mine.link import __version__ as LINK_VERSION
from astro_mine.link.cache._errors import LinkCacheError

__all__ = [
    "CacheKey",
    "build_cache_key",
    "cache_key",
    "canonical_digest",
    "hash_file",
    "plan_digest",
]

_CHUNK = 1 << 20  # 1 MiB — stream large kernels rather than reading them whole


def _json_default(obj: object) -> Any:
    """Canonicalize the value types Link inputs carry into JSON-native form."""
    if isinstance(obj, BaseModel):  # Core units/messages models (Epoch, ReferenceFrame, ...)
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):  # SurfaceNode, GroundStation
        return dataclasses.asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, frozenset | set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    raise TypeError(f"cannot canonicalize object of type {type(obj).__name__}")


def canonical_digest(obj: Any) -> str:
    """The SHA-256 of ``obj``'s canonical JSON — stable across dict order and Python runs.

    Keys are sorted and whitespace stripped, so two structurally-equal inputs hash
    identically. Pydantic models, dataclasses, enums, sets, and paths canonicalize through
    :func:`_json_default`; anything else must already be JSON-native or a ``TypeError`` is
    raised.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    """The SHA-256 of a file's bytes — the content address of a kernel or DEM input.

    Streams the file in chunks so multi-megabyte SPK/DEM inputs do not load whole. A missing
    path raises :class:`LinkCacheError`: an un-hashable pinned input must fail loudly, never
    hash to a default (link.md §9).
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise LinkCacheError(
            f"cannot hash missing cache input: {resolved} — a pinned kernel/DEM must exist"
        )
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_digest(plan: ContactPlan) -> str:
    """The content address of a :class:`ContactPlan` — SHA-256 of its byte-stable wire form.

    Core's ``contact_plan_to_wire`` serializes deterministically, so an identical plan always
    yields the same digest — the determinism check behind a reproducible comms-denied
    benchmark (link.md §5; conventions.md §1.5).
    """
    return hashlib.sha256(contact_plan_to_wire(plan)).hexdigest()


def _files_digest(paths: Iterable[str | Path]) -> str:
    """A digest over a set of files, hashed by content and order-independent."""
    return canonical_digest(sorted(hash_file(path) for path in paths))


@dataclass(frozen=True, slots=True)
class CacheKey:
    """The pinned-input identity of a Link computation (link.md §5).

    Each field is a digest string over one input class — SPICE ``kernels`` and ``terrain``
    DEM/horizon files by content, the ``nodes`` set (with SADF radios), the ``epoch`` window
    + sampling, and the fidelity ``config`` — plus the ``link_version`` that produced it. Two
    runs share a :attr:`digest` iff every pinned input matches, which is exactly when their
    products must be byte-for-byte identical.
    """

    kernels: str
    terrain: str
    nodes: str
    epoch: str
    config: str
    link_version: str = LINK_VERSION

    @property
    def digest(self) -> str:
        """The single content-address hash over all pinned-input digests."""
        return canonical_digest(dataclasses.asdict(self))


def cache_key(inputs: CacheKey) -> str:
    """The content-addressed cache key (digest) for a Link computation's pinned inputs."""
    return inputs.digest


def build_cache_key(
    *,
    kernels: Iterable[str | Path] = (),
    terrain: Iterable[str | Path] = (),
    nodes: Any = None,
    epoch: Any = None,
    config: Any = None,
) -> CacheKey:
    """Assemble a :class:`CacheKey` from raw inputs.

    ``kernels`` and ``terrain`` are file paths, hashed by **content** (so a re-downloaded but
    identical kernel still hits cache, and a truncated one misses). ``nodes`` / ``epoch`` /
    ``config`` are any canonicalizable inputs — e.g. the node dataclasses + SADF radios, the
    :class:`~astro_mine.core.units.EpochWindow` and step, and the fidelity/mask settings —
    hashed via :func:`canonical_digest`.
    """
    return CacheKey(
        kernels=_files_digest(kernels),
        terrain=_files_digest(terrain),
        nodes=canonical_digest(nodes),
        epoch=canonical_digest(epoch),
        config=canonical_digest(config),
    )
