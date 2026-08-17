# SPDX-License-Identifier: Apache-2.0
"""Seed / RNG stream manager (RM-P0-SIM-01).

Deterministic, dependency-light randomness for the stepping core. A single root seed fans
out into independent, named :class:`random.Random` streams via a SHA-256 of
``f"{root}:{name}"``, so per-agent (or per-purpose) draws are reproducible and mutually
independent without numpy. Same root seed ⇒ byte-identical streams ⇒ byte-identical traces
— the reproducibility contract (CX-REPRO).

Naming streams (rather than sharing one global generator) keeps each agent's draw sequence
stable even as agents are added, removed, or stepped in a different order elsewhere.
"""

from __future__ import annotations

import hashlib
import random

__all__ = ["RngStreams"]


class RngStreams:
    """A root-seeded fan-out of independent, named :class:`random.Random` streams."""

    def __init__(self, root_seed: int) -> None:
        self._root_seed = root_seed
        self._streams: dict[str, random.Random] = {}

    @property
    def root_seed(self) -> int:
        """The root seed every named stream is derived from."""
        return self._root_seed

    def stream(self, name: str) -> random.Random:
        """The named stream, created (and cached) deterministically on first request."""
        existing = self._streams.get(name)
        if existing is None:
            digest = hashlib.sha256(f"{self._root_seed}:{name}".encode()).digest()
            existing = random.Random(int.from_bytes(digest, "big"))
            self._streams[name] = existing
        return existing
