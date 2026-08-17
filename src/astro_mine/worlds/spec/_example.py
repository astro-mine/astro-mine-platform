# SPDX-License-Identifier: Apache-2.0
"""The shipped ``WorldSpec`` example — the copyable YAML the front door always implied (G2.11).

``WorldSpec`` has read YAML since it existed (:meth:`~astro_mine.worlds.spec.WorldSpec.from_yaml`)
and shipped none: the only real spec was authored in Python inside the anchor build script, so a
user had nothing to start from and nothing to check what they wrote against (UC-C5).

**It is package data, not an ``examples/`` directory.** A file outside the wheel reaches repo
cloners and nobody else, which is the same gap Guard's ``anchor.safety.yaml`` had (G2.7). Read it
through :func:`example_world_spec_text` so it resolves from an installed wheel too.

**It is a synthetic world, deliberately, and not the anchor.** The anchor's ``region`` is derived
from the ingested LOLA DEM's grid and its ``source_dem`` is pinned by the digest of a file this
repo does not contain, so it cannot be authored as a static document at all. What a starting spec
should be is small, complete, and honest about what is synthetic — which is what ``SourceRef``
already anticipates in documenting ``content_hash: None`` as the synthetic/illustrative case.

The same text backs `astro-mine new world` (RFC-0011 §7), so the example and the scaffold cannot
drift into disagreeing about what a WorldSpec looks like: there is one document, and the scaffold
substitutes an identity into it.
"""

from __future__ import annotations

from importlib import resources

__all__ = ["EXAMPLE_RESOURCE", "example_world_spec_text"]

#: The example's path within this package's data.
EXAMPLE_RESOURCE = "examples/synthetic_polar.world.yaml"

#: Anchored on ``spec`` rather than on ``spec.examples`` deliberately: the example directory holds
#: data, not code, so it has no ``__init__.py`` and is not a package to anchor on.
_ANCHOR = "astro_mine.worlds.spec"

#: The identity lines the scaffold substitutes. Anchored to the line start so a value appearing
#: elsewhere in the document — in a comment, or in the description — is never rewritten by accident.
_WORLD_ID_LINE = "world_id: example-polar-basin"
_VERSION_LINE = "version: 0.1.0"


def example_world_spec_text(*, world_id: str | None = None, version: str | None = None) -> str:
    """The example WorldSpec as YAML, optionally with a different identity substituted.

    With no arguments this is the shipped document byte-for-byte — the thing a user reads and
    copies. With ``world_id``/``version`` it is the same document under a new name, which is what
    `astro-mine new world` writes: a scaffold that emitted a *different* document from the one the
    docs point at would be two examples to keep valid, and they would drift.

    Read through :mod:`importlib.resources` rather than by path, so it resolves identically from a
    source checkout and from an installed wheel (and from a zipped one).
    """
    text = resources.files(_ANCHOR).joinpath(EXAMPLE_RESOURCE).read_text(encoding="utf-8")
    if world_id is not None:
        text = text.replace(_WORLD_ID_LINE, f"world_id: {world_id}", 1)
    if version is not None:
        text = text.replace(_VERSION_LINE, f"version: {version}", 1)
    return text
