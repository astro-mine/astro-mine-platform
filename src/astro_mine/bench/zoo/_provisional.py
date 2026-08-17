# SPDX-License-Identifier: Apache-2.0
"""Provisional content-pin digests for the scenario zoo (pre-Hub).

During Phase-0 incubation no content producer (Worlds/Fleet/Prospect/Link) publishes a resolvable
content digest yet, and the Worlds bundle can only be hashed by building it from the real LOLA DEM +
SPICE (worlds.md §5). A zoo :class:`~astro_mine.bench.scenario.ScenarioSpec` must still pin *every*
content reference by hash for byte-for-byte reproducibility (bench.md §5; conventions.md §5), so the
anchor scenario pins the real, final content **ids** against **provisional** digests produced here.

A provisional digest is a well-formed ``sha256:`` over a canonical descriptor of the pin, so it is:

* **deterministic** — two clean checkouts resolve the identical scenario (the BENCH-01 criterion);
* **self-documenting** — it embeds the producer and the exact command that yields the real digest;
* **identity-sensitive** — changing the pinned id or content version changes the scenario hash,
  keeping the "changing any input changes the hash" property meaningful.

Each provisional digest is replaced one-for-one by the producer's published digest once the content
is Hub-published (Phase 1); the anchor scenario's ``PROVENANCE.md`` records the per-pin recipe. Both
flagship scenarios (the lunar anchor and the Phase-3 NEO sample-return) pin content this way before
Hub exists, so the helper is scenario-agnostic.

Backlog: RM-P0-BENCH-02 — astro-mine-bench#2
"""

from __future__ import annotations

from astro_mine.bench.scenario._hash import content_hash

__all__ = ["provisional_pin_hash"]


def provisional_pin_hash(
    *, producer: str, content_id: str, content_version: str, recipe: str
) -> str:
    """A deterministic, self-documenting provisional ``sha256:`` digest for an unpublished pin.

    ``producer`` is the content component (worlds/fleet/prospect/link); ``content_id`` the stable
    content identity being pinned; ``content_version`` its SemVer; and ``recipe`` the exact producer
    command that yields the *real* digest once the bundle is published. The digest is over the
    sorted descriptor, so it is order-independent and stable across runs.
    """
    return content_hash(
        {
            "provisional_content_pin": {
                "content_id": content_id,
                "content_version": content_version,
                "producer": producer,
                "recipe": recipe,
            }
        }
    )
