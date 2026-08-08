"""Applied capability-tag taxonomy — autonomy negotiation + export-control gating.

Fleet **applies** the Core-owned capability-tag vocabulary; it never widens it
(``fleet.md`` §2.1, §2.5, §11 "own the vocabulary in Core; Fleet only applies/validates
tags"). Two masters, one vocabulary (``fleet.md`` §5, §9):

- **autonomy negotiation** — the tags an asset advertises for
  [Mind](https://github.com/astro-mine/docs)/Allocate task↔asset matching;
- **export-control gating** — the same tags are first-class gating metadata; a
  *reserved/gated* tag (``operational_targeting``, ``ground_truth_access``,
  ``comms.live_mission_link_prediction``) must never appear on an open-commons asset.

:func:`as_tags` validates a set of applied tags against Core's closed
:class:`~astro_mine.core.sadf.enums.CapabilityTag` vocabulary (a missing tag is a **Core
RFC**, never a Fleet-private extension); :func:`assert_open_commons` is the publish-boundary
export-control gate Fleet applies before pushing to [Hub](https://github.com/astro-mine/docs)
(defense in depth — Core's loader already rejects gated tags, ``fleet.md`` §9).

Backlog: RM-P1-FLEET-10 -- astro-mine-fleet#21
"""

from __future__ import annotations

from collections.abc import Iterable

from astro_mine.core.sadf.enums import GATED_CAPABILITY_TAGS, CapabilityTag

__all__ = [
    "GATED_CAPABILITY_TAGS",
    "CapabilityError",
    "CapabilityTag",
    "as_tags",
    "assert_open_commons",
    "gated_tags",
]


class CapabilityError(ValueError):
    """A capability tag is not in Core's vocabulary, or is gated in the open commons."""


def as_tags(values: Iterable[CapabilityTag | str]) -> list[CapabilityTag]:
    """Coerce applied capability strings to Core :class:`CapabilityTag`s, preserving order.

    Raises :class:`CapabilityError` for any value outside Core's closed vocabulary — the
    signal to open a **Core RFC** rather than mint a Fleet-private tag (``fleet.md`` §2.5).
    """
    tags: list[CapabilityTag] = []
    for value in values:
        try:
            tags.append(CapabilityTag(value))
        except ValueError as exc:
            known = ", ".join(sorted(tag.value for tag in CapabilityTag))
            raise CapabilityError(
                f"capability tag {value!r} is not in Core's vocabulary; a new tag is a Core RFC, "
                f"not a Fleet extension (known: {known})"
            ) from exc
    return tags


def gated_tags(values: Iterable[CapabilityTag | str]) -> list[CapabilityTag]:
    """The reserved/gated tags among *values* (empty for a publishable open-commons asset)."""
    return [tag for tag in as_tags(values) if tag in GATED_CAPABILITY_TAGS]


def assert_open_commons(values: Iterable[CapabilityTag | str]) -> None:
    """Assert *values* carry no reserved/gated tag — the export-control publish gate.

    Genuinely sensitive/dual-use capabilities are partitioned out of the open library
    (``fleet.md`` §9; conventions.md §12). Raises :class:`CapabilityError` naming the
    offending tag(s); a clean set returns ``None``.
    """
    gated = gated_tags(values)
    if gated:
        names = ", ".join(sorted(tag.value for tag in gated))
        raise CapabilityError(
            f"asset declares reserved/gated capability tag(s) not permitted in the open "
            f"commons: {names} (RFC-0001 §6; conventions.md §12)"
        )
