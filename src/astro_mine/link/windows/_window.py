# SPDX-License-Identifier: Apache-2.0
"""The contact window: a continuous interval of visibility between two nodes.

A :class:`ContactWindow` is the reduced product of the time-varying visibility series —
the half-open epoch interval ``[start, end)`` over which an ordered ``observer -> target``
pair stays connected. Connectivity is a function of epoch by construction; the interval is
the first-class product, and any boolean snapshot is derived from it, never the other way
around (link.md §2.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.core.units import Epoch

__all__ = ["ContactWindow"]


@dataclass(frozen=True)
class ContactWindow:
    """A continuous ``[start, end)`` interval of visibility for an ``observer -> target`` pair.

    ``start``/``end`` are Core :class:`~astro_mine.core.units.Epoch`s in TDB; ``end`` is the
    first epoch at which the pair is no longer visible (or the search-window end, if the
    contact is still open there). A window is always non-empty: ``end`` is strictly after
    ``start``.
    """

    observer: str
    target: str
    start: Epoch
    end: Epoch

    @property
    def duration_s(self) -> float:
        """The window length in SI seconds (``end - start`` in TDB)."""
        return self.end.tdb_seconds - self.start.tdb_seconds
