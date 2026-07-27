"""Ephemeris provider: a target's body-fixed position at an epoch.

Link borrows ephemerides from SPICE — "frames, epochs, body orientation, and orbits come
from SPICE/NAIF" (link.md §2.2) — through the shared :mod:`astro_mine.spice` foundation
(RFC-0002), never re-deriving them. The :class:`EphemerisProvider` protocol keeps the seam
injectable so the line-of-sight logic is testable without furnished kernels.
"""

from __future__ import annotations

from typing import Protocol

from astro_mine import spice
from astro_mine.core.units import Epoch, ReferenceFrame
from astro_mine.core.world import Vector
from astro_mine.link.geometry._errors import LinkGeometryError

__all__ = ["EphemerisProvider", "SpiceEphemeris"]


class EphemerisProvider(Protocol):
    """Resolves ``target``'s position in ``frame`` (SI metres, relative to the body centre)."""

    def position_body_fixed(self, target: str, epoch: Epoch, *, frame: ReferenceFrame) -> Vector:
        """The body-fixed position of ``target`` at ``epoch`` in ``frame``."""
        ...


class SpiceEphemeris:
    """:class:`EphemerisProvider` backed by :mod:`astro_mine.spice` (SPICE ``spkpos``).

    Resolves ``target`` relative to the central body of ``frame`` (e.g. ``MOON`` for
    ``MOON_BODY_FIXED``), expressed in ``frame``. The centre body defaults to ``frame.center``;
    pass ``body`` to override it. Kernels must be furnished first
    (:func:`astro_mine.spice.load_metakernel` / :func:`astro_mine.spice.kernel_pool`) — a
    missing kernel raises loudly rather than defaulting to a guessed position.
    """

    def __init__(self, *, body: str | None = None, abcorr: str | None = None) -> None:
        self._body = body
        self._abcorr = abcorr or spice.DEFAULT_ABCORR

    def position_body_fixed(self, target: str, epoch: Epoch, *, frame: ReferenceFrame) -> Vector:
        body = self._body or frame.center
        if not body:
            raise LinkGeometryError(
                f"frame {frame.name!r} has no centre body; cannot resolve an ephemeris "
                f"position for target {target!r}"
            )
        return spice.body_position(target, body, epoch, frame=frame, abcorr=self._abcorr)
