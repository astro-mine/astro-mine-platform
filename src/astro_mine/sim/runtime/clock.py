# SPDX-License-Identifier: Apache-2.0
"""Deterministic simulation clock (RM-P0-SIM-01).

A monotonic, fixed-base-rate clock expressed in Core's typed time primitives. It tracks
the integer ``tick`` and elapsed ``sim_time_s`` and renders the absolute SPICE epoch
(:class:`~astro_mine.core.units.Epoch`, TDB) as ``start + sim_time_s``.

**Dependency-light by construction.** The clock *names* TDB seconds and never calls SPICE
— absolute-time resolution is Worlds' job (RM-P0-WORLDS-02). Multi-rate engine
sub-stepping is internal to the engine layer (later RM-P0-SIM items); here the clock is
the single base rate the episode loop advances. It is an immutable value: :meth:`advanced`
returns the next tick rather than mutating in place, so a clock is freely shareable and
trivially reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from astro_mine.core.units import Epoch

__all__ = ["SimClock"]


@dataclass(frozen=True, slots=True)
class SimClock:
    """An immutable base-rate clock over a :class:`~astro_mine.core.units.Epoch` origin."""

    start_epoch: Epoch
    dt_s: float
    tick: int = 0
    sim_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {self.dt_s}")

    def now_epoch(self) -> Epoch:
        """The absolute TDB epoch at the current ``sim_time_s`` (``start + elapsed``)."""
        return Epoch(
            tdb_seconds=self.start_epoch.tdb_seconds + self.sim_time_s,
            scale=self.start_epoch.scale,
        )

    def advanced(self, *, dt_s: float | None = None) -> SimClock:
        """The clock one tick later. ``dt_s`` overrides the base rate for a variable step;
        otherwise the base rate advances both ``tick`` and ``sim_time_s``."""
        step = self.dt_s if dt_s is None else dt_s
        return replace(self, tick=self.tick + 1, sim_time_s=self.sim_time_s + step)
