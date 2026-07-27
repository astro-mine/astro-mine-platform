"""Astro-Mine-Sim — the multi-physics engine and scenario runtime.

The deterministic stepping :mod:`~astro_mine.sim.runtime`, the
:mod:`~astro_mine.sim.engines` adapter framework, the multi-domain
:mod:`~astro_mine.sim.coupling`, the multi-fidelity :mod:`~astro_mine.sim.scheduler`,
:mod:`~astro_mine.sim.sensors`, :mod:`~astro_mine.sim.power_thermal` evolution,
:mod:`~astro_mine.sim.comms` masking, :mod:`~astro_mine.sim.recording` (MCAP +
provenance), and oracle :mod:`~astro_mine.sim.validation`.

Engine-pluralist, contract-singular: many engines route behind one Core Environment
API. See ``docs/architecture/sim.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"
