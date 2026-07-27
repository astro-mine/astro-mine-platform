"""Astro-Mine-Worlds — celestial-body environment models.

Real planetary data in, simulatable world out: terrain from DEMs
(:mod:`~astro_mine.worlds.terrain`), illumination + PSR detection
(:mod:`~astro_mine.worlds.illumination`), surface thermal
(:mod:`~astro_mine.worlds.thermal`), regolith terramechanics parameters
(:mod:`~astro_mine.worlds.regolith`), the Environment-API world provider and
LOS/occlusion service (:mod:`~astro_mine.worlds.provider`), and the WorldSpec
bundle (:mod:`~astro_mine.worlds.spec`). SPICE frames/epochs/geometry come from
the shared :mod:`astro_mine.spice` (RFC-0002).

Phase 0 builds the lunar south-polar (Shackleton-de Gerlache) world only. See
``docs/architecture/worlds.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"
