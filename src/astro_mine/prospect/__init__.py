"""Astro-Mine-Prospect — probabilistic resource fields with explicit uncertainty.

Water-ice and mineral concentration as geostatistical distributions: the
:mod:`~astro_mine.prospect.field` contract (uncertainty-first), inference
:mod:`~astro_mine.prospect.backends` (GP / grid), public-data
:mod:`~astro_mine.prospect.priors`, the sealed ground-truth and updatable
:mod:`~astro_mine.prospect.belief` fields, their :mod:`~astro_mine.prospect.isolation`,
:mod:`~astro_mine.prospect.infogain` maps for active perception, and a
:mod:`~astro_mine.prospect.calibration` gate.

Uncertainty is the product, not a footnote. See ``docs/architecture/prospect.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"
