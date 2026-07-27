"""Astro-Mine-Fleet — SADF asset library and authoring toolchain.

The authoring :mod:`~astro_mine.fleet.cli`, :mod:`~astro_mine.fleet.importers`
(URDF/SDF + USD/glTF geometry), physical-plausibility :mod:`~astro_mine.fleet.lint`,
the reference asset :mod:`~astro_mine.fleet.library`, multi-fidelity
:mod:`~astro_mine.fleet.fidelity` profiles, and content-addressed
:mod:`~astro_mine.fleet.packaging`.

Fleet consumes the Core SADF; it never widens the waist. See
``docs/architecture/fleet.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"

# The Core interface versions Fleet is built against — advertised here so consumers and
# the contract test cite one source of truth (defined in :mod:`astro_mine.fleet._core`).
from astro_mine.fleet._core import CORE_INTERFACES

__all__ = ["CORE_INTERFACES", "__version__"]
