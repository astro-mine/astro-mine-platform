"""Astro-Mine-Spice — the shared SPICE foundation (RFC-0002).

Resolves Core's units/frames vocabulary through SPICE — the name->geometry step Core
defers (``core.md §2.3`` forbids heavy deps) — so every consumer (Worlds illumination/PSR,
Link LOS/contact windows, Sim's orbital engine, Transit) shares one SPICE implementation
instead of re-deriving it or depending on Worlds' geospatial stack.

Two layers: the **raw primitives** (kernel pool, :func:`et`, :func:`body_position`,
:func:`frame_transform`) a consumer drives over the furnished pool (e.g. Link's own
``gfposc`` window search), and the **topocentric helpers** (:func:`sun_geometry`,
:func:`earth_geometry`) that return the elevation/azimuth scalar consumers threshold
against terrain horizons. Window search and terrain occlusion live in the consumers,
not here.

See ``docs/architecture/spice.md`` and ``docs/rfc/0002-shared-spice-foundation.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from astro_mine.spice._geometry import (
    DEFAULT_ABCORR,
    EARTH_RADIUS_M,
    MOON_RADIUS_M,
    BodyGeometry,
    Site,
    SpiceGeometryError,
    body_geometry,
    body_position,
    earth_geometry,
    epoch_from_utc,
    epoch_range,
    et,
    frame_transform,
    sun_geometry,
)
from astro_mine.spice._kernels import (
    SpiceKernelError,
    clear_kernels,
    kernel_pool,
    load_metakernel,
)

__all__ = [
    "DEFAULT_ABCORR",
    "EARTH_RADIUS_M",
    "MOON_RADIUS_M",
    "BodyGeometry",
    "Site",
    "SpiceGeometryError",
    "SpiceKernelError",
    "__version__",
    "body_geometry",
    "body_position",
    "clear_kernels",
    "earth_geometry",
    "epoch_from_utc",
    "epoch_range",
    "et",
    "frame_transform",
    "kernel_pool",
    "load_metakernel",
    "sun_geometry",
]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"
