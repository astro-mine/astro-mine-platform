"""The illumination backend contract — the swappable Sun-visibility surface (RM-P1-WORLDS-10).

worlds.md §11 recommends *precomputed horizon maps as the default, GPU ray casting for the fine
on-demand path, CPU ray casting as the portable fallback*, and asks (§11 open question) whether *a
learned surrogate could replace ray casting for very large swarm-scale queries, with tracked error*.
Those are **field-model plugins** selected in ``WorldSpec`` (worlds.md §3 "Field models —
alternative illumination … implementations registered against the same abstract interface"), not a
Core change: a new backend registers against the **existing** illumination interface and the public
:class:`~astro_mine.worlds.illumination.IlluminationModel` API + PSR-mask semantics do not change
(conventions.md §1.1 narrow waist; §1.3 plugins over patches).

:class:`SunVisibilityModel` is that abstract interface made explicit — the structural query surface
every backend (horizon map, ray-cast CPU/GPU, learned surrogate) satisfies, plus the attributes the
:class:`~astro_mine.worlds.provider.DemWorldProvider` reads through it (``horizon`` / ``n_azimuth``
the always-present Link line-of-sight product; ``transform``/``height``/``width``/``crs``/
``void_mask``/``abcorr``/``illumination_hash`` for the world/terrain surface). The horizon-map
:class:`IlluminationModel` satisfies it byte-for-byte unchanged; the fine and surrogate backends
subclass it and override only the Sun-visibility/mask queries, leaving the horizon LOS map intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from astro_mine.core.units import Epoch, EpochWindow, PlanetaryCRS
    from astro_mine.worlds.illumination import PsrEpochSemantics, PsrResult

__all__ = ["DEFAULT_BACKEND", "SunVisibilityModel"]

#: The Phase-0 default backend selector — the precomputed per-azimuth horizon map
#: (RM-P0-WORLDS-03). A ``WorldSpec`` that leaves ``illumination_backend`` unset resolves to this,
#: so existing world hashes stay stable (the backend is only recorded in the manifest when a
#: non-default one is selected).
DEFAULT_BACKEND = "horizon"


class SunVisibilityModel(Protocol):
    """The illumination query surface a Worlds field-model backend implements.

    Structural (``Protocol``) so a backend need only match the shape — the horizon-map
    :class:`~astro_mine.worlds.illumination.IlluminationModel` satisfies it as written, and the
    ray-cast / surrogate backends satisfy it by subclassing. The four methods are the Sun-visibility
    surface consumers query; the attributes are what the provider reads *through* the model — most
    importantly the per-azimuth ``horizon`` map, which stays the always-present LOS product
    [Link] queries even when the Sun-visibility path is served by a finer backend (worlds.md §6).
    """

    #: Per-cell, per-azimuth terrain-skyline elevation map — the Link LOS product (``(H, W, n)``).
    horizon: NDArray[np.float32]
    #: Number of azimuth bins in :attr:`horizon`.
    n_azimuth: int
    #: The raster affine transform (map ``(x, y)`` ⇄ ``(row, col)``).
    transform: Any
    #: Grid dimensions.
    height: int
    width: int
    #: The SPICE aberration correction the Sun geometry is resolved with.
    abcorr: str
    #: The content hash of this illumination model (folds in the active backend).
    illumination_hash: str
    #: The world CRS.
    crs: PlanetaryCRS
    #: DEM voids, where PSR-ness is not trustworthy.
    void_mask: NDArray[np.bool_]

    def sun_visible(self, x: float, y: float, epoch: Epoch) -> bool:
        """Is the Sun above the terrain horizon at projected ``(x, y)``, ``epoch``?"""
        ...

    def illumination_at(self, x: float, y: float, epoch: Epoch) -> tuple[bool, float]:
        """``(sun_visible, sun_elevation_deg)`` at a cell/epoch."""
        ...

    def illuminated_mask(self, epoch: Epoch) -> NDArray[np.bool_]:
        """Cells lit at ``epoch``."""
        ...

    def psr_mask(
        self, window: EpochWindow, step_s: float, *, semantics: PsrEpochSemantics
    ) -> PsrResult:
        """Cells never lit across ``window`` sampled at ``step_s`` — the PSR mask."""
        ...
