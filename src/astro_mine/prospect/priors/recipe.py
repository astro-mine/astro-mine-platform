"""The Prior bundle, the recipe registry, and the default Shackleton water-ice recipe.

A :class:`Prior` is a spatially-varying Gaussian prior over a world's grid (per-cell ``mean`` +
``variance``) carrying the :class:`~astro_mine.prospect.priors.provenance.Provenance` that cites its
public sources. It seeds the belief field (and is the sampling base for a sealed ground truth,
RM-P0-PROSPECT-04) and realizes as a :class:`~astro_mine.prospect.backends.grid.GridField` — a Core
``ResourceField`` — via :meth:`Prior.as_field`, so a consumer queries it like any other field.

A :data:`PriorRecipe` is the deterministic builder ``grid -> Prior``; recipes register by name so
:func:`~astro_mine.prospect.priors.load_prior` can resolve one. This is the seam the real
raster-ingest recipe (RM-P1-PROSPECT-12, #11) plugs into: a new recipe registers alongside the
parametric default with **no change** to ``load_prior`` or any consumer.

:func:`shackleton_water_ice_v1` is the Phase-0 default: an **offline, deterministic, cited
parametric** prior — a LEND background WEH blended to the LCROSS Cabeus water anchor by a Diviner
cold-trap proxy over LOLA polar geometry (see ``RECIPE.md``). It ingests no rasters; the offline
local tier (``LUNAR-TR-004``) runs without network or account.

Backlog: RM-P0-PROSPECT-03 — astro-mine-prospect#3
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

# MOON_BODY_FIXED is the field's query frame (its centre matches the Shackleton CRS body).
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.prospect.backends.grid import GridField
from astro_mine.prospect.field.metadata import FieldGrid, FieldMetadata
from astro_mine.prospect.priors.catalog import (
    CITATIONS,
    LCROSS_WATER_WT_FRACTION,
    LCROSS_WATER_WT_SIGMA,
    LEND_BACKGROUND_WEH,
    SHACKLETON_CRS,
    SPECIES,
    UNIT,
)
from astro_mine.prospect.priors.provenance import Provenance

__all__ = [
    "Prior",
    "PriorRecipe",
    "get_recipe",
    "list_recipes",
    "register_recipe",
    "shackleton_water_ice_v1",
]

#: A deterministic prior builder: a grid in → a fitted :class:`Prior` out.
PriorRecipe = Callable[[FieldGrid], "Prior"]

# Recipe-specific knobs of the default parametric fit (recorded in provenance for reproducibility).
_BACKGROUND_SIGMA = 0.004
_COLDNESS_LENGTH_SCALE_M = 12_000.0


class Prior:
    """A dataset-derived prior over a world's grid — per-cell mean/variance + cited provenance.

    Construct via a :data:`PriorRecipe` (e.g. :func:`shackleton_water_ice_v1`), not directly. The
    arrays are ``(n_rows, n_cols)`` over ``metadata.grid`` and held read-only; :attr:`content_hash`
    content-addresses the prior (provenance + arrays + grid), so a Bench scenario can pin it.
    """

    def __init__(
        self,
        metadata: FieldMetadata,
        mean: NDArray[np.float64],
        variance: NDArray[np.float64],
        provenance: Provenance,
    ) -> None:
        grid = metadata.grid
        if grid is None:
            raise ValueError("Prior requires metadata.grid (a FieldGrid spatial domain)")
        shape = (grid.n_rows, grid.n_cols)
        if mean.shape != shape or variance.shape != shape:
            raise ValueError(
                f"mean/variance arrays must have shape {shape} (n_rows, n_cols); "
                f"got mean {mean.shape}, variance {variance.shape}"
            )
        if bool(np.any(variance < 0.0)):
            raise ValueError("variance grid must be non-negative everywhere")
        self._metadata = metadata
        self._mean = np.ascontiguousarray(mean, dtype=np.float64)
        self._variance = np.ascontiguousarray(variance, dtype=np.float64)
        self._mean.flags.writeable = False
        self._variance.flags.writeable = False
        self._provenance = provenance

    @property
    def metadata(self) -> FieldMetadata:
        """The species/unit and CRS/grid binding the prior is defined over."""
        return self._metadata

    @property
    def mean(self) -> NDArray[np.float64]:
        """The per-cell prior mean ``(n_rows, n_cols)`` (read-only)."""
        return self._mean

    @property
    def variance(self) -> NDArray[np.float64]:
        """The per-cell prior variance ``(n_rows, n_cols)`` (read-only)."""
        return self._variance

    @property
    def provenance(self) -> Provenance:
        """The cited-source provenance record."""
        return self._provenance

    def as_field(self) -> GridField:
        """Realize the prior as a queryable :class:`GridField` (a Core ``ResourceField``)."""
        return GridField(self._metadata, self._mean, self._variance)

    @property
    def content_hash(self) -> str:
        """A stable content address over the provenance, the per-cell arrays, and the grid."""
        digest = hashlib.sha256()
        digest.update(self._provenance.content_hash.encode("utf-8"))
        digest.update(self._metadata.model_dump_json().encode("utf-8"))
        digest.update(self._mean.tobytes())
        digest.update(self._variance.tobytes())
        return digest.hexdigest()


_RECIPES: dict[str, PriorRecipe] = {}

#: recipe key -> the name the prior publishes under. See :func:`default_artifact_name`.
_ARTIFACT_NAMES: dict[str, str] = {}

_VERSION_SUFFIX = re.compile(r"-v\d+$")


def default_artifact_name(recipe_name: str) -> str:
    """The conforming published artifact name a recipe key maps to (``conventions.md`` §13).

    **A recipe key and an artifact name are two different things**, and conflating them is what let
    ``shackleton_water_ice_pds_v1`` reach the registry. A key is a Python-side identifier: it names
    a callable, it is what ``load_prior`` and ``get_recipe`` take, and it is snake_case because the
    function it names is. An artifact name is what the published bytes are addressed by, and §13
    requires bare kebab-case with no version in the name.

    Publishing defaulted to the key, so the key silently *became* the artifact name and dragged its
    shape into the registry with it. Deriving instead of defaulting keeps the key free to look like
    Python while making the published name conformant by construction: underscores become hyphens
    and a trailing ``-v<n>`` is dropped, because the version belongs in the tag.

        shackleton_water_ice_v1      -> shackleton-water-ice
        shackleton_water_ice_pds_v1  -> shackleton-water-ice-pds

    Raises :class:`ValueError` if the derivation is not conformant, so a key that cannot produce a
    valid name fails at registration rather than at publish — or, worse, silently at neither.
    """
    from astro_mine.hub.registry import is_valid_artifact_name

    candidate = _VERSION_SUFFIX.sub("", recipe_name.replace("_", "-"))
    if not is_valid_artifact_name(candidate):
        raise ValueError(
            f"prior recipe {recipe_name!r} does not derive a conforming artifact name "
            f"(got {candidate!r}); pass artifact_name= explicitly (conventions.md §13)"
        )
    return candidate


def register_recipe(name: str, recipe: PriorRecipe, *, artifact_name: str | None = None) -> None:
    """Register a prior recipe under ``name`` (fails loudly on a duplicate name).

    ``artifact_name`` is the name the prior publishes under; it defaults to
    :func:`default_artifact_name` of the key, and is validated either way. It is a keyword with a
    derived default rather than a required argument so that existing registrations — including
    community ones through :func:`~astro_mine.prospect.publish.publish_recipe` — keep working while
    still being unable to mint a non-conforming name.
    """
    if name in _RECIPES:
        raise ValueError(f"prior recipe {name!r} is already registered")
    if artifact_name is None:
        resolved = default_artifact_name(name)
    else:
        from astro_mine.hub.registry import validate_artifact_name

        resolved = validate_artifact_name(artifact_name)
    _RECIPES[name] = recipe
    _ARTIFACT_NAMES[name] = resolved


def artifact_name_for(recipe_name: str) -> str:
    """The published artifact name for ``recipe_name``, registered or derived.

    Unregistered keys are derived rather than rejected: a caller may hand
    :func:`~astro_mine.prospect.publish.publish_prior` a :class:`Prior` whose
    ``provenance.recipe`` was never registered in this process, and that is a legitimate ad-hoc
    publish, not an error. It still gets a conforming name.
    """
    known = _ARTIFACT_NAMES.get(recipe_name)
    return known if known is not None else default_artifact_name(recipe_name)


def list_artifact_names() -> tuple[str, ...]:
    """The published artifact names of all registered prior recipes, sorted.

    This — not :func:`list_recipes` — is the set that has to satisfy §13, and the distinction is the
    whole point of splitting the two concepts.
    """
    return tuple(sorted(_ARTIFACT_NAMES.values()))


def get_recipe(name: str) -> PriorRecipe:
    """Resolve a registered prior recipe by ``name`` (``ValueError`` if unknown)."""
    try:
        return _RECIPES[name]
    except KeyError:
        known = ", ".join(sorted(_RECIPES)) or "(none)"
        raise ValueError(f"unknown prior {name!r}; registered priors are: {known}") from None


def list_recipes() -> tuple[str, ...]:
    """The names of all registered prior recipes, sorted."""
    return tuple(sorted(_RECIPES))


def _radial_coldness(grid: FieldGrid) -> NDArray[np.float64]:
    """A parametric cold-trap proxy: Gaussian-decaying from the pole (grid origin) outward.

    A stand-in for the real Diviner-temperature / PSR cold-trap field (which the raster-ingest
    recipe, #11, supplies): coldness is 1 at the pole and falls off with a characteristic length
    scale, so the prior concentrates ice in the deep polar cold traps. Cells are evaluated at their
    centres, in the CRS's projected metres.
    """
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    xs = grid.min_x_m + (np.arange(grid.n_cols) + 0.5) * dx
    ys = grid.min_y_m + (np.arange(grid.n_rows) + 0.5) * dy
    gx, gy = np.meshgrid(xs, ys)  # (n_rows, n_cols)
    r2 = gx * gx + gy * gy
    ls = _COLDNESS_LENGTH_SCALE_M
    cold: NDArray[np.float64] = np.exp(-r2 / (2.0 * ls * ls))
    return cold


def shackleton_water_ice_v1(
    grid: FieldGrid,
    *,
    coldness: NDArray[np.float64] | None = None,
) -> Prior:
    """The Phase-0 default water-ice prior — cited, parametric, offline (prospect.md §2.4, §12).

    Blends a LEND background WEH up to the LCROSS Cabeus water anchor by a cold-trap weight, and
    scales the prior uncertainty with it (most ice ⇒ most uncertain, per the LCROSS spread — honest
    uncertainty, prospect.md §9). ``coldness`` is the load-bearing **conditioning hook**: pass a
    grid-shaped weight in ``[0, 1]`` (e.g. a real Worlds PSR/temperature layer) to drive the prior;
    omit it for the parametric radial default. The result aligns to ``grid`` and the Shackleton CRS.
    """
    shape = (grid.n_rows, grid.n_cols)
    if coldness is None:
        cold = _radial_coldness(grid)
    else:
        cold = np.ascontiguousarray(coldness, dtype=np.float64)
        if cold.shape != shape:
            raise ValueError(f"coldness must have grid shape {shape}; got {cold.shape}")
        if bool(np.any(cold < 0.0)) or bool(np.any(cold > 1.0)):
            raise ValueError("coldness weights must lie in [0, 1]")

    mean = LEND_BACKGROUND_WEH + (LCROSS_WATER_WT_FRACTION - LEND_BACKGROUND_WEH) * cold
    sigma = _BACKGROUND_SIGMA + (LCROSS_WATER_WT_SIGMA - _BACKGROUND_SIGMA) * cold
    variance = sigma * sigma

    metadata = FieldMetadata(
        species=SPECIES, unit=UNIT, frame=MOON_BODY_FIXED, crs=SHACKLETON_CRS, grid=grid
    )
    provenance = Provenance(
        recipe="shackleton_water_ice_v1",
        recipe_version="1.0.0",
        citations=CITATIONS,
        derivation=(
            "Parametric cold-trap-weighted WEH prior: a LEND background blended to the LCROSS "
            "Cabeus water anchor by a Diviner cold-trap proxy over LOLA polar geometry. Offline "
            "Phase-0 fit — published-characterization anchors, no raster ingest (see RECIPE.md)."
        ),
        params={
            "background_weh": LEND_BACKGROUND_WEH,
            "peak_weh": LCROSS_WATER_WT_FRACTION,
            "background_sigma": _BACKGROUND_SIGMA,
            "peak_sigma": LCROSS_WATER_WT_SIGMA,
            "coldness_length_scale_m": _COLDNESS_LENGTH_SCALE_M,
        },
    )
    return Prior(metadata, mean, variance, provenance)


register_recipe("shackleton_water_ice_v1", shackleton_water_ice_v1)
