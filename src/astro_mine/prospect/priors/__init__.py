"""Water-ice / hydrogen priors from public lunar datasets (RM-P0-PROSPECT-03).

The dataset-derived, **provenance-tracked** prior that seeds the belief field before any observation
(prospect.md §2.4, §12). :func:`load_prior` resolves a named recipe and returns a :class:`Prior` —
a per-cell Gaussian field over the Shackleton-de Gerlache CRS/grid (aligned to Worlds,
RM-P0-WORLDS-01) carrying a :class:`Provenance` that cites its public sources (LOLA / Diviner /
LEND / M³ / LCROSS). ``prior.as_field()`` realizes it as a Core ``ResourceField``.

The Phase-0 default ``shackleton_water_ice_v1`` is an offline, deterministic, **cited parametric**
fit (no raster ingest — see ``recipe.py`` and ``RECIPE.md``). The real PDS raster-ingest recipe
``shackleton_water_ice_pds_v1`` (RM-P1-PROSPECT-12, ``pds.py``) plugs into the same registry with
no consumer change: it fits from real Diviner/LEND/M³ + LOLA-PSR rasters (materialized as a
content-addressed conditioning bundle), while the parametric default stays the offline default.

Backlog: RM-P0-PROSPECT-03 — https://github.com/astro-mine/astro-mine-prospect/issues/3
"""

from __future__ import annotations

from astro_mine.prospect.field.metadata import FieldGrid
from astro_mine.prospect.priors.catalog import (
    CITATIONS,
    SHACKLETON_CRS,
    SHACKLETON_PRIOR_GRID,
    SPECIES,
    UNIT,
)
from astro_mine.prospect.priors.pds import (
    PDS_RECIPE_NAME,
    build_pds_prior,
    load_conditioning_bundle,
    shackleton_water_ice_pds_v1,
)
from astro_mine.prospect.priors.provenance import DatasetCitation, Provenance
from astro_mine.prospect.priors.recipe import (
    Prior,
    PriorRecipe,
    get_recipe,
    list_recipes,
    register_recipe,
    shackleton_water_ice_v1,
)

__all__ = [
    "CITATIONS",
    "PDS_RECIPE_NAME",
    "SHACKLETON_CRS",
    "SHACKLETON_PRIOR_GRID",
    "SPECIES",
    "UNIT",
    "DatasetCitation",
    "Prior",
    "PriorRecipe",
    "Provenance",
    "build_pds_prior",
    "list_priors",
    "load_conditioning_bundle",
    "load_prior",
    "register_recipe",
    "shackleton_water_ice_pds_v1",
    "shackleton_water_ice_v1",
]


def load_prior(name: str = "shackleton_water_ice_v1", *, grid: FieldGrid | None = None) -> Prior:
    """Load a named, provenance-tracked prior, fitted over ``grid`` (default: the Shackleton grid).

    Resolves the recipe registered under ``name`` and runs it; an unknown ``name`` raises
    ``ValueError``. Pass ``grid`` to fit the prior over a specific Worlds grid (it must carry the
    same CRS the recipe targets); omit it for the canonical :data:`SHACKLETON_PRIOR_GRID`.
    """
    recipe = get_recipe(name)
    return recipe(SHACKLETON_PRIOR_GRID if grid is None else grid)


def list_priors() -> tuple[str, ...]:
    """The names of all registered priors (sorted)."""
    return list_recipes()
