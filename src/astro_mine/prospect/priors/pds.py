"""The real PDS raster-ingest water-ice prior-recipe (RM-P1-PROSPECT-12).

Fits the Shackleton water-ice prior from **real public planetary rasters** — Diviner bolometric
temperature, LEND epithermal-neutron suppression, M³ surficial-hydration band depth, and the
LOLA + SPICE-derived permanently-shadowed-region (PSR) mask — replacing the Phase-0 *parametric
radial* cold-trap proxy (:func:`~astro_mine.prospect.priors.recipe.shackleton_water_ice_v1`) with
the measured cold-trap geometry (``prospect.md §2.4, §6``). The four layers are ingested and
reprojected onto the prior grid by :mod:`astro_mine.prospect.priors.ingest` (the ``[ingest]``
extra) and materialized as a content-addressed conditioning bundle; **this module reads that
bundle with numpy alone** and fits, so the local tier (``LUNAR-TR-004``) never pulls GDAL.

The recipe registers as ``shackleton_water_ice_pds_v1`` alongside — and additively to — the
parametric default, which stays the offline default (no consumer API change): a new name in the
same registry, resolved by :func:`~astro_mine.prospect.priors.load_prior`. Its
:class:`~astro_mine.prospect.priors.provenance.Provenance` carries the **real per-product content
hashes** (``DatasetCitation.source_hash``), so the fit reproduces from cited public inputs and a
Bench scenario can pin it.

Backlog: RM-P1-PROSPECT-12 — https://github.com/astro-mine/astro-mine-prospect/issues/11
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.units import MOON_BODY_FIXED, PlanetaryCRS
from astro_mine.prospect.field.metadata import FieldGrid, FieldMetadata
from astro_mine.prospect.priors.catalog import (
    DIVINER,
    DIVINER_COLD_TRAP_TEMP_K,
    LCROSS,
    LCROSS_WATER_WT_FRACTION,
    LCROSS_WATER_WT_SIGMA,
    LEND,
    LEND_BACKGROUND_WEH,
    LOLA,
    M3,
    SPECIES,
    UNIT,
)
from astro_mine.prospect.priors.ingest import (
    CONDITIONING_MEMBER,
    MANIFEST_MEMBER,
    bundle_content_hash,
    validate_manifest_crs,
)
from astro_mine.prospect.priors.provenance import DatasetCitation, Provenance
from astro_mine.prospect.priors.recipe import Prior, register_recipe

__all__ = [
    "CONDITIONING_DIR_ENV",
    "PDS_RECIPE_NAME",
    "build_pds_prior",
    "load_conditioning_bundle",
    "shackleton_water_ice_pds_v1",
]

#: The registered name of the real raster-ingest recipe (additive to the parametric default).
PDS_RECIPE_NAME = "shackleton_water_ice_pds_v1"
#: Env var pointing at the materialized conditioning-bundle directory the recipe fits from. The
#: multi-GB raster fetch + ingest is a one-time, cached, documented step (scripts/); the recipe
#: reads only the small materialized bundle, so setting this is all the offline fit needs.
CONDITIONING_DIR_ENV = "ASTRO_MINE_PROSPECT_CONDITIONING"

# --- fit knobs (recorded in provenance.params, so the fit reconstructs byte-for-byte) ---------
# Matches the parametric recipe's honest-uncertainty scheme: a LEND background WEH blended toward
# the LCROSS Cabeus anchor by an ice-favorability weight, with uncertainty scaled up with the mean.
_BACKGROUND_SIGMA = 0.004  # keep in lockstep with recipe._BACKGROUND_SIGMA
_TEMP_FLOOR_K = 40.0  # coldest plausible polar cold-trap floor → full cold-trap weight
_M3_BAND_DEPTH_REF = 0.15  # M³ band depth giving full surficial-water weight
_COVERAGE_SIGMA_INFLATION = 0.5  # up to +50% sigma where no conditioning layer covers a cell
# Role weights of the four ice-favorability indicators. Diviner temperature + PSR carry the
# fine spatial cold-trap signal (240 m / 120 m); LEND is coarse (3°, ~near-uniform over the 60 km
# prior box) and M³ is sparsely illuminated, so both are down-weighted regional/surficial nudges.
_ROLE_WEIGHTS: dict[str, float] = {
    "psr": 0.45,
    "measured_temperature": 0.35,
    "neutron_suppression": 0.10,
    "band_depth": 0.10,
}


class ConditioningBundle:
    """A materialized, content-addressed conditioning bundle, read with numpy alone.

    Verifies its arrays against the manifest ``content_hash`` on load (fail-closed) and exposes
    each layer by ``role`` plus the per-product ``source_hash`` the fit stamps into provenance.
    """

    def __init__(self, grid: FieldGrid, crs: PlanetaryCRS, manifest: dict[str, object]) -> None:
        self.grid = grid
        self.crs = crs
        self._manifest = manifest
        layers = manifest["layers"]
        assert isinstance(layers, dict)
        self._layers = layers
        self._by_role = {str(meta["role"]): name for name, meta in layers.items()}
        self._arrays: dict[str, NDArray[np.float64]] = {}

    def _load_arrays(self, npz_path: Path) -> None:
        with np.load(npz_path) as npz:
            arrays32 = {name: np.asarray(npz[name], dtype=np.float32) for name in npz.files}
        recorded = str(self._manifest["content_hash"])
        actual = bundle_content_hash(arrays32)
        if actual != recorded:
            raise ValueError(
                f"conditioning bundle content hash mismatch: manifest {recorded} != arrays "
                f"{actual} — the bundle is corrupt or tampered (fail-closed)"
            )
        self._arrays = {name: arr.astype(np.float64) for name, arr in arrays32.items()}

    @property
    def roles(self) -> tuple[str, ...]:
        """The conditioning roles present (``psr`` / ``measured_temperature`` / …), sorted."""
        return tuple(sorted(self._by_role))

    def layer(self, role: str) -> NDArray[np.float64] | None:
        """The ``(n_rows, n_cols)`` values for ``role`` (NaN off-coverage); ``None`` if absent."""
        name = self._by_role.get(role)
        return None if name is None else self._arrays[name]

    def source_hash(self, role: str) -> str | None:
        """The ingested-raster content hash for ``role`` (``None`` if the role is absent)."""
        name = self._by_role.get(role)
        return None if name is None else str(self._layers[name]["source_hash"])


def load_conditioning_bundle(bundle_dir: str | Path) -> ConditioningBundle:
    """Read a materialized conditioning bundle from ``bundle_dir`` (numpy only; fail-closed hash).

    Expects the two members :data:`~astro_mine.prospect.priors.ingest.MANIFEST_MEMBER` and
    :data:`~astro_mine.prospect.priors.ingest.CONDITIONING_MEMBER` written by
    :func:`~astro_mine.prospect.priors.ingest.materialize_conditioning_bundle`.
    """
    path = Path(bundle_dir)
    manifest = json.loads((path / MANIFEST_MEMBER).read_text(encoding="utf-8"))
    grid = FieldGrid.model_validate(manifest["grid"])
    # Schema-validate the manifest's CRS against Core's canonical units.schema.json, then guard it
    # (rule 6) before any numpy/rasterio machinery sees it — a missing or Earth-shaped CRS on a
    # lunar body fails here with a schema/guard error naming the field, not a downstream rasterio
    # error (RFC-0007; conventions.md §5).
    crs = validate_manifest_crs(manifest["crs"])
    bundle = ConditioningBundle(grid, crs, manifest)
    bundle._load_arrays(path / CONDITIONING_MEMBER)
    return bundle


def _favorability(bundle: ConditioningBundle) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coverage-weighted ice-favorability weight ``∈ [0, 1]`` and per-cell coverage fraction.

    Each present conditioning layer is mapped to an ice-favorability in ``[0, 1]`` (colder Diviner
    → higher; more PSR shadow → higher; more neutron suppression → higher; deeper M³ band → higher),
    NaN where the layer does not cover a cell. The layers are combined as a **coverage-weighted
    average** using :data:`_ROLE_WEIGHTS`, so missing layers drop out and never bias the weight, and
    ``coverage`` is the fraction of role-weight present at each cell (drives sigma inflation).
    """
    shape = (bundle.grid.n_rows, bundle.grid.n_cols)
    favs: list[NDArray[np.float64]] = []
    weights: list[float] = []

    def add(role: str, fav: NDArray[np.float64]) -> None:
        favs.append(fav)
        weights.append(_ROLE_WEIGHTS[role])

    psr = bundle.layer("psr")
    if psr is not None:
        add("psr", np.clip(psr, 0.0, 1.0))
    temp = bundle.layer("measured_temperature")
    if temp is not None:
        t_cold = (DIVINER_COLD_TRAP_TEMP_K - temp) / (DIVINER_COLD_TRAP_TEMP_K - _TEMP_FLOOR_K)
        add("measured_temperature", np.clip(t_cold, 0.0, 1.0))
    supp = bundle.layer("neutron_suppression")
    if supp is not None:
        add("neutron_suppression", np.clip(supp, 0.0, 1.0))
    band = bundle.layer("band_depth")
    if band is not None:
        add("band_depth", np.clip(band / _M3_BAND_DEPTH_REF, 0.0, 1.0))

    if not favs:
        raise ValueError("conditioning bundle carries no usable conditioning layers")

    fav_stack = np.stack(favs)  # (k, R, C)
    wt_stack = np.asarray(weights, dtype=np.float64)[:, None, None]
    present = np.isfinite(fav_stack)
    num = np.sum(np.where(present, wt_stack * np.nan_to_num(fav_stack), 0.0), axis=0)
    den = np.sum(np.where(present, wt_stack, 0.0), axis=0)
    weight = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)
    coverage = den / float(sum(weights))
    return np.clip(weight, 0.0, 1.0).reshape(shape), coverage.reshape(shape)


def build_pds_prior(grid: FieldGrid, bundle: ConditioningBundle) -> Prior:
    """Fit the real raster-ingest water-ice prior over ``grid`` from a conditioning ``bundle``.

    Blends a LEND background WEH toward the LCROSS Cabeus anchor by the coverage-weighted
    ice-favorability of the ingested Diviner/LEND/M³/PSR layers, scaling sigma with the mean
    (honest uncertainty) and inflating it where conditioning coverage is thin. ``grid`` must match
    the grid the bundle was materialized for. Returns a :class:`Prior` whose provenance carries the
    real per-product ``source_hash`` of each ingested raster.
    """
    if grid != bundle.grid:
        raise ValueError(
            "grid does not match the conditioning bundle's grid; the raster-ingest recipe fits on "
            "the grid its layers were reprojected onto (rebuild the bundle for a different grid)"
        )
    weight, coverage = _favorability(bundle)

    peak_weh = LCROSS_WATER_WT_FRACTION
    peak_sigma = LCROSS_WATER_WT_SIGMA
    mean = LEND_BACKGROUND_WEH + (peak_weh - LEND_BACKGROUND_WEH) * weight
    sigma = _BACKGROUND_SIGMA + (peak_sigma - _BACKGROUND_SIGMA) * weight
    sigma = sigma * (1.0 + _COVERAGE_SIGMA_INFLATION * (1.0 - coverage))
    variance = sigma * sigma

    metadata = FieldMetadata(
        species=SPECIES, unit=UNIT, frame=MOON_BODY_FIXED, crs=bundle.crs, grid=grid
    )
    citations = _ingested_citations(bundle)
    provenance = Provenance(
        recipe=PDS_RECIPE_NAME,
        recipe_version="1.0.0",
        citations=citations,
        derivation=(
            "Real PDS raster-ingest WEH prior: a LEND background WEH blended to the LCROSS Cabeus "
            "water anchor by the coverage-weighted ice-favorability of the ingested Diviner "
            "bolometric temperature, LEND epithermal-neutron suppression, M³ surficial-hydration "
            "band depth, and LOLA + SPICE-derived PSR mask, each reprojected onto the Shackleton "
            "prior grid. sigma scales with the mean (LCROSS spread) and inflates where coverage "
            "is thin — honest uncertainty (prospect.md §9). See RECIPE.md."
        ),
        params={
            "background_weh": LEND_BACKGROUND_WEH,
            "peak_weh": peak_weh,
            "background_sigma": _BACKGROUND_SIGMA,
            "peak_sigma": peak_sigma,
            "cold_trap_threshold_k": DIVINER_COLD_TRAP_TEMP_K,
            "temp_floor_k": _TEMP_FLOOR_K,
            "m3_band_depth_ref": _M3_BAND_DEPTH_REF,
            "coverage_sigma_inflation": _COVERAGE_SIGMA_INFLATION,
            "weight_psr": _ROLE_WEIGHTS["psr"],
            "weight_temperature": _ROLE_WEIGHTS["measured_temperature"],
            "weight_neutron_suppression": _ROLE_WEIGHTS["neutron_suppression"],
            "weight_band_depth": _ROLE_WEIGHTS["band_depth"],
        },
    )
    return Prior(metadata, mean, variance, provenance)


# Which cited dataset each conditioning role fills the source_hash of.
_ROLE_CITATION = {
    "psr": LOLA,  # PSR mask derived from the LOLA DEM + SPICE illumination geometry
    "measured_temperature": DIVINER,
    "neutron_suppression": LEND,
    "band_depth": M3,
}


def _ingested_citations(bundle: ConditioningBundle) -> tuple[DatasetCitation, ...]:
    """The cited datasets with the real ``source_hash`` of each ingested raster filled in.

    LCROSS stays a magnitude anchor (no raster, ``source_hash=None``); the other four carry the
    content hash of the raster actually ingested, in the catalog's derivation order.
    """
    filled: dict[str, DatasetCitation] = {}
    for role, citation in _ROLE_CITATION.items():
        source_hash = bundle.source_hash(role)
        if source_hash is not None:
            filled[citation.short_name] = citation.model_copy(update={"source_hash": source_hash})
    ordered = (LOLA, DIVINER, LEND, M3, LCROSS)
    return tuple(filled.get(c.short_name, c) for c in ordered)


def shackleton_water_ice_pds_v1(grid: FieldGrid) -> Prior:
    """The real raster-ingest water-ice prior (prospect.md §2.4, §6, §12) — registered recipe.

    Resolves the materialized conditioning bundle from the :data:`CONDITIONING_DIR_ENV` env var and
    fits :func:`build_pds_prior` over ``grid``. Raises ``FileNotFoundError`` with the build command
    if the bundle is absent — the multi-GB raster fetch + ingest is a one-time, documented step
    (``scripts/fetch_pds_conditioning.py`` → ``scripts/build_pds_prior.py``); the parametric
    ``shackleton_water_ice_v1`` remains the offline default that always works.
    """
    bundle_dir = os.environ.get(CONDITIONING_DIR_ENV)
    if not bundle_dir or not Path(bundle_dir, MANIFEST_MEMBER).exists():
        raise FileNotFoundError(
            f"no conditioning bundle found (set {CONDITIONING_DIR_ENV} to a directory built by "
            "`python scripts/build_pds_prior.py`). The parametric 'shackleton_water_ice_v1' is the "
            "offline default and needs no raster ingest."
        )
    return build_pds_prior(grid, load_conditioning_bundle(bundle_dir))


register_recipe(PDS_RECIPE_NAME, shackleton_water_ice_pds_v1)
