"""The cited public lunar datasets + the canonical Shackleton CRS/grid (prospect.md §2.4, §6).

The Phase-0 water-ice prior is anchored to the **published characterizations** of five public
lunar datasets — cited here as :class:`~astro_mine.prospect.priors.provenance.DatasetCitation`
constants — and is defined over the lunar south-polar **Shackleton-de Gerlache** CRS/grid, kept
consistent with the Worlds reprojected grid (RM-P0-WORLDS-01) so a prior layers cleanly on the
world (prospect.md §5 "layered on a world, never freestanding"). The named numeric anchors the
recipe blends (LCROSS water fraction, LEND background WEH, the Diviner cold-trap threshold) are
constants too, so the derivation is transparent and auditable.

Backlog: RM-P0-PROSPECT-03 — https://github.com/astro-mine/astro-mine-prospect/issues/3
"""

from __future__ import annotations

from astro_mine.core.units import MOON, PlanetaryCRS
from astro_mine.prospect.field.metadata import FieldGrid
from astro_mine.prospect.priors.provenance import DatasetCitation

__all__ = [
    "CITATIONS",
    "DIVINER",
    "DIVINER_COLD_TRAP_TEMP_K",
    "LCROSS",
    "LCROSS_WATER_WT_FRACTION",
    "LCROSS_WATER_WT_SIGMA",
    "LEND",
    "LEND_BACKGROUND_WEH",
    "LOLA",
    "M3",
    "SHACKLETON_CRS",
    "SHACKLETON_PRIOR_GRID",
    "SPECIES",
    "UNIT",
]

# --- what the field models -------------------------------------------------------------------

#: The modeled resource species and its SI unit token (validated against the Core vocabulary).
SPECIES = "water_equivalent_hydrogen"
UNIT = "mass_fraction"

# --- the canonical target: the Shackleton-de Gerlache CRS/grid (aligned to Worlds) -----------

#: Lunar south-polar stereographic CRS, modelling the Moon as a sphere (PROJ ``+R``); the same
#: explicit CRS the Worlds Shackleton DEM is reprojected to (RM-P0-WORLDS-01; conventions.md §5).
SHACKLETON_CRS = PlanetaryCRS(
    body=MOON,
    body_fixed_frame="MOON_ME",
    reference_radius_m=1_737_400.0,
    projection="+proj=stere +lat_0=-90 +R=1737400",
)

#: The default prior grid: a 60x60 km box about the south pole at 250 m/px, in the CRS's projected
#: metres. The pole sits at the grid origin (x=y=0), i.e. the grid centre. Documented to match the
#: Worlds reprojected grid extent/resolution for the target region.
SHACKLETON_PRIOR_GRID = FieldGrid(
    min_x_m=-30_000.0,
    min_y_m=-30_000.0,
    max_x_m=30_000.0,
    max_y_m=30_000.0,
    n_rows=240,
    n_cols=240,
)

# --- published characterization anchors (mass fraction unless noted) --------------------------

#: LCROSS Cabeus ejecta: 5.6 ± 2.9 wt% water (Colaprete et al. 2010) — high-ice magnitude anchor.
LCROSS_WATER_WT_FRACTION = 0.056
LCROSS_WATER_WT_SIGMA = 0.029
#: LEND broad polar epithermal-neutron background → ~0.5 wt% bulk WEH away from cold traps.
LEND_BACKGROUND_WEH = 0.005
#: Diviner surface-water-ice stability threshold (~110 K cold traps; Paige et al. 2010), Kelvin.
DIVINER_COLD_TRAP_TEMP_K = 110.0

# --- the cited datasets ----------------------------------------------------------------------

LOLA = DatasetCitation(
    short_name="LOLA",
    instrument="Lunar Orbiter Laser Altimeter",
    mission="Lunar Reconnaissance Orbiter",
    product="PDS LRO-L-LOLA-4-GDR-V1.0 (south-polar DEM)",
    reference="Smith et al. (2010), Space Sci. Rev. 150, 209-241",
    role="south-polar topography → PSR / illumination geometry (cold-trap siting)",
)
DIVINER = DatasetCitation(
    short_name="Diviner",
    instrument="Diviner Lunar Radiometer Experiment",
    mission="Lunar Reconnaissance Orbiter",
    product="PDS LRO-L-DLRE-4-RDR-V1.0 (bolometric temperatures)",
    reference="Paige et al. (2010), Science 330, 479-482",
    role="cold-trap temperatures (<~110 K) gating surface water-ice stability",
)
LEND = DatasetCitation(
    short_name="LEND",
    instrument="Lunar Exploration Neutron Detector",
    mission="Lunar Reconnaissance Orbiter",
    product="PDS LRO-L-LEND-4-RDR-V1.0 (epithermal neutron count rates)",
    reference=(
        "Mitrofanov et al. (2010), Science 330, 483-486; Sanin et al. (2017), Icarus 283, 20-30"
    ),
    role="epithermal-neutron suppression → bulk water-equivalent hydrogen (WEH) magnitude",
)
M3 = DatasetCitation(
    short_name="M3",
    instrument="Moon Mineralogy Mapper",
    mission="Chandrayaan-1",
    product="PDS CH1-ORB-L-M3-4-L2-REFLECTANCE-V1.0",
    reference="Pieters et al. (2009), Science 326, 568-572; Li et al. (2018), PNAS 115, 8907-8912",
    role="surficial OH / H2O absorption → near-surface ice presence at the poles",
)
LCROSS = DatasetCitation(
    short_name="LCROSS",
    instrument="Lunar Crater Observation and Sensing Satellite",
    mission="LCROSS",
    product="Cabeus impact-plume spectroscopy (NSVF data release)",
    reference="Colaprete et al. (2010), Science 330, 463-468",
    role="ground-truth magnitude anchor: 5.6 ± 2.9 wt% water in Cabeus ejecta",
)

#: The full cited basis of the default prior, in derivation order.
CITATIONS: tuple[DatasetCitation, ...] = (LOLA, DIVINER, LEND, M3, LCROSS)
