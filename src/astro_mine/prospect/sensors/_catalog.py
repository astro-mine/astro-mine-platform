# SPDX-License-Identifier: Apache-2.0
"""The anchor scenario's instrument set — the built-in sensor likelihoods (scenario §6).

The lunar polar water-ice scout carries a **neutron spectrometer, NIR, GPR and a drill**
(scenario §6, "Scout / prospector rover"), and prospect.md §3 names exactly those as the likelihoods
the ``sensors/`` extension point must supply. Each is a :class:`~astro_mine.prospect.sensors.\\
_likelihood.SensorLikelihood` declaring its own Core ``SensorKind``/``CapabilityTag``, its **spatial
footprint**, and its **depth response** — so the four instruments read the *same* field and honestly
disagree, instead of collapsing to one scalar ``noise_sigma``:

========================  ==============  ==============================  ====================
Likelihood                Footprint       Depth window                    Gain
========================  ==============  ==============================  ====================
``neutron_spectrometer``  2 m (broad)     0 - 1.0 m, attenuating (0.5 m)  ~1 (definitional)
``nir_reflectance``       0.5 m           0 - 5 mm, **surface only**      <<1 (desiccated lag)
``gpr``                   1 m             0.05 - 3.0 m (subsurface)       >1 (buried ice)
``drill_assay``           0.05 m (point)  0.3 - 1.0 m (the sampled core)  >1
``point_gaussian``        0 (point)       0 - 1.0 m, flat                 1 (exactly)
========================  ==============  ==============================  ====================

The numbers are **reduced-order and illustrative** — a scenario or a Fleet asset overrides them by
registering its own likelihood — but their *ordering* is physical and load-bearing. An NIR reading
of a desiccated surface is weak evidence about the buried column (a gain far below 1 means a
precision ``gain**2 / sigma**2`` far below ``1 / sigma**2``), while a drill assay of the same cell
is strong evidence; and a neutron spectrometer informs a whole neighbourhood at once through its
footprint.

**On footprint and grid scale.** These are *rover-borne* instruments (scenario §6), so their
footprints are metres — while the anchor prospecting grid is 250 m per cell. A metre-scale footprint
is therefore **sub-cell** at prospecting scale and correctly degenerates to a point measurement
there: at that resolution it is the **depth response** that separates the instruments, and it does
so at every scale. The footprint becomes load-bearing on a fine, local field (a dense survey of a
single cold trap), which is exactly when it should. Fudging the footprints upward to make them
visible on a coarse grid would be modeling the grid, not the instrument.

``point_gaussian`` is the :data:`~astro_mine.prospect.sensors._likelihood.DEFAULT_LIKELIHOOD_NAME`
default — zero footprint, full reference column, flat sensitivity, hence ``gain == 1`` **exactly** —
so an untagged observation conditions precisely as it did before this seam existed.
"""

from __future__ import annotations

from astro_mine.core.sadf.enums import CapabilityTag, SensorKind
from astro_mine.prospect.sensors._likelihood import (
    DEFAULT_LIKELIHOOD_NAME,
    REFERENCE_COLUMN_DEPTH_M,
    DepthResponse,
    SensorLikelihood,
)

__all__ = ["BUILTIN_LIKELIHOODS"]


#: The zero-footprint, full-column, flat-sensitivity Gaussian point measurement — the Phase-0 model,
#: kept as the registered default so the un-tagged conditioning path is unchanged (``gain == 1``).
POINT_GAUSSIAN = SensorLikelihood(
    name=DEFAULT_LIKELIHOOD_NAME,
    kind=SensorKind.NEUTRON_SPECTROMETER,
    capability=CapabilityTag.PROSPECTING_NEUTRON,
    footprint_sigma_m=0.0,
    depth=DepthResponse(top_m=0.0, bottom_m=REFERENCE_COLUMN_DEPTH_M, attenuation_length_m=None),
    noise_sigma=0.05,
    description=(
        "Zero-footprint Gaussian point measurement of the reference column (gain 1) — the "
        "backward-compatible default likelihood for an observation carrying no instrument tag."
    ),
)

#: Epithermal-neutron spectrometer: the **definitional** WEH instrument — it integrates the top
#: metre (surface-weighted, hence the 0.5 m attenuation length) over a broad footprint.
NEUTRON_SPECTROMETER = SensorLikelihood(
    name="neutron_spectrometer",
    kind=SensorKind.NEUTRON_SPECTROMETER,
    capability=CapabilityTag.PROSPECTING_NEUTRON,
    footprint_sigma_m=2.0,
    depth=DepthResponse(top_m=0.0, bottom_m=REFERENCE_COLUMN_DEPTH_M, attenuation_length_m=0.5),
    noise_sigma=0.02,
    description=(
        "Epithermal-neutron spectrometer — broad footprint, surface-weighted integration of the "
        "top metre of regolith. The instrument the water-equivalent-hydrogen field is defined by."
    ),
)

#: NIR reflectance: a **surface-only** instrument. It sees the top few millimetres — the desiccated
#: lag deposit — so it is precise about the surface and weak evidence about the buried column.
NIR_REFLECTANCE = SensorLikelihood(
    name="nir_reflectance",
    kind=SensorKind.NIR_SPECTROMETER,
    capability=CapabilityTag.PROSPECTING_NIR,
    footprint_sigma_m=0.5,
    depth=DepthResponse(top_m=0.0, bottom_m=0.005, attenuation_length_m=None),
    noise_sigma=0.01,
    description=(
        "Near-infrared reflectance — a surface-only measurement (top ~5 mm). Reads the desiccated "
        "lag deposit, so its depth gain is far below 1 and it informs the buried column weakly."
    ),
)

#: Ground-penetrating radar: a narrow footprint sounding **into** the subsurface, past the lag
#: deposit — the complement of NIR.
GPR = SensorLikelihood(
    name="gpr",
    kind=SensorKind.GPR,
    capability=CapabilityTag.PROSPECTING_GPR,
    footprint_sigma_m=1.0,
    depth=DepthResponse(top_m=0.05, bottom_m=3.0, attenuation_length_m=1.2),
    noise_sigma=0.03,
    description=(
        "Ground-penetrating radar — a narrow footprint sounding 0.05-3 m with two-way attenuation. "
        "Sees past the desiccated lag into buried ice, so its depth gain exceeds 1."
    ),
)

#: Drill assay: an essentially **point** sample of a specific cored interval — the most direct,
#: lowest-noise, and most expensive measurement in the set.
DRILL_ASSAY = SensorLikelihood(
    name="drill_assay",
    kind=SensorKind.DRILL_ASSAY,
    capability=CapabilityTag.PROSPECTING_DRILL_ASSAY,
    footprint_sigma_m=0.05,
    depth=DepthResponse(top_m=0.3, bottom_m=1.0, attenuation_length_m=None),
    noise_sigma=0.002,
    description=(
        "Drill-core assay — a point sample of the 0.3-1.0 m cored interval, uniformly weighted. "
        "The lowest-noise, highest-gain instrument in the set (and the most expensive to take)."
    ),
)

#: Every likelihood registered at import (:mod:`astro_mine.prospect.sensors`).
BUILTIN_LIKELIHOODS: tuple[SensorLikelihood, ...] = (
    POINT_GAUSSIAN,
    NEUTRON_SPECTROMETER,
    NIR_REFLECTANCE,
    GPR,
    DRILL_ASSAY,
)
