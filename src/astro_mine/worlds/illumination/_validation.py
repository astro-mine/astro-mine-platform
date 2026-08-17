# SPDX-License-Identifier: Apache-2.0
"""Illumination/PSR validation against published lunar references (worlds.md §10).

worlds.md §10 and the Phase-0 Worlds exit criteria both require that "illumination/PSR results
[are] regression-tested against **published lunar illumination and PSR datasets** (e.g.
LOLA-derived PSR catalogs) **with explicit error budgets**" — and worlds.md §9 explains why it is
not optional: "the PSR mask, illumination, and slope/bearing fields it produces feed
safety-relevant decisions … therefore data correctness is a first-class safety concern".

Neither the architecture nor the roadmap fixes a *number*; both mandate that one be stated. This
module is where a reference and its budget become machine-checkable artifacts rather than prose:

- :class:`PsrReference` — a **committed** published reference (value + citation + the exact harness
  configuration it is comparable under) loaded from JSON. Living in the repo (``validation/``) is
  the point: the reference and the budget are reviewable, diffable, and cited.
- :func:`psr_statistics` — reduce a :class:`~astro_mine.worlds.illumination.PsrResult` to the
  scalars a published catalog quotes: the permanently-shadowed **area fraction** and **area (km²)**.
- :func:`validate_psr` — compare, and report pass/fail against the stated budget.

The comparison lives here (in the package), not in ``scripts/``, so the same code path runs from the
CLI harness *and* from the marker-gated regression test — a validation nobody can re-run is not a
validation (conventions.md §11 "golden tests & determinism gates").

Backlog: RM-P0-WORLDS-03; ``LUNAR-FR-001``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from astro_mine.worlds.illumination import PsrResult

__all__ = [
    "VALIDATION_SCHEMA",
    "PsrReference",
    "PsrValidation",
    "psr_statistics",
    "validate_psr",
]

VALIDATION_SCHEMA = "astro-mine-worlds/illumination-validation/v0.1"

_M2_PER_KM2 = 1.0e6


@dataclass(frozen=True)
class PsrReference:
    """A published PSR reference for a region: the value, its budget, and how to reproduce it.

    ``psr_area_fraction`` is the fraction of the modelled grid that is permanently shadowed, and
    ``tolerance_area_fraction`` the **absolute** error budget on it. ``harness`` records the
    configuration the reference is comparable under (resolution, azimuth bins, horizon radius,
    aberration correction, epoch window, PSR semantics) — a PSR fraction is meaningless without it,
    because "permanent" is defined by the sampled window (worlds.md §11's open question, made
    explicit by :class:`~astro_mine.worlds.illumination.PsrEpochSemantics`).
    """

    region: str
    source: str
    psr_area_fraction: float
    tolerance_area_fraction: float
    harness: dict[str, Any]
    psr_area_km2: float | None = None
    tolerance_area_km2: float | None = None
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> PsrReference:
        """Load a committed reference document (the ``validation/*.reference.json`` artifacts)."""
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = doc.get("schema")
        if schema != VALIDATION_SCHEMA:
            raise ValueError(f"{path} has schema {schema!r}, expected {VALIDATION_SCHEMA!r}")
        reference = doc["reference"]
        budget = doc["error_budget"]
        return cls(
            region=str(doc["region"]),
            source=str(reference["source"]),
            psr_area_fraction=float(reference["psr_area_fraction"]),
            psr_area_km2=(
                float(reference["psr_area_km2"]) if "psr_area_km2" in reference else None
            ),
            tolerance_area_fraction=float(budget["psr_area_fraction_abs"]),
            tolerance_area_km2=(
                float(budget["psr_area_km2_abs"]) if "psr_area_km2_abs" in budget else None
            ),
            harness=dict(doc.get("harness", {})),
            notes=str(doc.get("notes", "")),
        )


@dataclass(frozen=True)
class PsrValidation:
    """The outcome of comparing a computed PSR mask against a :class:`PsrReference`."""

    region: str
    source: str
    psr_area_fraction: float
    psr_area_km2: float
    reference_area_fraction: float
    tolerance_area_fraction: float
    error_area_fraction: float
    passed: bool
    n_psr_cells: int
    n_cells: int
    n_void_cells: int
    resolution_m: float
    n_epochs: int
    semantics: str
    illumination_hash: str
    psr_hash: str

    def to_artifact(self) -> dict[str, Any]:
        """The committed result artifact — the evidence the harness actually ran on real data."""
        return {
            "schema": VALIDATION_SCHEMA,
            "region": self.region,
            "result": {
                "psr_area_fraction": self.psr_area_fraction,
                "psr_area_km2": self.psr_area_km2,
                "n_psr_cells": self.n_psr_cells,
                "n_cells": self.n_cells,
                "n_void_cells": self.n_void_cells,
                "resolution_m": self.resolution_m,
                "n_epochs": self.n_epochs,
                "semantics": self.semantics,
                "illumination_hash": self.illumination_hash,
                "psr_hash": self.psr_hash,
            },
            "reference": {
                "source": self.source,
                "psr_area_fraction": self.reference_area_fraction,
            },
            "error_budget": {"psr_area_fraction_abs": self.tolerance_area_fraction},
            "comparison": {
                "error_area_fraction": self.error_area_fraction,
                "passed": self.passed,
            },
        }


def psr_statistics(psr: PsrResult, *, resolution_m: float) -> dict[str, Any]:
    """Reduce a PSR mask to the scalars a published catalog quotes.

    The **area fraction** is over the whole modelled grid, and the **area** is that in km² at the
    grid's ground sample distance — the quantity LOLA-derived PSR catalogs tabulate. DEM voids are
    counted and reported separately: a void cell's PSR-ness is not trustworthy (its elevation was
    interpolated), so a reference comparison must know how much of the grid it covers. On the
    anchor's LOLA product the void count is zero, but a coarser or clipped DEM would not be.
    """
    mask = np.asarray(psr.mask, dtype=bool)
    void = np.asarray(psr.void_mask, dtype=bool)
    n_cells = int(mask.size)
    n_psr = int(mask.sum())
    cell_area_km2 = (resolution_m * resolution_m) / _M2_PER_KM2
    return {
        "psr_area_fraction": n_psr / n_cells,
        "psr_area_km2": n_psr * cell_area_km2,
        "n_psr_cells": n_psr,
        "n_cells": n_cells,
        "n_void_cells": int(void.sum()),
    }


def validate_psr(psr: PsrResult, reference: PsrReference, *, resolution_m: float) -> PsrValidation:
    """Compare a computed PSR mask against a published reference and its stated error budget.

    Passes iff ``|computed - reference| <= tolerance`` on the permanently-shadowed **area
    fraction**. The absolute (not relative) budget is deliberate: the quantity is itself a fraction
    in [0, 1], and the published references carry an absolute spread of the same kind.
    """
    stats = psr_statistics(psr, resolution_m=resolution_m)
    fraction = float(stats["psr_area_fraction"])
    error = abs(fraction - reference.psr_area_fraction)
    return PsrValidation(
        region=reference.region,
        source=reference.source,
        psr_area_fraction=fraction,
        psr_area_km2=float(stats["psr_area_km2"]),
        reference_area_fraction=reference.psr_area_fraction,
        tolerance_area_fraction=reference.tolerance_area_fraction,
        error_area_fraction=error,
        passed=error <= reference.tolerance_area_fraction,
        n_psr_cells=int(stats["n_psr_cells"]),
        n_cells=int(stats["n_cells"]),
        n_void_cells=int(stats["n_void_cells"]),
        resolution_m=resolution_m,
        n_epochs=psr.n_epochs,
        semantics=psr.semantics.value,
        illumination_hash=psr.illumination_hash,
        psr_hash=psr.psr_hash,
    )
