"""Closed vocabularies for run-provenance.

Run provenance carries **no** closed enum vocabulary in v0.1: engine names, fidelity-tier
labels, RNG-stream names, and error-budget metric names are all free-form strings (the
closed asset-fidelity vocabulary is Core-owned in :mod:`astro_mine.core.sadf.enums`, not
duplicated here). This module exists so the schema drift guard
(``scripts/check_model_drift.py``) imports every component uniformly; enums are added
here additively if a run-provenance field ever needs a Core-owned closed set.
"""

from __future__ import annotations

__all__: list[str] = []
