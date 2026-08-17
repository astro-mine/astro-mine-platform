# SPDX-License-Identifier: Apache-2.0
"""Astro-Mine-Surrogate — learned surrogate models with calibrated error bounds.

GNN emulators for the most expensive physics in :mod:`~astro_mine.sim` — above all granular
media and excavation contact — served as low-cost, Core-described plugins behind the
physics-step contract. Every surrogate ships a calibrated ``ErrorReport``: the error is the
product, not a footnote.

This module is the **contract layer** (RM-P1-SURR-01) and imports only Core + Pydantic: the
:class:`~astro_mine.surrogate.model.SurrogateModel` seam a learned fidelity tier satisfies, the
:class:`~astro_mine.surrogate.report.ErrorReport` it carries as its bound, and
:func:`~astro_mine.surrogate.manifest.build_surrogate_manifest`, which publishes it as a Core
:class:`~astro_mine.core.registry.PluginManifest`. The implementation lives in subpackages that
pull their own (optional) stacks:

- :mod:`astro_mine.surrogate.models` — the learned-DEM granular/excavation GNS: a deep ensemble
  with split-conformal calibrated bounds and an enforced trust region (RM-P1-SURR-02; torch).
- :mod:`astro_mine.surrogate.datagen` / :mod:`~astro_mine.surrogate.eval` /
  :mod:`~astro_mine.surrogate.retrain` — the offline build loop: sampling policy + Sobol/LHS
  design + active learning against a ``RolloutOracle`` seam, an immutable content-addressed
  dataset store, the coverage/error-budget promotion gate, and offline retrain with gated
  promotion (RM-P1-SURR-03; the ``[datasets]`` extra).
- :mod:`astro_mine.surrogate.serve` — the served fidelity tier: self-contained ONNX export,
  ONNX-Runtime inference, signed Hub publish, and fail-closed load (RM-P1-SURR-04; the
  ``[serve]`` extra). ``ServedBackend.NATIVE_GRAPH`` is declarable in a manifest but **has no
  serving runtime** — ONNX is the only served backend (see ``README.md`` § Known limitations).
- :mod:`astro_mine.surrogate.drift` — OOD/drift monitoring of live queries and the hybrid
  schedule-or-drift re-validation trigger (RM-P1-SURR-04; numpy only).

See ``docs/architecture/surrogate.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0"

# The Core interface versions Surrogate is built against — advertised here so consumers
# and the contract test cite one source of truth (defined in
# :mod:`astro_mine.surrogate._core`).
from astro_mine.surrogate._core import CORE_INTERFACES

# The three foundational contracts (RM-P1-SURR-01). ``__version__`` is defined above
# before these imports so :mod:`astro_mine.surrogate.manifest` can read it without a cycle.
from astro_mine.surrogate.enums import ChannelKind, PhysicsDomain, ServedBackend
from astro_mine.surrogate.manifest import SurrogateAttributes, build_surrogate_manifest
from astro_mine.surrogate.model import (
    ChannelVector,
    ParticleFields,
    Prediction,
    SurrogateModel,
    SurrogateState,
)
from astro_mine.surrogate.report import (
    Bound,
    CategoricalMetrics,
    ChannelError,
    ContinuousMetrics,
    CoveragePoint,
    ErrorReport,
    OracleRef,
    RolloutError,
    SubstitutionPolicy,
    TailBehavior,
    TrustRegion,
)

__all__ = [
    "CORE_INTERFACES",
    "Bound",
    "CategoricalMetrics",
    "ChannelError",
    "ChannelKind",
    "ChannelVector",
    "ContinuousMetrics",
    "CoveragePoint",
    "ErrorReport",
    "OracleRef",
    "ParticleFields",
    "PhysicsDomain",
    "Prediction",
    "RolloutError",
    "ServedBackend",
    "SubstitutionPolicy",
    "SurrogateAttributes",
    "SurrogateModel",
    "SurrogateState",
    "TailBehavior",
    "TrustRegion",
    "__version__",
    "build_surrogate_manifest",
]
