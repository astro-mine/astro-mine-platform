# SPDX-License-Identifier: Apache-2.0
"""Inference backends behind the ResourceField contract (RM-P0-PROSPECT-02; RM-P1-PROSPECT-10).

Interchangeable geostatistical backends, all implementing the Core
:class:`~astro_mine.prospect.field.ResourceField` contract via
:class:`~astro_mine.prospect.field.BaseResourceField`:

- :class:`~astro_mine.prospect.backends.gp.GPField` — a GPyTorch sparse/variational GP (the
  principled-uncertainty path, prospect.md §11);
- :class:`~astro_mine.prospect.backends.grid.GridField` — a simple Gaussian grid (the fast,
  dependency-light reference path);
- :class:`~astro_mine.prospect.backends.gmrf.GMRFField` — a GMRF/SPDE sparse-precision field for
  **large lattice domains** (RM-P1-PROSPECT-10), parametric uncertainty;
- :class:`~astro_mine.prospect.backends.generative.GenerativeEnsembleField` — a deep-generative
  / normalizing-flow field for **non-Gaussian / multimodal** structure (RM-P1-PROSPECT-10),
  ensemble uncertainty.

:func:`make_backend` selects between them by ``kind`` — the field-backends extension point — so the
choice is a config detail invisible to consumers (Sim/Bench): every kind returns a field exposing
the same uncertainty-first surface. New backends register here, behind the existing extension point,
with **no Core change** (the exit criterion for RM-P1-PROSPECT-10).

Backlog: RM-P0-PROSPECT-02 / RM-P1-PROSPECT-10 — astro-mine-prospect#20
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from astro_mine.core.resource import Position

# Eagerly export only the dependency-light members (numpy + Core), so importing this package — and
# therefore the resource-field `from_bundle` reconstruction path (RM-P1-PROSPECT-13) — stays
# torch-free. The inference-heavy backends (GP → gpytorch/torch, generative → torch, GMRF → scipy)
# import lazily, on first attribute access (PEP 562) and per `make_backend` kind, so the public API
# is unchanged but a consumer that only needs the grid field never pays for the inference stack.
from astro_mine.prospect.backends.grid import GridField
from astro_mine.prospect.field.base import BaseResourceField
from astro_mine.prospect.field.metadata import FieldMetadata

if TYPE_CHECKING:
    from astro_mine.prospect.backends.generative import GenerativeEnsembleField
    from astro_mine.prospect.backends.gmrf import GMRFField
    from astro_mine.prospect.backends.gp import GPField

__all__ = [
    "GMRFField",
    "GPField",
    "GenerativeEnsembleField",
    "GridField",
    "make_backend",
]

_KINDS = ("gp", "grid", "gmrf", "generative")

#: The inference-heavy backends, imported lazily by :func:`__getattr__` so ``import
#: astro_mine.prospect.backends`` (and the grid-only ``from_bundle`` path) stays torch-free.
_LAZY_BACKENDS = {
    "GenerativeEnsembleField": "astro_mine.prospect.backends.generative",
    "GMRFField": "astro_mine.prospect.backends.gmrf",
    "GPField": "astro_mine.prospect.backends.gp",
}


def __getattr__(name: str) -> Any:
    """Import an inference-heavy backend on first access (PEP 562), keeping the package light."""
    module_name = _LAZY_BACKENDS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)


def make_backend(
    kind: str,
    metadata: FieldMetadata,
    *,
    train_points: Sequence[Position] | None = None,
    train_values: Sequence[float] | None = None,
    **config: Any,
) -> BaseResourceField:
    """Construct a ResourceField backend by ``kind`` — the field-backends extension point.

    Every kind (``"gp"``, ``"grid"``, ``"gmrf"``, ``"generative"``) takes the same ``metadata`` and
    optional ``train_points``/``train_values`` observations and returns a
    :class:`~astro_mine.prospect.field.BaseResourceField` honoring the Core contract — so a consumer
    is identical across backends. Backend-specific knobs (kernel iterations, prior, correlation
    length, ensemble size, …) pass through ``config``. An unknown ``kind`` raises ``ValueError``.
    """
    if kind == "grid":
        return GridField.build(
            metadata, train_points=train_points, train_values=train_values, **config
        )
    if kind == "gp":
        from astro_mine.prospect.backends.gp import GPField

        return GPField(metadata, train_points=train_points, train_values=train_values, **config)
    if kind == "gmrf":
        from astro_mine.prospect.backends.gmrf import GMRFField

        return GMRFField.build(
            metadata, train_points=train_points, train_values=train_values, **config
        )
    if kind == "generative":
        from astro_mine.prospect.backends.generative import GenerativeEnsembleField

        return GenerativeEnsembleField.build(
            metadata, train_points=train_points, train_values=train_values, **config
        )
    raise ValueError(f"unknown backend kind {kind!r}; known kinds are {_KINDS}")
