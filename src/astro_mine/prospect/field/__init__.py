# SPDX-License-Identifier: Apache-2.0
"""ResourceField contract — uncertainty-first; no point-estimate API (prospect.md §2/§3).

Prospect does **not** define its own resource-field contract — the narrow waist owns it. This
module re-exports the Core-owned :class:`~astro_mine.core.resource.ResourceField` Protocol, its
:class:`~astro_mine.core.resource.FieldDistribution` result and :data:`~astro_mine.core.resource.\
Position` query type, and the :func:`~astro_mine.core.resource.check_resource_field` conformance
utility, so every Prospect field, backend, and consumer codes against one contract (keep to the
waist, don't widen it — CONTRIBUTING.md). Two safety properties are encoded in the *absence* of
methods on that contract: no point-estimate-only accessor (prospect.md §2.1) and no ground-truth
accessor on the agent-facing surface (prospect.md §9).

On top of the waist this package adds the Prospect-side affordances the sealed ground-truth and
belief variants share:

- :class:`~astro_mine.prospect.field.metadata.FieldMetadata` (and :class:`~astro_mine.prospect.\
  field.metadata.FieldGrid`) — the species/unit and CRS/grid binding, consistent with Worlds;
- :class:`~astro_mine.prospect.field.base.BaseResourceField` — the shared base that makes the two
  variants identical at the contract surface and composes the uncertainty-first ``posterior``.

Concrete fields and inference backends land in later items (RM-P0-PROSPECT-02/04).

Backlog: RM-P0-PROSPECT-01 — astro-mine-prospect#1
"""

from __future__ import annotations

from astro_mine.core.resource import (
    FieldDistribution,
    Position,
    ResourceField,
    ResourceFieldContractError,
    check_resource_field,
)
from astro_mine.prospect.field.base import DEFAULT_QUANTILES, BaseResourceField
from astro_mine.prospect.field.metadata import FieldGrid, FieldMetadata

__all__ = [
    "DEFAULT_QUANTILES",
    "BaseResourceField",
    "FieldDistribution",
    "FieldGrid",
    "FieldMetadata",
    "Position",
    "ResourceField",
    "ResourceFieldContractError",
    "check_resource_field",
]
