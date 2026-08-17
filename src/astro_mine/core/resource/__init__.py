# SPDX-License-Identifier: Apache-2.0
"""Resource-field API — the uncertainty-first probabilistic resource contract (prospect.md).

A Core-owned thin Protocol for a probabilistic resource field: a query surface over a
(position[, epoch]) that returns *distributions* — mean, variance, quantiles, samples, and a
full posterior summary — never a bare point estimate. Both Prospect's sealed ground-truth
realizations and its evolving belief fields implement this one contract (prospect.md §2.2),
so autonomy runs identically against a synthetic world and a live estimate; the
geostatistical backend (GP, GMRF, deep-generative, grid) is a plugin behind it (prospect.md
§2.7).

Two safety properties are encoded structurally, in the *absence* of methods: there is **no
point-estimate-only accessor** (every query carries uncertainty; prospect.md §2.1) and **no
ground-truth accessor** on this agent-facing surface (ground-truth isolation; prospect.md
§9). Core owns only the *shape* — no inference, physics, or IO.

Public API:

- the contract — :class:`ResourceField` (runtime-checkable Protocol);
- the result — :class:`FieldDistribution` (and the :data:`Position` query type);
- conformance — :func:`check_resource_field` and :class:`ResourceFieldContractError`.
"""

from __future__ import annotations

from astro_mine.core.resource import conformance, model, protocol
from astro_mine.core.resource.conformance import (
    ResourceFieldContractError,
    check_resource_field,
)
from astro_mine.core.resource.model import FieldDistribution, Position
from astro_mine.core.resource.protocol import ResourceField

__all__ = [
    "FieldDistribution",
    "Position",
    "ResourceField",
    "ResourceFieldContractError",
    "check_resource_field",
    "conformance",
    "model",
    "protocol",
]
