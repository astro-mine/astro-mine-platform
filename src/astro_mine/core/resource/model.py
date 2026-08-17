# SPDX-License-Identifier: Apache-2.0
"""Resource-field API — result container (RM-P0-CORE; prospect.md §2/§3).

The frozen, uncertainty-first return of :meth:`ResourceField.posterior`: a distributional
summary at a queried point, carrying a posterior mean **and** a calibrated uncertainty
(variance, optional quantiles) — never a bare point estimate (prospect.md §2.1 "uncertainty
is the product, not a footnote"). A lightweight in-memory dataclass, not a wire document.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = ["FieldDistribution", "Position"]

#: A 3-D query point in the field's reference frame (SI metres).
Position = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class FieldDistribution:
    """The distributional summary of a resource field at a point — the return of
    :meth:`~astro_mine.core.resource.ResourceField.posterior`.

    Uncertainty-first by construction: ``mean`` is always paired with ``variance`` (and,
    optionally, a ``quantiles`` map from quantile level ``q`` in ``[0, 1]`` to value), so a
    consumer can never read a point estimate without its uncertainty (prospect.md §2.1).
    ``species`` and ``unit`` echo the field's resource species and SI unit for provenance.
    """

    mean: float
    variance: float
    quantiles: Mapping[float, float] = field(default_factory=dict)
    species: str | None = None
    unit: str | None = None
