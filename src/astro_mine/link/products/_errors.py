# SPDX-License-Identifier: Apache-2.0
"""Errors for the Link products layer."""

from __future__ import annotations

__all__ = ["LinkProductsError"]


class LinkProductsError(RuntimeError):
    """A Link product cannot be assembled from its inputs.

    Raised when a contact window references a node absent from the declared contact
    graph, or a mask/sampler is queried for an unknown agent — a topology
    inconsistency Link surfaces rather than silently dropping. A malformed assembled
    :class:`~astro_mine.core.messages.ContactPlan` instead raises Core's
    ``MessagesValidationError`` from the consumer-driven contract check
    (:func:`~astro_mine.core.messages.validate_contact_plan`); Link **degrades loudly**
    rather than emitting a plan a downstream consumer cannot trust (link.md §2.9).
    """
