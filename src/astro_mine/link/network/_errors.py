# SPDX-License-Identifier: Apache-2.0
"""Network-layer (DTN) errors."""

from __future__ import annotations

from astro_mine.link.products._errors import LinkProductsError

__all__ = ["LinkNetworkError"]


class LinkNetworkError(LinkProductsError):
    """A store-and-forward delivery could not be modeled — an unknown endpoint or a malformed
    query. Subclasses :class:`~astro_mine.link.products.LinkProductsError` so a caller can catch
    every Link product failure uniformly. Non-delivery (no path within the horizon) is **not**
    an error — it is a modeled outcome (:class:`~astro_mine.link.network.Delivery` with
    ``delivered=False``); Link degrades loudly on bad *inputs*, honestly on bad *connectivity*."""
