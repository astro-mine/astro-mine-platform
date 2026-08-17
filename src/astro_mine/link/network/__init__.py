# SPDX-License-Identifier: Apache-2.0
"""Delay-tolerant networking: CGR contact graph + store-and-forward delivery (RM-P1-LINK-11).

The abstract DTN layer on top of the constellation contact graph: :class:`ContactGraph`
(a CGR-style, per-node index of time-ordered :class:`Contact`\\ s derived from a Core
``ContactPlan``) and :class:`DeliveryModel` (store-and-forward modeled delivery-time or
non-delivery, with a multi-fidelity instantaneous↔store-and-forward dial). Link *models*
store-and-forward and supplies the contact plan; it is not a live DTN agent (link.md §1, §11).

Backlog: RM-P1-LINK-11 -- astro-mine-link#18
"""

from __future__ import annotations

from astro_mine.link.network._contactgraph import Contact, ContactGraph
from astro_mine.link.network._dtn import Delivery, DeliveryFidelity, DeliveryModel
from astro_mine.link.network._errors import LinkNetworkError

__all__ = [
    "Contact",
    "ContactGraph",
    "Delivery",
    "DeliveryFidelity",
    "DeliveryModel",
    "LinkNetworkError",
]
