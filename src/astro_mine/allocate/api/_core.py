# SPDX-License-Identifier: Apache-2.0
"""Allocate's binding to the Core narrow waist (scaffold; prerequisite for RM-P1-ALLOC-*).

The single place that declares which Core interface versions Allocate is built against,
plus a thin compatibility check over them. Allocate implements the **allocation
sub-interface** of Core's Policy/Planner API — a Core-described plugin that Mind loads and
delegates the combinatorial assignment to (allocate.md §6); it *consumes* the waist and
never widens it. This module only records the contract Allocate builds against so the
plugin manifest, the registry, and the contract test all cite one source of truth.
"""

from __future__ import annotations

from astro_mine.core import compat

__all__ = ["CORE_INTERFACES", "assert_core_compatible"]

#: Core interface versions Allocate is built against — the input to the registry's version
#: negotiation. Allocate implements the ``policy`` (Policy/Planner) contract as the
#: allocation sub-interface and builds a :class:`~astro_mine.core.registry.PluginManifest`
#: against the ``registry`` contract (RM-P1-ALLOC-01). Pinned to this Core, not invented.
CORE_INTERFACES: dict[str, str] = {"policy": "0.1.0", "registry": "0.1.0"}


def assert_core_compatible() -> None:
    """Assert the installed Core satisfies the interface versions Allocate is built against.

    Delegates to :func:`astro_mine.core.compat.assert_core_compatible`. This is the
    consumer-driven contract test Allocate runs in its own CI (RM-P0-CORE-07); raises
    :class:`~astro_mine.core.compat.IncompatibleCoreInterface` on any mismatch.
    """
    compat.assert_core_compatible(CORE_INTERFACES)
