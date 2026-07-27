"""Surrogate's binding to the Core narrow waist (the foundation the RM-P1-SURR-* surface rests on).

The single place that declares which Core interface versions Surrogate is built against,
plus a thin compatibility check over them. A trained surrogate is a fidelity *tier*
behind the Core physics-step / Environment contract — a Core-described plugin that
:mod:`~astro_mine.sim` loads (surrogate.md §"Core boundary"); it *consumes* the waist and
never widens it. This module only records the contract Surrogate builds against so the
served plugin manifest, the registry, and the contract test all cite one source of truth.
"""

from __future__ import annotations

from astro_mine.core import compat

__all__ = ["CORE_INTERFACES", "assert_core_compatible"]

#: Core interface versions Surrogate is built against — the input to the registry's
#: version negotiation. A surrogate is a fidelity *tier* behind the Core ``env``
#: (Environment) contract, and RM-P1-SURR-01 adds the ``registry`` manifest contract it
#: builds a :class:`~astro_mine.core.registry.PluginManifest` against
#: (:mod:`astro_mine.surrogate.manifest`). A *field* surrogate's manifest declares the
#: ``world_provider`` interface instead of ``env`` at build time, per its domain — that is
#: a per-artifact declaration, distinct from the surface this *package* is compiled against.
CORE_INTERFACES: dict[str, str] = {"env": "0.1.0", "registry": "0.1.0"}


def assert_core_compatible() -> None:
    """Assert the installed Core satisfies the interface versions Surrogate is built against.

    Delegates to :func:`astro_mine.core.compat.assert_core_compatible`. This is the
    consumer-driven contract test Surrogate runs in its own CI (RM-P0-CORE-07); raises
    :class:`~astro_mine.core.compat.IncompatibleCoreInterface` on any mismatch.
    """
    compat.assert_core_compatible(CORE_INTERFACES)
