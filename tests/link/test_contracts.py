"""Contract stubs: each Link subpackage exists and its placeholder is wired.

Pins the public surface and the not-yet-implemented entry points so the backlog has a
concrete starting point. Replace the ``NotImplementedError`` checks with real behavior
as each RM-P0-LINK-* item lands.
"""

from __future__ import annotations

import pytest

from astro_mine.link import budget, cache, geometry, products, windows


def test_subpackages_import() -> None:
    for module in (geometry, windows, budget, products, cache):
        assert module is not None


def test_los_degrades_loudly_without_a_world() -> None:
    # RM-P0-LINK-01 landed: compute_los composes SPICE geometry with the Core WorldProvider
    # and refuses to assume connectivity. Full behavior lives in test_geometry.py.
    with pytest.raises(geometry.LinkGeometryError):
        geometry.compute_los(None, None, None, world=None)
