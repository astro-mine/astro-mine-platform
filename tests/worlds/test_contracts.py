"""Public-surface pins: subpackages import, and the not-yet-implemented seams stay stubbed.

As each RM-P0-WORLDS-* item lands it replaces its stub with real behavior; this file
keeps the still-deferred entry points honest. The ``terrain``/``crs`` surfaces are
implemented in issue #1 (``test_terrain.py`` / ``test_crs.py``), ``illumination`` in
issue #3 (``test_illumination.py``), ``thermal`` in issue #4 (``test_thermal.py``),
``regolith`` in issue #5 (``test_regolith.py``), and ``provider`` in issue #6
(``test_provider.py``). The ``spec`` bundle is issue #7 (``test_spec.py``). SPICE
frames/geometry moved to the shared ``astro_mine.spice`` package (RFC-0002, issue #19).
"""

from __future__ import annotations

from astro_mine.worlds import (
    illumination,
    provider,
    regolith,
    spec,
    terrain,
    thermal,
)


def test_subpackages_import() -> None:
    for module in (terrain, illumination, thermal, regolith, provider, spec):
        assert module is not None


def test_spec_bundle_surface_is_implemented() -> None:
    # RM-P0-WORLDS-07 (issue #7) — behavior lives in test_spec.py.
    for name in ("WorldSpec", "WorldBundle", "build_world_bundle"):
        assert hasattr(spec, name)
