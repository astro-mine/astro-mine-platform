"""Public-surface pins: subpackages import, and the not-yet-implemented seams stay stubbed.

As each RM-P0-FLEET-* item lands it replaces its stub with real behavior; this file
keeps the still-deferred entry points honest. The ``packaging`` surface is exercised in
``test_packaging.py``; the reference ``library`` is implemented in issue #4 and exercised in
``test_library.py``. ``fleet.cli`` is absent by design — platform#1 moved every command surface to
``astro-mine-cli`` — and this file still imported it, so the whole Fleet suite failed to collect.
"""

from __future__ import annotations

from astro_mine.fleet import catalog, fidelity, importers, library, lint, packaging


def test_subpackages_import() -> None:
    for module in (importers, lint, library, fidelity, packaging, catalog):
        assert module is not None


def test_reference_library_is_populated() -> None:
    # RM-P0-FLEET-04 (issue #4): the stub now serves the anchor roster. Detailed
    # coverage lives in test_library.py; this only pins that the seam is live.
    names = library.available()
    assert names, "reference library must expose the anchor roster"
    doc = library.load_reference(names[0])
    assert doc.asset.identity.id
