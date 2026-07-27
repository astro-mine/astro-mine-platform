"""Public-surface pins: subpackages import, and the not-yet-implemented seams stay stubbed.

As each RM-P0-FLEET-* item lands it replaces its stub with real behavior; this file
keeps the still-deferred entry points honest. The ``cli`` and ``packaging`` surfaces are
implemented in issue #1 and exercised in ``test_cli.py`` / ``test_packaging.py``; the
reference ``library`` is implemented in issue #4 and exercised in ``test_library.py``.
"""

from __future__ import annotations

from astro_mine.fleet import catalog, cli, fidelity, importers, library, lint, packaging


def test_subpackages_import() -> None:
    for module in (cli, importers, lint, library, fidelity, packaging, catalog):
        assert module is not None


def test_reference_library_is_populated() -> None:
    # RM-P0-FLEET-04 (issue #4): the stub now serves the anchor roster. Detailed
    # coverage lives in test_library.py; this only pins that the seam is live.
    names = library.available()
    assert names, "reference library must expose the anchor roster"
    doc = library.load_reference(names[0])
    assert doc.asset.identity.id
