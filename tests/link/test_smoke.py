"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.link as link

    assert isinstance(link.__version__, str)
    assert link.__version__
