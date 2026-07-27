"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.spice as spice

    assert isinstance(spice.__version__, str)
    assert spice.__version__
