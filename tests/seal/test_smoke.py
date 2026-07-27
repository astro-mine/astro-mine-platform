"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.seal as seal

    assert isinstance(seal.__version__, str)
    assert seal.__version__
