"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.prospect as prospect

    assert isinstance(prospect.__version__, str)
    assert prospect.__version__
