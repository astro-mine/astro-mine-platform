"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.surrogate as surrogate

    assert isinstance(surrogate.__version__, str)
    assert surrogate.__version__
