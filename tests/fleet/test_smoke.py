"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.fleet as fleet

    assert isinstance(fleet.__version__, str)
    assert fleet.__version__
