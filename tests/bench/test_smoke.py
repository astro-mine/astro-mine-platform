"""Smoke test: the package imports and exposes its version and local scoring path."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.bench as bench

    assert isinstance(bench.__version__, str)
    assert bench.__version__
    assert callable(bench.run)
