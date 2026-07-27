"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.learn as learn

    assert isinstance(learn.__version__, str)
    assert learn.__version__
