"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.worlds as worlds

    assert isinstance(worlds.__version__, str)
    assert worlds.__version__
