"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.core as core

    assert isinstance(core.__version__, str)
    assert core.__version__
