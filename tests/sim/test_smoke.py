"""Smoke test: the package imports and exposes its version."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.sim as sim

    assert isinstance(sim.__version__, str)
    assert sim.__version__
