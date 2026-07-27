"""Smoke test: the package imports and exposes its version and submit()."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.cloud as cloud

    assert isinstance(cloud.__version__, str)
    assert cloud.__version__
    assert callable(cloud.submit)
