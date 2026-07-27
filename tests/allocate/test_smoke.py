"""Smoke test: the package imports, exposes its version, and can reach Core's
interface-version negotiation surface (the contract-test hook this package will use as
its ``RM-P1-ALLOC-*`` interfaces land)."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.allocate as allocate

    assert isinstance(allocate.__version__, str)
    assert allocate.__version__


def test_core_compat_contract_hook() -> None:
    # Placeholder for the Core-version contract test. Allocate builds against Core's
    # narrow waist; once its interface surface exists, this becomes an
    # ``assert_core_compatible({...})`` check. For now, prove the pinned Core exposes
    # the compat negotiation utilities the check will use.
    from astro_mine.core import compat

    assert isinstance(compat.CORE_INTERFACE_VERSIONS, dict)
    assert compat.CORE_INTERFACE_VERSIONS
