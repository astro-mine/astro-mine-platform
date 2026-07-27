"""Smoke test: the package imports, exposes its version, and Core is wired in."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.guard as guard

    assert isinstance(guard.__version__, str)
    assert guard.__version__


def test_core_compat_hook_available() -> None:
    # Placeholder Core-version contract hook (RM-P0-CORE-07): prove the tag-pinned
    # astro-mine-core dependency resolves and its interface-version negotiation
    # machinery is importable. The concrete interfaces Guard is built against — the
    # Core Policy/Planner API its PolicyShield implements and the Environment API it
    # reads — are declared and asserted with the RM-P1-GUARD-* feature work; this only
    # proves the wiring exists.
    from astro_mine.core import compat

    assert compat.CORE_INTERFACE_VERSIONS
    # The compatibility rule is reflexive: a version always satisfies itself.
    for interface_version in compat.CORE_INTERFACE_VERSIONS.values():
        assert compat.check_compatible(interface_version, interface_version)
