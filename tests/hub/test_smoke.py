"""Smoke test: the package imports, exposes its version, and can reach Core's compat hook.

Scaffold-only. The Core-version contract test is a *placeholder*: Hub does not yet declare
which Core interfaces it implements (that lands with the RM-P1-HUB-* feature issues), so we
only assert the tag-pinned Core resolves and its negotiation machinery
(:mod:`astro_mine.core.compat`) is importable and self-consistent.
"""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.hub as hub

    assert isinstance(hub.__version__, str)
    assert hub.__version__


def test_core_compat_hook_available() -> None:
    # Placeholder Core-version contract hook: the tag-pinned Core is importable and its
    # interface-negotiation machinery is wired. Feature issues will replace this with a
    # concrete assert_core_compatible(HUB's claimed interfaces) contract test.
    from astro_mine.core import compat

    assert compat.CORE_INTERFACE_VERSIONS  # non-empty published interface set
    # The compatibility rule is reflexive: every version satisfies itself.
    for interface_version in compat.CORE_INTERFACE_VERSIONS.values():
        assert compat.check_compatible(interface_version, interface_version)
