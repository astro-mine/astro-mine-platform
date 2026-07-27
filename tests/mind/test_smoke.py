"""Smoke test: the package imports, exposes its version, and can reach Core's
interface-version negotiation machinery.

The scaffold declares no feature interfaces yet — the interface versions this package
implements against ``astro_mine.core.compat`` land with the RM-P1-MIND-* issues — so the
contract-test hook below is a placeholder that proves Core's machinery is importable and
callable from Mind.
"""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.mind as mind

    assert isinstance(mind.__version__, str)
    assert mind.__version__


def test_core_compat_hook() -> None:
    # Placeholder Core-version contract-test hook. Mind claims no feature interfaces at
    # scaffold time (those arrive with RM-P1-MIND-*), but Core's negotiation machinery
    # must be importable and callable; an empty claim is trivially satisfied.
    from astro_mine.core import compat

    assert compat.CORE_INTERFACE_VERSIONS  # Core publishes a non-empty interface set
    compat.assert_core_compatible({})  # no claimed interfaces yet -> must not raise
