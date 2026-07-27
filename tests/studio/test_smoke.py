"""Smoke test: the package imports, exposes its version, and is Core-compatible."""

from __future__ import annotations


def test_package_imports() -> None:
    import astro_mine.studio as studio

    assert isinstance(studio.__version__, str)
    assert studio.__version__


def test_core_interface_contract() -> None:
    # Consumer-driven Core-version contract (conventions.md §11). Studio consumes the
    # objective (ObjectiveSpec), env + policy (the design-loop Protocols), messages
    # (ActionBatch/Observation), and registry (plugin manifest) interfaces; assert the
    # pinned Core satisfies the versions this package implements against.
    from astro_mine.core import compat

    compat.assert_core_compatible(
        {
            "objective": "0.1.0",
            "env": "0.1.0",
            "policy": "0.1.0",
            "messages": "0.1.0",
            "registry": "0.1.0",
        }
    )
