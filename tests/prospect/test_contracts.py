"""Contract stubs: each Prospect subpackage exists and its placeholder is wired.

Pins the public surface and the not-yet-implemented entry points so the backlog has a
concrete starting point. Replace the ``NotImplementedError`` checks with real behavior
as each RM-P0-PROSPECT-* item lands.
"""

from __future__ import annotations

from astro_mine.prospect import (
    backends,
    belief,
    calibration,
    field,
    infogain,
    isolation,
    priors,
)


def test_subpackages_import() -> None:
    for module in (field, backends, priors, belief, isolation, infogain, calibration):
        assert module is not None


def test_resource_field_contract_is_exported() -> None:
    assert hasattr(field, "ResourceField")


def test_backends_expose_the_concrete_fields() -> None:
    # The factory is implemented (RM-P0-PROSPECT-02); behavior lives in test_backends.py.
    assert hasattr(backends, "make_backend")
    assert hasattr(backends, "GPField")
    assert hasattr(backends, "GridField")
