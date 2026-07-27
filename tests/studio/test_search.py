"""STUDIO-02 — pluggable search: DoE seeding, NSGA-II, backend registry."""

from __future__ import annotations

import pytest

from astro_mine.studio.designspace import (
    NSGAII,
    SearchBackend,
    get_backend,
    register_backend,
    registered_backends,
)
from astro_mine.studio.models import AssetChoice, DecisionSpace

_SPACE = DecisionSpace(
    assets=[AssetChoice(sadf_ref="rover", max_count=6), AssetChoice(sadf_ref="relay", max_count=4)]
)


def test_builtin_backends_registered() -> None:
    assert {"nsga2", "nsga2-lhs"} <= set(registered_backends())
    assert isinstance(get_backend("nsga2"), SearchBackend)


def test_get_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown search backend"):
        get_backend("does-not-exist")


def test_register_custom_backend() -> None:
    register_backend("nsga2-test-clone", lambda: NSGAII(doe="lhs"))
    assert "nsga2-test-clone" in registered_backends()


@pytest.mark.parametrize("doe", ["sobol", "lhs"])
def test_initial_doe_is_in_bounds_and_deterministic(doe: str) -> None:
    backend = NSGAII(doe=doe)
    a = backend.initial(_SPACE, seed=3, n=8)
    b = backend.initial(_SPACE, seed=3, n=8)
    assert a == b  # deterministic in the seed
    assert len(a) == 8
    for vector in a:
        assert 0 <= vector[0] <= 6 and 0 <= vector[1] <= 4


def test_evolve_produces_in_bounds_children() -> None:
    backend = NSGAII()
    ranked = [
        ((4, 1), (10.0, 2.0)),
        ((1, 3), (3.0, 6.0)),
        ((2, 2), (5.0, 5.0)),
        ((5, 0), (12.0, 1.0)),
    ]
    children = backend.evolve(ranked, (True, True), _SPACE, seed=1, n=16)
    assert len(children) == 16
    assert children == backend.evolve(ranked, (True, True), _SPACE, seed=1, n=16)  # deterministic
    for vector in children:
        assert 0 <= vector[0] <= 6 and 0 <= vector[1] <= 4


def test_evolve_reseeds_when_population_too_small() -> None:
    backend = NSGAII()
    # fewer than two ranked points → fall back to a DoE reseed (still deterministic)
    children = backend.evolve([((1, 1), (1.0, 1.0))], (True, True), _SPACE, seed=2, n=4)
    assert len(children) == 4
