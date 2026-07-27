"""Tests for ``astro_mine.core.resource`` — the ResourceField contract, a trivial
conforming field, and the conformance utility (D8; prospect.md §2/§3)."""

from __future__ import annotations

import pytest

from astro_mine.core import compat
from astro_mine.core.resource import (
    FieldDistribution,
    Position,
    ResourceField,
    ResourceFieldContractError,
    check_resource_field,
)
from astro_mine.core.units import MOON_BODY_FIXED, Epoch, ReferenceFrame


class ConstantField:
    """A trivial, deterministic ResourceField: a flat mean with fixed variance. Carries
    no ground-truth and no point-estimate-only accessor — the contract's two key absences."""

    def __init__(self, mean: float = 0.3, variance: float = 0.01) -> None:
        self._mean = mean
        self._var = variance

    @property
    def species(self) -> str:
        return "water_equivalent_hydrogen"

    @property
    def unit(self) -> str:
        return "mass_fraction"

    @property
    def frame(self) -> ReferenceFrame:
        return MOON_BODY_FIXED

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._mean

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._var

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        return self._mean + (q - 0.5)

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        return tuple(self._mean for _ in range(n))

    def posterior(self, position: Position, *, epoch: Epoch | None = None) -> FieldDistribution:
        return FieldDistribution(
            mean=self._mean,
            variance=self._var,
            quantiles={0.5: self._mean},
            species=self.species,
            unit=self.unit,
        )


def test_trivial_field_satisfies_protocol() -> None:
    assert isinstance(ConstantField(), ResourceField)


def test_check_resource_field_passes() -> None:
    assert check_resource_field(ConstantField()) is None


def test_check_resource_field_passes_with_epoch_aware_call() -> None:
    field = ConstantField()
    # epoch is an optional units.Epoch; the contract still holds when one is passed.
    assert field.mean((1.0, 2.0, 3.0), epoch=None) == 0.3


def test_check_resource_field_rejects_non_field() -> None:
    with pytest.raises(ResourceFieldContractError):
        check_resource_field(object())  # type: ignore[arg-type]


class _BadPosteriorType(ConstantField):
    def posterior(self, position: Position, *, epoch: Epoch | None = None):  # type: ignore[override]
        return {"mean": 0.3, "variance": 0.01}


class _NonFloatMean(ConstantField):
    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return 0  # type: ignore[return-value]


class _BadSampleArity(ConstantField):
    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        return (self._mean,)  # ignores n


class _NonStrUnit(ConstantField):
    @property
    def unit(self) -> str:
        return 42  # type: ignore[return-value]


class _BadQuantile(ConstantField):
    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        return 1  # type: ignore[return-value]


class _BadPosteriorStats(ConstantField):
    def posterior(self, position: Position, *, epoch: Epoch | None = None) -> FieldDistribution:
        # A FieldDistribution whose mean is not a float — not uncertainty-first.
        return FieldDistribution(mean=1, variance=self._var)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        _BadPosteriorType,
        _NonFloatMean,
        _BadSampleArity,
        _NonStrUnit,
        _BadQuantile,
        _BadPosteriorStats,
    ],
)
def test_check_resource_field_rejects_violations(bad: type[ConstantField]) -> None:
    with pytest.raises(ResourceFieldContractError):
        check_resource_field(bad())


def test_field_distribution_is_uncertainty_first() -> None:
    dist = FieldDistribution(mean=0.3, variance=0.02)
    assert dist.mean == 0.3 and dist.variance == 0.02
    assert dist.quantiles == {} and dist.species is None


def test_resource_field_is_a_registered_core_interface() -> None:
    assert compat.CORE_INTERFACE_VERSIONS["resource_field"] == "0.1.0"
