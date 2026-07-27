"""ErrorReport validation, immutability, and content-addressing (RM-P1-SURR-01).

The error is the product: these tests prove an ErrorReport is a machine-consumable,
per-channel-typed, immutable, content-addressed bound — and that malformed bounds fail
loudly rather than silently (surrogate.md §2 principle 1, §6; core.md principle 7).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.surrogate import (
    Bound,
    CategoricalMetrics,
    ChannelError,
    ChannelKind,
    ContinuousMetrics,
    CoveragePoint,
    RolloutError,
    TailBehavior,
    TrustRegion,
)
from tests.surrogate.factories import granular_report, illumination_report

_TAIL = TailBehavior(p95_abs_error=3.0, p99_abs_error=5.0, max_abs_error=9.0)
_CONT = ContinuousMetrics(
    unit="N", rmse=1.0, coverage=[CoveragePoint(nominal=0.9, empirical=0.9)], tail=_TAIL
)
_CAT = CategoricalMetrics(
    classes=["a", "b"], accuracy=0.9, reliability=[CoveragePoint(nominal=0.9, empirical=0.9)]
)


def test_valid_reports_build_for_both_families() -> None:
    assert granular_report().domain.value == "granular_excavation"
    illum = illumination_report()
    # A field surrogate carries a categorical channel and no rollout facet.
    assert illum.rollout is None
    kinds = {c.kind for c in illum.channels}
    assert ChannelKind.CATEGORICAL in kinds and ChannelKind.CONTINUOUS in kinds


def test_report_is_frozen() -> None:
    report = granular_report()
    with pytest.raises(ValidationError):
        report.surrogate_name = "mutated"  # type: ignore[misc]


def test_report_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        granular_report(unexpected="x")


def test_continuous_channel_requires_a_continuous_block() -> None:
    with pytest.raises(ValidationError, match="continuous"):
        ChannelError(channel="f", kind=ChannelKind.CONTINUOUS, categorical=_CAT)


def test_categorical_channel_requires_a_categorical_block() -> None:
    with pytest.raises(ValidationError, match="categorical"):
        ChannelError(channel="v", kind=ChannelKind.CATEGORICAL, continuous=_CONT)


def test_channel_cannot_carry_both_blocks() -> None:
    with pytest.raises(ValidationError):
        ChannelError(channel="f", kind=ChannelKind.CONTINUOUS, continuous=_CONT, categorical=_CAT)


def test_a_channel_can_be_continuous_or_categorical() -> None:
    assert ChannelError(channel="f", kind=ChannelKind.CONTINUOUS, continuous=_CONT).continuous
    assert ChannelError(channel="v", kind=ChannelKind.CATEGORICAL, categorical=_CAT).categorical


def test_duplicate_channel_names_are_rejected() -> None:
    dup = ChannelError(channel="f", kind=ChannelKind.CONTINUOUS, continuous=_CONT)
    with pytest.raises(ValidationError, match="duplicate channel"):
        granular_report(channels=[dup, dup])


def test_bound_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="below low"):
        Bound(low=1.0, high=0.0)


def test_trust_region_requires_at_least_one_dimension() -> None:
    with pytest.raises(ValidationError):
        TrustRegion(bounds={})


def test_rollout_horizon_length_must_match() -> None:
    with pytest.raises(ValidationError, match="horizon"):
        RolloutError(horizon_steps=3, rmse_by_horizon=[1.0, 2.0])


def test_coverage_probabilities_are_bounded() -> None:
    with pytest.raises(ValidationError):
        CoveragePoint(nominal=1.5, empirical=0.9)


def test_content_hash_is_deterministic_and_sensitive() -> None:
    a = granular_report()
    b = granular_report()
    assert a.content_hash() == b.content_hash()  # same content -> same address
    assert a.content_hash().startswith("sha256:")
    # A changed bound changes the address (immutable artifact identity).
    c = granular_report(surrogate_version="0.2.0")
    assert c.content_hash() != a.content_hash()


def test_content_hash_is_stable_across_channel_order_of_construction() -> None:
    # The canonical JSON sorts keys, so identical content hashes identically regardless
    # of dict insertion order in the trust region.
    r1 = granular_report(
        trust_region=TrustRegion(
            bounds={"a": Bound(low=0.0, high=1.0), "b": Bound(low=0.0, high=2.0)}
        )
    )
    r2 = granular_report(
        trust_region=TrustRegion(
            bounds={"b": Bound(low=0.0, high=2.0), "a": Bound(low=0.0, high=1.0)}
        )
    )
    assert r1.content_hash() == r2.content_hash()


def test_report_requires_at_least_one_channel() -> None:
    with pytest.raises(ValidationError):
        granular_report(channels=[])
