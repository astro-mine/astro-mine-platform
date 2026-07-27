"""Valid ``ErrorReport`` fixtures for the two contract families (test helpers).

``granular_report`` is the RM-P1-SURR-02 excavation case: all-continuous channels with an
autoregressive rollout facet. ``illumination_report`` is the RM-P1-WORLDS-10 field case: a
categorical visibility channel + a continuous flux channel, no rollout. Together they
exercise the contract's domain-generality.
"""

from __future__ import annotations

from typing import Any

from astro_mine.surrogate import (
    Bound,
    CategoricalMetrics,
    ChannelError,
    ChannelKind,
    ContinuousMetrics,
    CoveragePoint,
    ErrorReport,
    OracleRef,
    PhysicsDomain,
    RolloutError,
    SubstitutionPolicy,
    TailBehavior,
    TrustRegion,
)

_HASH = "sha256:" + "ab" * 32


def granular_report(**overrides: Any) -> ErrorReport:
    """A calibrated excavation step-surrogate report (all continuous, with rollout)."""
    kwargs: dict[str, Any] = dict(
        surrogate_name="excavation-gnn",
        surrogate_version="0.1.0",
        domain=PhysicsDomain.GRANULAR_EXCAVATION,
        channels=[
            ChannelError(
                channel="reaction_force_n",
                kind=ChannelKind.CONTINUOUS,
                continuous=ContinuousMetrics(
                    unit="N",
                    rmse=1.5,
                    coverage=[
                        CoveragePoint(nominal=0.5, empirical=0.5),
                        CoveragePoint(nominal=0.9, empirical=0.89),
                    ],
                    tail=TailBehavior(p95_abs_error=3.0, p99_abs_error=5.0, max_abs_error=9.0),
                ),
            ),
        ],
        trust_region=TrustRegion(
            bounds={
                "tool_depth_m": Bound(low=0.0, high=0.5),
                "soil_cohesion_pa": Bound(low=100.0, high=5000.0),
            }
        ),
        validation_dataset_hash=_HASH,
        oracle=OracleRef(
            producer="astro-mine-sim", producer_version="0.1.0", config_hash="sha256:" + "11" * 32
        ),
        substitution_policy=SubstitutionPolicy(recommended_error_budget={"reaction_force_n": 2.0}),
        rollout=RolloutError(horizon_steps=3, rmse_by_horizon=[1.0, 1.3, 1.7]),
    )
    kwargs.update(overrides)
    return ErrorReport(**kwargs)


def illumination_report(**overrides: Any) -> ErrorReport:
    """A calibrated learned-illumination field-surrogate report (categorical + continuous)."""
    kwargs: dict[str, Any] = dict(
        surrogate_name="illumination-surrogate",
        surrogate_version="0.1.0",
        domain=PhysicsDomain.ILLUMINATION_FIELD,
        channels=[
            ChannelError(
                channel="visibility",
                kind=ChannelKind.CATEGORICAL,
                categorical=CategoricalMetrics(
                    classes=["lit", "penumbra", "shadow"],
                    accuracy=0.97,
                    reliability=[CoveragePoint(nominal=0.9, empirical=0.91)],
                ),
            ),
            ChannelError(
                channel="solar_flux_wpm2",
                kind=ChannelKind.CONTINUOUS,
                continuous=ContinuousMetrics(
                    unit="W/m^2",
                    rmse=12.0,
                    coverage=[CoveragePoint(nominal=0.9, empirical=0.9)],
                    tail=TailBehavior(p95_abs_error=25.0, p99_abs_error=40.0, max_abs_error=80.0),
                ),
            ),
        ],
        trust_region=TrustRegion(
            bounds={
                "northing_m": Bound(low=-5000.0, high=5000.0),
                "easting_m": Bound(low=-5000.0, high=5000.0),
                "epoch_s": Bound(low=0.0, high=3.15e7),
                "sun_elevation_deg": Bound(low=-2.0, high=2.0),
            }
        ),
        validation_dataset_hash=_HASH,
        oracle=OracleRef(producer="astro-mine-worlds", producer_version="0.1.0"),
        substitution_policy=SubstitutionPolicy(
            recommended_error_budget={"solar_flux_wpm2": 15.0, "visibility": 0.05}
        ),
    )
    kwargs.update(overrides)
    return ErrorReport(**kwargs)
