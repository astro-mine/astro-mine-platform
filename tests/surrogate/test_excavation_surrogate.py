"""The learned-DEM excavation surrogate as a SurrogateModel (RM-P1-SURR-02)."""

from __future__ import annotations

import numpy as np

from astro_mine.surrogate import Prediction, SurrogateModel
from astro_mine.surrogate.models.dataset import load_dem_dataset


def _query(ds, config=None) -> dict:
    state = ds.states[0, 0]  # (N, 4) at t0 of the first config
    return {
        "position": state[:, :2],
        "velocity": state[:, 2:],
        "tool_x": np.array([ds.tool_x[0, 0]]),
        "config": ds.params[0] if config is None else config,
    }


def test_surrogate_satisfies_the_surrogate_model_protocol(surrogate) -> None:
    assert isinstance(surrogate, SurrogateModel)


def test_predict_returns_per_particle_fields_with_nonnegative_uncertainty(surrogate) -> None:
    ds = load_dem_dataset()
    pred = surrogate.predict(_query(ds))
    assert isinstance(pred, Prediction)
    n = ds.n_particles
    assert pred.fields["position"].shape == (n, 2)
    assert pred.fields["velocity"].shape == (n, 2)
    assert pred.field_uncertainty["position"].shape == (n, 2)
    assert (pred.field_uncertainty["position"] >= 0.0).all()
    assert pred.in_domain is True


def test_out_of_domain_query_lowers_the_flag_and_inflates_uncertainty(surrogate) -> None:
    ds = load_dem_dataset()
    in_pred = surrogate.predict(_query(ds))
    ood = ds.params[0].copy()
    ood[0] = ds.params[:, 0].max() * 3.0  # density far outside the trained box
    ood_pred = surrogate.predict(_query(ds, ood))
    assert ood_pred.in_domain is False
    assert ood_pred.ood_margin is not None and ood_pred.ood_margin < 0.0
    assert (
        ood_pred.field_uncertainty["position"].mean() > in_pred.field_uncertainty["position"].mean()
    )


def test_error_report_is_carried_and_traceable(surrogate) -> None:
    er = surrogate.error_report
    assert er.domain == "granular_excavation"
    assert er.validation_dataset_hash == load_dem_dataset().content_hash()
    assert er.oracle.producer == "astro-mine-sim"
    assert {c.channel for c in er.channels} == {"pos_x", "pos_z", "vel_x", "vel_z"}
    assert er.rollout is not None and len(er.rollout.rmse_by_horizon) >= 1


def test_autoregressive_rollout_produces_a_trajectory(surrogate) -> None:
    ds = load_dem_dataset()
    traj = surrogate.rollout(
        ds.states[0, 0],
        float(ds.tool_x[0, 0]),
        ds.params[0],
        steps=4,
        tool_speed=float(ds.params[0, 3]),
    )
    assert traj.shape == (5, ds.n_particles, 4)
    assert np.isfinite(traj).all()
