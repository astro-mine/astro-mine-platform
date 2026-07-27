"""The frozen DEM fixture + trust region (RM-P1-SURR-02) — numpy only, no torch."""

from __future__ import annotations

import numpy as np

from astro_mine.surrogate.models.dataset import load_dem_dataset
from astro_mine.surrogate.models.trust_region import ExcavationTrustRegion


def test_dataset_loads_with_expected_shape() -> None:
    ds = load_dem_dataset()
    assert ds.states.shape == (ds.n_configs, ds.n_steps + 1, ds.n_particles, 4)
    assert ds.tool_x.shape == (ds.n_configs, ds.n_steps + 1)
    assert ds.params.shape[0] == ds.n_configs
    assert ds.feature_names == ("pos_x", "pos_z", "vel_x", "vel_z")
    assert ds.dt_s > 0.0 and ds.bed_width_m > 0.0


def test_dataset_content_hash_is_stable_and_data_addressed() -> None:
    a, b = load_dem_dataset(), load_dem_dataset()
    assert a.content_hash() == b.content_hash()
    assert a.content_hash().startswith("sha256:")


def test_trust_region_contains_training_configs_and_flags_outside() -> None:
    ds = load_dem_dataset()
    tr = ExcavationTrustRegion.from_configs(ds.param_names, ds.params)
    for config in ds.params:  # every training config is in-domain with a non-negative margin
        assert tr.contains(config)
        assert tr.margin(config) >= 0.0
    # a config with density far above the trained range is out of domain, margin negative
    ood = ds.params[0].copy()
    ood[0] = ds.params[:, 0].max() * 2.0
    assert not tr.contains(ood)
    assert tr.margin(ood) < 0.0


def test_trust_region_projects_to_report_bounds() -> None:
    ds = load_dem_dataset()
    tr = ExcavationTrustRegion.from_configs(ds.param_names, ds.params)
    report_tr = tr.to_report_trust_region()
    assert set(report_tr.bounds) == set(ds.param_names)
    for bound in report_tr.bounds.values():
        assert bound.low <= bound.high


def test_trust_region_margin_is_zero_for_degenerate_box() -> None:
    # a single config gives a zero-width box in every dimension → margin defined as 0.0
    tr = ExcavationTrustRegion.from_configs(("a", "b"), np.array([[1.0, 2.0]]))
    assert tr.margin(np.array([1.0, 2.0])) == 0.0
