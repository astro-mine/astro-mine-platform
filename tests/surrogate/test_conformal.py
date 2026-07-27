"""Split-conformal calibration (RM-P1-SURR-02) — numpy only, no torch.

Conformal must deliver its finite-sample marginal-coverage guarantee: calibrate on one draw,
and a fresh draw from the same distribution is covered at ~the nominal rate. An over-confident
(too-small) std must be corrected by a larger multiplier.
"""

from __future__ import annotations

import numpy as np

from astro_mine.surrogate.models.conformal import calibrate_conformal


def test_conformal_achieves_marginal_coverage_on_fresh_data() -> None:
    rng = np.random.default_rng(0)
    # one channel: residual ~ |N(0, sigma)|, reported std is a constant under-estimate.
    sigma = 1.0
    cal_resid = np.abs(rng.normal(0.0, sigma, size=(2000, 1)))
    cal_std = np.full((2000, 1), 0.5)  # deliberately over-confident
    cal = calibrate_conformal(cal_resid, cal_std, ("x",), nominal_coverage=0.9)

    test_resid = np.abs(rng.normal(0.0, sigma, size=(4000, 1)))
    test_std = np.full((4000, 1), 0.5)
    half = cal.half_widths(test_std)
    coverage = float(np.mean(test_resid <= half))
    assert coverage >= 0.88  # ~0.9 nominal, finite-sample slack
    assert cal.quantiles[0] > 1.0  # corrected the under-confident std upward


def test_conformal_is_per_channel() -> None:
    rng = np.random.default_rng(1)
    resid = np.abs(rng.normal(0.0, [1.0, 3.0], size=(1500, 2)))
    std = np.ones((1500, 2))
    cal = calibrate_conformal(resid, std, ("a", "b"), nominal_coverage=0.9)
    assert cal.channel_names == ("a", "b")
    assert cal.quantiles[1] > cal.quantiles[0]  # wider channel gets a larger multiplier
