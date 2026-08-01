"""Shared fixtures for the learned-DEM surrogate tests (RM-P1-SURR-02).

Training an ensemble is the expensive step, so the surrogate is built **once per session** with a
small-but-real config (fast on CPU) and reused across the model/calibration/contract tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def train_config():
    from astro_mine.surrogate.models import TrainConfig

    # Small by design, but the cost is set by the *dataset*, not the bed: the DEM fixture is an
    # 81-config grid, so one full-batch epoch runs over 153,090 nodes / 735,902 edges. At the
    # original 200 epochs this single fixture was 346 s of a 516 s suite — two thirds of the job,
    # and every other surrogate test waiting on it.
    #
    # 60 is a *training budget*, not a convergence point: the model is still improving at 200
    # epochs, and no test here asserts the surrogate is accurate. What they assert is that it is
    # well-formed and honestly calibrated — and the conformal layer calibrates whatever model it
    # is given, so coverage barely moves (0.877-0.910 empirical against a 0.8 gate, versus
    # 0.885-0.910 at 200). Budget-over-max and max-over-2*RMSE keep >=1.5x and >=9x margin, and
    # an OOD query still inflates uncertainty ~4x. Ensemble size stays at 2: the conformal layer
    # needs a real epistemic spread, and one member has none.
    return TrainConfig(hidden=24, message_passing_steps=2, epochs=60, ensemble_size=2)


@pytest.fixture(scope="session")
def surrogate(train_config):
    from astro_mine.surrogate.models import build_excavation_surrogate

    return build_excavation_surrogate(config=train_config, seed=0)


@pytest.fixture(scope="session")
def served_bundle(surrogate):
    """The surrogate exported to an ONNX bundle once per session (RM-P1-SURR-04)."""
    from astro_mine.surrogate.serve import export_excavation_surrogate

    return export_excavation_surrogate(surrogate)


@pytest.fixture
def served_query(surrogate):
    """A representative in-domain ParticleFields query drawn from the fixture's first frame."""
    import numpy as np

    ds = surrogate._dataset
    state = ds.states[0, 0]
    return {
        "position": state[:, :2],
        "velocity": state[:, 2:],
        "tool_x": np.array([ds.tool_x[0, 0]]),
        "config": ds.params[0],
    }


@pytest.fixture
def ood_query(served_query, surrogate):
    """The same query with an excavation parameter driven far outside the trust region."""
    config = served_query["config"].copy()
    config[0] = surrogate._trust_region.upper[0] * 5.0
    return {**served_query, "config": config}


@pytest.fixture(scope="session")
def keypair():
    """An ECDSA keypair for signing/verifying a served manifest (the [publish] extra)."""
    from astro_mine.hub.supply_chain import generate_keypair

    return generate_keypair()
