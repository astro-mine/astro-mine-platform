"""Shared fixtures for the learned-DEM surrogate tests (RM-P1-SURR-02).

Training an ensemble is the expensive step, so the surrogate is built **once per session** with a
small-but-real config (fast on CPU) and reused across the model/calibration/contract tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def train_config():
    from astro_mine.surrogate.models import TrainConfig

    # Small by design: the reference bed is tiny and CI trains on CPU (~20 s, full-batch GD).
    # Enough ensemble members + epochs to give the conformal layer a real, calibrated spread.
    return TrainConfig(hidden=24, message_passing_steps=2, epochs=200, ensemble_size=2)


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
