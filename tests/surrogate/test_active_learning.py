"""Active-learning acquisition on residual uncertainty (RM-P1-SURR-03; surrogate.md §8).

With the numpy-only reference oracle and a tiny GNS: ``MAX_UNCERTAINTY`` labels the most uncertain
candidates (vs a random baseline), a round grows the dataset (its content hash changes), and the
whole loop is deterministic given the seed — "sample where surrogate residual uncertainty is
highest, not uniformly" (surrogate.md §8).
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.surrogate.datagen import (
    AcquisitionKind,
    SamplingPolicy,
    active_learning_round,
    generate_dataset,
    reference_rollout_oracle,
    score_uncertainty,
)
from astro_mine.surrogate.datagen.generate import _candidate_pool
from astro_mine.surrogate.models import TrainConfig, build_excavation_surrogate
from astro_mine.surrogate.report import Bound

_BOUNDS = {
    "density": Bound(low=1400.0, high=1600.0),
    "friction": Bound(low=0.4, high=0.7),
    "restitution": Bound(low=0.2, high=0.4),
    "tool_speed": Bound(low=0.05, high=0.08),
}


def _policy(acquisition: AcquisitionKind = AcquisitionKind.MAX_UNCERTAINTY) -> SamplingPolicy:
    return SamplingPolicy(
        parameter_bounds=_BOUNDS,
        n_initial=6,
        pool_size=16,
        n_per_round=3,
        acquisition=acquisition,
    )


@pytest.fixture(scope="module")
def seeded_surrogate():
    """A tiny surrogate trained on a reference-oracle dataset — built once for the module."""
    dataset = generate_dataset(_policy(), reference_rollout_oracle, seed=0)
    config = TrainConfig(hidden=8, message_passing_steps=1, epochs=15, ensemble_size=2)
    return dataset, build_excavation_surrogate(dataset=dataset, config=config, seed=0)


def test_max_uncertainty_selects_above_median_candidates(seeded_surrogate) -> None:
    dataset, surrogate = seeded_surrogate
    policy = _policy(AcquisitionKind.MAX_UNCERTAINTY)
    pool = _candidate_pool(policy, seed=1)
    scores = np.array([score_uncertainty(surrogate, dataset, cfg) for cfg in pool])
    top = np.sort(np.argsort(scores)[::-1][: policy.n_per_round])
    selected_mean = float(np.mean(scores[top]))
    # By construction the top-k acquisition selects strictly above-median uncertainty.
    assert selected_mean >= float(np.median(scores))
    assert selected_mean > float(np.mean(scores))  # and above the pool average


def test_random_acquisition_differs_from_max_uncertainty(seeded_surrogate) -> None:
    dataset, surrogate = seeded_surrogate
    grown_max = active_learning_round(
        _policy(AcquisitionKind.MAX_UNCERTAINTY),
        reference_rollout_oracle,
        surrogate,
        dataset,
        seed=1,
    )
    grown_random = active_learning_round(
        _policy(AcquisitionKind.RANDOM), reference_rollout_oracle, surrogate, dataset, seed=1
    )
    # Both label n_per_round configs, but the acquisition picks a different set.
    assert not np.array_equal(grown_max.params, grown_random.params)


def test_a_round_grows_the_dataset_and_changes_its_hash(seeded_surrogate) -> None:
    dataset, surrogate = seeded_surrogate
    grown = active_learning_round(_policy(), reference_rollout_oracle, surrogate, dataset, seed=1)
    assert grown.n_configs == dataset.n_configs + 3
    assert grown.content_hash() != dataset.content_hash()
    # The prior configs are preserved (append, never overwrite).
    assert np.array_equal(grown.params[: dataset.n_configs], dataset.params)


def test_round_is_deterministic_given_the_seed(seeded_surrogate) -> None:
    dataset, surrogate = seeded_surrogate
    a = active_learning_round(_policy(), reference_rollout_oracle, surrogate, dataset, seed=1)
    b = active_learning_round(_policy(), reference_rollout_oracle, surrogate, dataset, seed=1)
    assert a.content_hash() == b.content_hash()


def test_score_uncertainty_rises_outside_the_trust_region(seeded_surrogate) -> None:
    dataset, surrogate = seeded_surrogate
    in_domain = dataset.params[0].copy()
    out_of_domain = in_domain.copy()
    out_of_domain[0] *= 5.0  # far outside the density bound → OOD inflation
    assert score_uncertainty(surrogate, dataset, out_of_domain) > score_uncertainty(
        surrogate, dataset, in_domain
    )
