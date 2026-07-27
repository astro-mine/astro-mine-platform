"""Offline retrain + gated promotion + full provenance (RM-P1-SURR-03; surrogate.md §5, §10, §11).

A retrain produces a new SemVer version (the prior stays reproducible), admits the model **only**
through the promotion gate, and — on a pass — carries full reproduction provenance (train/validation
hashes, seed, lockfile, hyperparameters, sampling policy). torch CPU is not bit-portable, so the
model-value assertions are tolerance gates (as in ``test_calibration_and_determinism``); the
version/provenance/gate structure is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.surrogate.datagen import (
    SamplingPolicy,
    generate_dataset,
    reference_rollout_oracle,
    write_dataset,
)
from astro_mine.surrogate.eval import PromotionCriteria
from astro_mine.surrogate.models import TrainConfig, build_excavation_surrogate
from astro_mine.surrogate.report import Bound
from astro_mine.surrogate.retrain import BumpKind, retrain_surrogate
from astro_mine.surrogate.retrain.harness import _bump_version

_BOUNDS = {
    "density": Bound(low=1400.0, high=1600.0),
    "friction": Bound(low=0.4, high=0.7),
    "restitution": Bound(low=0.2, high=0.4),
    "tool_speed": Bound(low=0.05, high=0.08),
}
_LOCKFILE = "sha256:" + "1c" * 32
_CONFIG = TrainConfig(hidden=8, message_passing_steps=1, epochs=15, ensemble_size=2)


def _policy() -> SamplingPolicy:
    return SamplingPolicy(parameter_bounds=_BOUNDS, n_initial=6, pool_size=8, n_per_round=2)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(_policy(), reference_rollout_oracle, seed=0)


@pytest.fixture(scope="module")
def promoted(dataset):
    """One gate-passing retrain, reused across the promotion/provenance assertions."""
    return retrain_surrogate(
        dataset=dataset,
        hyperparameters=_CONFIG,
        seed=0,
        prior_version="0.1.0",
        criteria=PromotionCriteria(),
        code_version="deadbeef",
        env_lockfile_hash=_LOCKFILE,
        sampling_policy=_policy(),
        bump=BumpKind.MINOR,
    )


def test_bump_version_minor_and_patch_and_error() -> None:
    assert _bump_version("0.1.0", BumpKind.MINOR) == "0.2.0"
    assert _bump_version("1.4.2", BumpKind.PATCH) == "1.4.3"
    with pytest.raises(ValueError, match="SemVer"):
        _bump_version("1.0", BumpKind.MINOR)


def test_gate_pass_promotes_a_new_version_with_a_bundle(promoted) -> None:
    assert promoted.promoted is True
    assert promoted.new_version == "0.2.0"  # minor > prior 0.1.0 (new data)
    assert promoted.bundle is not None
    # The bundle round-trips and carries the calibrated report.
    assert promoted.bundle.error_report.surrogate_version == "0.2.0"


def test_gate_pass_records_full_reproduction_provenance(promoted, dataset) -> None:
    prov = promoted.provenance
    assert prov.seed == 0
    assert prov.env_lockfile == _LOCKFILE
    # input_hashes = [train_split_hash, validation_dataset_hash] — both recorded.
    assert len(prov.input_hashes) == 2
    assert prov.input_hashes[1] == dataset.content_hash()
    assert "hyperparameters" in prov.source_content_hashes
    assert "sampling_policy" in prov.source_content_hashes
    assert prov.source_content_hashes["sampling_policy"] == _policy().content_hash()
    # provenance.digest is the served bundle's content hash.
    assert prov.digest == promoted.bundle.content_hash()


def test_patch_bump_marks_a_hyperparameter_only_refit(dataset) -> None:
    result = retrain_surrogate(
        dataset=dataset,
        hyperparameters=_CONFIG,
        seed=0,
        prior_version="0.1.0",
        criteria=PromotionCriteria(),
        code_version="deadbeef",
        env_lockfile_hash=_LOCKFILE,
        bump=BumpKind.PATCH,
    )
    assert result.new_version == "0.1.1"
    # No sampling policy supplied → no sampling_policy source hash.
    assert "sampling_policy" not in result.provenance.source_content_hashes


def test_gate_fail_is_not_promoted_and_yields_no_bundle(dataset) -> None:
    result = retrain_surrogate(
        dataset=dataset,
        hyperparameters=_CONFIG,
        seed=0,
        prior_version="0.1.0",
        criteria=PromotionCriteria(error_budget={"pos_x": 1e-12}),  # impossible tolerance
        code_version="deadbeef",
        env_lockfile_hash=_LOCKFILE,
        sampling_policy=_policy(),
    )
    assert result.promoted is False
    assert result.bundle is None
    assert not result.gate.passed
    assert result.gate.reasons  # actionable failure reasons
    # Provenance still records the (rejected) attempt, but with no artifact digest.
    assert result.provenance.seed == 0
    assert result.provenance.digest is None
    assert result.provenance.source_content_hashes["sampling_policy"] == _policy().content_hash()


def test_retrain_reads_an_immutable_dataset_ref(dataset, tmp_path) -> None:
    ref = write_dataset(dataset, tmp_path, name="dem-al", version="0.1.0")
    result = retrain_surrogate(
        dataset=ref,
        hyperparameters=_CONFIG,
        seed=0,
        prior_version="0.1.0",
        criteria=PromotionCriteria(),
        code_version="deadbeef",
        env_lockfile_hash=_LOCKFILE,
    )
    assert result.promoted
    # The ref's recorded train split hash flows into provenance input_hashes.
    assert result.provenance.input_hashes[0] == ref.train_split_hash


def test_prior_version_remains_reproducible(dataset) -> None:
    # Retraining does not touch the prior; the prior is reproducible from the same seed + data.
    a = build_excavation_surrogate(dataset=dataset, config=_CONFIG, seed=0, version="0.1.0")
    b = build_excavation_surrogate(dataset=dataset, config=_CONFIG, seed=0, version="0.1.0")
    ra = [c.continuous.rmse for c in a.error_report.channels]
    rb = [c.continuous.rmse for c in b.error_report.channels]
    assert np.allclose(ra, rb)  # deterministic within the torch-CPU tolerance caveat
