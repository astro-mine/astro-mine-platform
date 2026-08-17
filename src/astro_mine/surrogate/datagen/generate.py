# SPDX-License-Identifier: Apache-2.0
"""Label the design against the oracle into a :class:`DemDataset` + active learning (RM-P1-SURR-03).

The two halves of ``datagen``'s **build** loop (surrogate.md §3, §8):

- :func:`generate_dataset` — run the policy's space-filling
  :mod:`~astro_mine.surrogate.datagen.design` through the
  :class:`~astro_mine.surrogate.datagen.oracle.RolloutOracle` and stack the labeled rollouts into a
  :class:`~astro_mine.surrogate.models.dataset.DemDataset`.
- :func:`active_learning_round` — score a candidate pool by the current surrogate's residual/field
  **uncertainty**, label the top ``n_per_round`` (or a random baseline), and append them — "sample
  where surrogate residual uncertainty is highest, not uniformly" (§8).

numpy + scipy only — never torch, never Sim. The surrogate is taken through the numpy-free
:class:`~astro_mine.surrogate.model.SurrogateModel` seam, so this stays in the datagen layer.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import qmc

from astro_mine.surrogate.datagen.design import _box, design_points
from astro_mine.surrogate.datagen.oracle import (
    REFERENCE_BED_WIDTH_M,
    REFERENCE_DT_S,
    REFERENCE_FEATURE_NAMES,
    REFERENCE_TOOL_HEIGHT_M,
    RolloutOracle,
    RolloutSample,
)
from astro_mine.surrogate.datagen.policy import AcquisitionKind, SamplingPolicy
from astro_mine.surrogate.model import SurrogateModel
from astro_mine.surrogate.models.dataset import DemDataset

__all__ = ["active_learning_round", "generate_dataset"]

FloatArray = npt.NDArray[np.float64]


def _label(oracle: RolloutOracle, configs: FloatArray, *, seed: int) -> list[RolloutSample]:
    """Label each config row through the oracle with a per-config derived seed (deterministic)."""
    return [oracle(configs[i], seed + i) for i in range(configs.shape[0])]


def _stack(
    samples: list[RolloutSample],
    *,
    dt_s: float,
    bed_width_m: float,
    tool_height_m: float,
    feature_names: tuple[str, ...],
    param_names: tuple[str, ...],
) -> DemDataset:
    """Stack per-config rollouts into a :class:`DemDataset` carrying the rig metadata."""
    return DemDataset(
        states=np.stack([s.states for s in samples]),
        tool_x=np.stack([s.tool_x for s in samples]),
        params=np.stack([s.params for s in samples]),
        dt_s=dt_s,
        bed_width_m=bed_width_m,
        tool_height_m=tool_height_m,
        feature_names=feature_names,
        param_names=param_names,
    )


def generate_dataset(
    policy: SamplingPolicy,
    oracle: RolloutOracle,
    *,
    seed: int = 0,
    dt_s: float = REFERENCE_DT_S,
    bed_width_m: float = REFERENCE_BED_WIDTH_M,
    tool_height_m: float = REFERENCE_TOOL_HEIGHT_M,
    feature_names: tuple[str, ...] = REFERENCE_FEATURE_NAMES,
) -> DemDataset:
    """Run ``policy``'s space-filling design through ``oracle`` into a labeled :class:`DemDataset`.

    The design (Sobol/LHS/grid, deterministic in ``policy.seed``) fixes the configs; ``oracle``
    labels each into a particle rollout with a per-config seed derived from ``seed``. The rig
    metadata (``dt_s``/``bed_width_m``/``tool_height_m``) defaults to the reference proxy's rig; a
    Sim-backed oracle passes its own. ``param_names`` is ``policy.param_names`` — the config column
    order the design produced.
    """
    configs = design_points(policy)
    samples = _label(oracle, configs, seed=seed)
    return _stack(
        samples,
        dt_s=dt_s,
        bed_width_m=bed_width_m,
        tool_height_m=tool_height_m,
        feature_names=feature_names,
        param_names=policy.param_names,
    )


def _candidate_pool(policy: SamplingPolicy, seed: int) -> FloatArray:
    """A Latin-hypercube pool of ``pool_size`` candidate configs over the box (deterministic)."""
    lower, upper = _box(policy)
    sampler = qmc.LatinHypercube(d=len(lower), seed=seed)
    unit = sampler.random(policy.pool_size)
    return np.asarray(lower + unit * (upper - lower), dtype=np.float64)


def score_uncertainty(surrogate: SurrogateModel, base: DemDataset, config: FloatArray) -> float:
    """The surrogate's mean calibrated field uncertainty at ``config`` on a representative state.

    Queries the surrogate at ``base``'s first frame with the candidate ``config`` and averages the
    per-particle ``field_uncertainty`` — the acquisition signal (surrogate.md §8: sample where the
    surrogate's residual uncertainty is highest). No oracle label is needed to score a candidate,
    which is the point: labeling is the expensive step the acquisition rations.
    """
    frame = base.states[0, 0]
    query = {
        "position": frame[:, :2],
        "velocity": frame[:, 2:],
        "tool_x": np.array([base.tool_x[0, 0]], dtype=np.float64),
        "config": np.asarray(config, dtype=np.float64),
    }
    prediction = surrogate.predict(query)
    means = [float(np.mean(a)) for a in prediction.field_uncertainty.values() if np.size(a)]
    return float(np.mean(means)) if means else 0.0


def active_learning_round(
    policy: SamplingPolicy,
    oracle: RolloutOracle,
    surrogate: SurrogateModel,
    existing: DemDataset,
    *,
    seed: int = 0,
) -> DemDataset:
    """Score a candidate pool, label the selected ``n_per_round`` configs, and append them.

    ``MAX_UNCERTAINTY`` ranks the pool by :func:`score_uncertainty` and takes the top
    ``n_per_round``; ``RANDOM`` draws ``n_per_round`` uniformly (the baseline the acquisition
    beats). The chosen configs are labeled through ``oracle`` and appended to ``existing`` — a
    larger dataset whose content hash therefore changes. Deterministic given ``seed`` + ``policy``.
    """
    pool = _candidate_pool(policy, seed)
    if policy.acquisition is AcquisitionKind.MAX_UNCERTAINTY:
        scores = np.array([score_uncertainty(surrogate, existing, cfg) for cfg in pool])
        chosen = np.argsort(scores)[::-1][: policy.n_per_round]
    else:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(pool.shape[0], size=policy.n_per_round, replace=False)
    chosen_configs = pool[np.sort(chosen)]

    samples = _label(oracle, chosen_configs, seed=seed + pool.shape[0])
    return DemDataset(
        states=np.concatenate([existing.states, np.stack([s.states for s in samples])]),
        tool_x=np.concatenate([existing.tool_x, np.stack([s.tool_x for s in samples])]),
        params=np.concatenate([existing.params, np.stack([s.params for s in samples])]),
        dt_s=existing.dt_s,
        bed_width_m=existing.bed_width_m,
        tool_height_m=existing.tool_height_m,
        feature_names=existing.feature_names,
        param_names=existing.param_names,
    )
