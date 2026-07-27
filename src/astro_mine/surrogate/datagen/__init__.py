"""``datagen`` — the offline build loop's high-fidelity sampling (RM-P1-SURR-03; surrogate.md §3).

Query the oracle, design the experiments, label, and archive: a declarative
:class:`SamplingPolicy` (Sobol/LHS/grid design + active-learning acquisition), the Core-typed
:class:`RolloutOracle` seam Sim data arrives through (a numpy-only reference proxy always available;
the Sim-backed oracle behind the ``[datagen]`` extra), :func:`generate_dataset` /
:func:`active_learning_round`, and the immutable content-addressed dataset store.

Importing this subpackage requires the ``[datasets]`` extra (``zarr``/``pyarrow``/``scipy``) —
mirroring how :mod:`astro_mine.surrogate.serve` needs ``[serve]``; the base package import never
pulls the datagen stack. It imports **only** Core + numpy + scipy, never ``astro_mine.sim`` (the
narrow waist): Sim rollouts arrive through the :class:`RolloutOracle` seam, not an import.
"""

from __future__ import annotations

from astro_mine.surrogate.datagen.design import (
    design_points,
    grid_design,
    lhs_design,
    sobol_design,
)
from astro_mine.surrogate.datagen.generate import (
    active_learning_round,
    generate_dataset,
    score_uncertainty,
)
from astro_mine.surrogate.datagen.oracle import (
    RolloutOracle,
    RolloutSample,
    reference_rollout_oracle,
)
from astro_mine.surrogate.datagen.policy import (
    AcquisitionKind,
    DesignKind,
    SamplingPolicy,
)
from astro_mine.surrogate.datagen.store import (
    DatasetRef,
    read_dataset,
    split_dataset,
    write_dataset,
)

__all__ = [
    "AcquisitionKind",
    "DatasetRef",
    "DesignKind",
    "RolloutOracle",
    "RolloutSample",
    "SamplingPolicy",
    "active_learning_round",
    "design_points",
    "generate_dataset",
    "grid_design",
    "lhs_design",
    "read_dataset",
    "reference_rollout_oracle",
    "score_uncertainty",
    "sobol_design",
    "split_dataset",
    "write_dataset",
]
