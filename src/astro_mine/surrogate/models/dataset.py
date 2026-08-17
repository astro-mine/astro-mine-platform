# SPDX-License-Identifier: Apache-2.0
"""The frozen DEM training fixture the learned-DEM surrogate learns from (RM-P1-SURR-02).

Loads the content-addressed particle-rollout dataset produced from the high-fidelity SIM-06 DEM
engine by ``scripts/gen_dem_dataset.py`` (the ``[datagen]`` extra). The surrogate trains and
validates against *this frozen artifact* — so ``astro_mine.surrogate`` imports only Core + numpy
+ torch, never Sim (the narrow waist; conventions.md §1.1). numpy only; no torch.

The content hash is taken over the **array data** (not the timestamped ``.npz`` zip bytes), so it
is the stable ``validation_dataset_hash`` the ErrorReport carries — reproducible from the committed
fixture regardless of how the zip was written.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files

import numpy as np
import numpy.typing as npt

from astro_mine.core.hashing import content_hash

__all__ = ["DemDataset", "load_dem_dataset"]

FloatArray = npt.NDArray[np.float64]

_DATA_PACKAGE = "astro_mine.surrogate.data"
_DATASET_FILE = "dem_excavation_v1.npz"


@dataclass(frozen=True)
class DemDataset:
    """A grid of DEM excavation rollouts: particle trajectories under varied excavation params.

    ``states`` is ``(C, T+1, N, 4)`` — for each of ``C`` configs, ``T+1`` timesteps of ``N``
    particles' ``(pos_x, pos_z, vel_x, vel_z)``; ``tool_x`` is the blade position ``(C, T+1)``;
    ``params`` is ``(C, P)`` the excavation config (density, friction, restitution, tool_speed).
    A ``(state_t, tool_x_t, config) -> state_{t+1}`` transition is one training example.
    """

    states: FloatArray
    tool_x: FloatArray
    params: FloatArray
    dt_s: float
    bed_width_m: float
    tool_height_m: float
    feature_names: tuple[str, ...]
    param_names: tuple[str, ...]
    #: The ``sha256:`` content address of the :class:`~astro_mine.surrogate.datagen.SamplingPolicy`
    #: this fixture was swept under — the declarative box whose corners *are* the trust region a
    #: surrogate trained here will declare (``ExcavationTrustRegion.from_configs``). Carried so the
    #: published manifest can pin the policy by hash and a reader of the artifact can trace the
    #: surrogate's domain back to the declaration that set it, rather than inferring it from the
    #: sampled configs and hoping (surrogate#17). ``None`` for a fixture generated before the sweep
    #: was policy-driven.
    sampling_policy_hash: str | None = None

    @property
    def n_configs(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.states.shape[1]) - 1

    @property
    def n_particles(self) -> int:
        return int(self.states.shape[2])

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address over the canonical array data.

        Over the concatenated big-endian float64 bytes of ``states``/``tool_x``/``params`` in a
        fixed order — stable across machines and independent of the ``.npz`` container's
        (non-reproducible) zip metadata, so it is a portable ``validation_dataset_hash``.
        """
        blob = b"".join(
            np.ascontiguousarray(a, dtype=">f8").tobytes()
            for a in (self.states, self.tool_x, self.params)
        )
        return content_hash(blob)


def load_dem_dataset() -> DemDataset:
    """Load the committed DEM rollout fixture from the package data."""
    resource = files(_DATA_PACKAGE).joinpath(_DATASET_FILE)
    with as_file(resource) as path, np.load(path) as npz:
        return DemDataset(
            states=npz["states"].astype(np.float64),
            tool_x=npz["tool_x"].astype(np.float64),
            params=npz["params"].astype(np.float64),
            dt_s=float(npz["dt_s"][0]),
            bed_width_m=float(npz["bed_width_m"][0]),
            tool_height_m=float(npz["tool_height_m"][0]),
            feature_names=tuple(str(n) for n in npz["feature_names"].tolist()),
            param_names=tuple(str(n) for n in npz["param_names"].tolist()),
            sampling_policy_hash=(
                str(npz["sampling_policy_hash"][0]) if "sampling_policy_hash" in npz else None
            ),
        )
