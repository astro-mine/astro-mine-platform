# SPDX-License-Identifier: Apache-2.0
"""Immutable, content-addressed dataset IO — Zarr arrays + Parquet tabular (RM-P1-SURR-03).

Datasets are **immutable, content-addressed, versioned** (surrogate.md §5; conventions.md §5): a
surrogate references the exact dataset hashes it trained and validated against, and a retrain
produces a new version without ever overwriting the prior. This module is that store:

- :func:`write_dataset` persists a :class:`~astro_mine.surrogate.models.dataset.DemDataset` under an
  immutable ``name/version`` directory — the particle/field N-D arrays (``states``/``tool_x``) as
  **Zarr** and the tabular config features (``params``) as **Parquet**, per conventions.md §5 — and
  returns a :class:`DatasetRef` pinning the whole-dataset content hash plus the train/validation
  split hashes. It **refuses to overwrite** an existing ``name:version`` (the immutability rule).
- :func:`read_dataset` reads a ref back and **verifies** the content hash fail-closed.

The content hash reuses :meth:`DemDataset.content_hash` — taken over the canonical big-endian array
bytes, so it is **stable regardless of Zarr chunking or Parquet encoding**. numpy + zarr + pyarrow
only; no torch, no Sim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from astro_mine.surrogate.models.dataset import DemDataset

__all__ = ["DatasetRef", "read_dataset", "split_dataset", "write_dataset"]

FloatArray = npt.NDArray[np.float64]

_STATES_ENTRY = "states.zarr"
_TOOL_X_ENTRY = "tool_x.zarr"
_PARAMS_ENTRY = "params.parquet"
_META_ENTRY = "meta.json"
#: The default held-out validation fraction (by config) a store records in its split hashes.
_DEFAULT_VAL_FRACTION = 0.25


@dataclass(frozen=True)
class DatasetRef:
    """A content-addressed pointer to a stored dataset — the hashes a surrogate records.

    ``content_hash`` is the whole dataset's address; ``train_split_hash``/``val_split_hash`` pin the
    config-level train/validation partition (:func:`split_dataset`) a retrain records in its
    provenance ``input_hashes``. ``path`` is where the immutable ``name/version`` store lives so
    :func:`read_dataset` can re-open it; it is *not* part of the identity — the hashes are.
    """

    name: str
    version: str
    content_hash: str
    train_split_hash: str
    val_split_hash: str
    path: Path


def _subset(dataset: DemDataset, idx: npt.NDArray[np.intp]) -> DemDataset:
    """A :class:`DemDataset` over the config subset ``idx`` (same rig metadata)."""
    return DemDataset(
        states=dataset.states[idx],
        tool_x=dataset.tool_x[idx],
        params=dataset.params[idx],
        dt_s=dataset.dt_s,
        bed_width_m=dataset.bed_width_m,
        tool_height_m=dataset.tool_height_m,
        feature_names=dataset.feature_names,
        param_names=dataset.param_names,
    )


def split_dataset(
    dataset: DemDataset, *, val_fraction: float = _DEFAULT_VAL_FRACTION, seed: int = 0
) -> tuple[DemDataset, DemDataset]:
    """Partition a dataset by config into ``(train, validation)`` deterministically.

    A seeded permutation holds out ``round(val_fraction * n_configs)`` configs (at least one) for
    validation and trains on the rest. Deterministic in ``seed``, so the recorded split hashes are
    reproducible. Requires at least two configs so both partitions are non-empty.
    """
    n = dataset.n_configs
    if n < 2:
        raise ValueError(f"cannot split a dataset of {n} config(s) into train + validation")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = min(n - 1, max(1, round(val_fraction * n)))
    val_idx = np.sort(perm[:n_val]).astype(np.intp)
    train_idx = np.sort(perm[n_val:]).astype(np.intp)
    return _subset(dataset, train_idx), _subset(dataset, val_idx)


def write_dataset(
    dataset: DemDataset,
    path: Path | str,
    *,
    name: str,
    version: str,
    val_fraction: float = _DEFAULT_VAL_FRACTION,
    split_seed: int = 0,
) -> DatasetRef:
    """Persist ``dataset`` immutably under ``path/name/version`` and return its :class:`DatasetRef`.

    Writes ``states``/``tool_x`` as Zarr and ``params`` as Parquet (conventions.md §5) plus a
    ``meta.json`` of the rig scalars and recorded hashes. **Refuses to overwrite** an existing
    ``name:version`` (raises :class:`FileExistsError`) — a retrain must publish a new version, never
    mutate a prior (surrogate.md §5).
    """
    target = Path(path) / name / version
    if target.exists():
        raise FileExistsError(
            f"dataset {name}:{version} already exists at {target}; datasets are immutable — "
            "publish a new version rather than overwrite (surrogate.md §5)"
        )
    target.mkdir(parents=True)

    content_hash = dataset.content_hash()
    train, validation = split_dataset(dataset, val_fraction=val_fraction, seed=split_seed)

    zarr.save_array(str(target / _STATES_ENTRY), np.ascontiguousarray(dataset.states))
    zarr.save_array(str(target / _TOOL_X_ENTRY), np.ascontiguousarray(dataset.tool_x))
    table = pa.table({name_: dataset.params[:, j] for j, name_ in enumerate(dataset.param_names)})
    pq.write_table(table, str(target / _PARAMS_ENTRY))

    meta = {
        "name": name,
        "version": version,
        "dt_s": dataset.dt_s,
        "bed_width_m": dataset.bed_width_m,
        "tool_height_m": dataset.tool_height_m,
        "feature_names": list(dataset.feature_names),
        "param_names": list(dataset.param_names),
        "content_hash": content_hash,
        "train_split_hash": train.content_hash(),
        "val_split_hash": validation.content_hash(),
    }
    (target / _META_ENTRY).write_text(json.dumps(meta, sort_keys=True, indent=2))

    return DatasetRef(
        name=name,
        version=version,
        content_hash=content_hash,
        train_split_hash=train.content_hash(),
        val_split_hash=validation.content_hash(),
        path=target,
    )


def read_dataset(ref: DatasetRef) -> DemDataset:
    """Read a :class:`DatasetRef` back into a :class:`DemDataset`, verifying its content hash.

    Fail-closed: reconstructs the arrays from Zarr/Parquet, rebuilds the dataset, and raises
    :class:`ValueError` if its content hash does not equal ``ref.content_hash`` — a tampered or
    truncated store never yields a dataset (core.md principle 7).
    """
    target = ref.path
    meta = json.loads((target / _META_ENTRY).read_text())
    param_names = tuple(meta["param_names"])

    states = np.asarray(zarr.load(str(target / _STATES_ENTRY)), dtype=np.float64)
    tool_x = np.asarray(zarr.load(str(target / _TOOL_X_ENTRY)), dtype=np.float64)
    table = pq.read_table(str(target / _PARAMS_ENTRY))
    params = np.column_stack([table[name].to_numpy() for name in param_names]).astype(np.float64)

    dataset = DemDataset(
        states=states,
        tool_x=tool_x,
        params=params,
        dt_s=float(meta["dt_s"]),
        bed_width_m=float(meta["bed_width_m"]),
        tool_height_m=float(meta["tool_height_m"]),
        feature_names=tuple(meta["feature_names"]),
        param_names=param_names,
    )
    if dataset.content_hash() != ref.content_hash:
        raise ValueError(
            f"dataset at {target} hashes to {dataset.content_hash()} but the ref pins "
            f"{ref.content_hash} — the store is tampered or truncated"
        )
    return dataset
