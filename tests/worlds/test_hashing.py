"""A content hash covers content, not the toolchain that produced it (astro-mine-worlds#46).

The regression these guard against is subtle because it never produced a *wrong* number — it
produced an unreproducible one. ``astro-mine-worlds`` is versioned by hatch-vcs, so its version
tracks git commit distance; folding it into the hashed metadata meant a **bit-identical** world
rebuilt one commit later minted a different ``terrain_hash`` / ``illumination_hash`` /
``world_hash``. Content-addressing was really commit-addressing, and no one could rebuild a
published world from its recorded recipe and land on its digest (``LUNAR-TR-004``;
``conventions.md §11``).

Nothing is lost by excluding it: every Worlds hash covers its own arrays, and the bundle folds each
field store's ``store_hash`` into ``world_hash`` — so a toolchain that writes different bytes still
moves the hash, through the bytes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from astro_mine.worlds._hashing import PROVENANCE_KEYS, canonical_meta_bytes, content_meta
from astro_mine.worlds.illumination._horizon import horizon_hash, psr_mask_hash
from astro_mine.worlds.illumination._topocentric import topocentric_horizon_hash
from astro_mine.worlds.regolith._fields import regolith_hash
from astro_mine.worlds.terrain._layers import terrain_hash
from astro_mine.worlds.thermal._solver import diurnal_hash


def _meta(version: str) -> dict[str, Any]:
    """A manifest whose only difference between calls is the recorded toolchain."""
    return {
        "schema": "astro-mine-worlds/world/v0.1",
        "grid": {"width": 4, "height": 3},
        "params": {"n_azimuth": 8},
        "toolchain": {"astro_mine_worlds": version, "numpy": np.__version__},
    }


def test_content_meta_strips_only_the_provenance_keys() -> None:
    stripped = content_meta(_meta("0.1.dev18"))

    assert "toolchain" not in stripped
    assert stripped == {
        "schema": "astro-mine-worlds/world/v0.1",
        "grid": {"width": 4, "height": 3},
        "params": {"n_azimuth": 8},
    }
    assert frozenset({"toolchain"}) == PROVENANCE_KEYS


def test_canonical_meta_bytes_ignores_the_toolchain() -> None:
    assert canonical_meta_bytes(_meta("0.1.dev18")) == canonical_meta_bytes(_meta("0.1.dev24"))


@pytest.mark.parametrize(
    ("hash_fn", "payload"),
    [
        pytest.param(
            terrain_hash,
            {"elevation": np.arange(12, dtype=np.float32).reshape(3, 4)},
            id="terrain",
        ),
        pytest.param(
            regolith_hash,
            {"bulk_density": np.full((3, 4), 1500.0, dtype=np.float32)},
            id="regolith",
        ),
    ],
)
def test_layer_hashes_are_stable_across_a_toolchain_bump(
    hash_fn: Any, payload: dict[str, Any]
) -> None:
    """The exact failure that made every world digest churn per commit."""
    assert hash_fn(payload, _meta("0.1.dev18")) == hash_fn(payload, _meta("0.1.dev24"))


@pytest.mark.parametrize(
    "hash_fn", [horizon_hash, topocentric_horizon_hash], ids=["horizon", "topocentric"]
)
def test_horizon_hashes_are_stable_across_a_toolchain_bump(hash_fn: Any) -> None:
    horizon = np.linspace(0.0, 10.0, 96, dtype=np.float32).reshape(3, 4, 8)

    assert hash_fn(horizon, _meta("0.1.dev18")) == hash_fn(horizon, _meta("0.1.dev24"))


def test_psr_and_diurnal_hashes_are_stable_across_a_toolchain_bump() -> None:
    mask = np.zeros((3, 4), dtype=np.bool_)
    mask[1, 1] = True
    void = np.zeros((3, 4), dtype=np.bool_)
    temperatures = np.linspace(90.0, 390.0, 16, dtype=np.float64)

    assert psr_mask_hash(mask, void, _meta("0.1.dev18")) == psr_mask_hash(
        mask, void, _meta("0.1.dev24")
    )
    assert diurnal_hash(temperatures, _meta("0.1.dev18")) == diurnal_hash(
        temperatures, _meta("0.1.dev24")
    )


def test_the_data_still_moves_every_hash() -> None:
    """The other half of the contract: excluding the toolchain must not blind a hash to content."""
    meta = _meta("0.1.dev18")
    layers = {"elevation": np.arange(12, dtype=np.float32).reshape(3, 4)}
    nudged = {"elevation": layers["elevation"] + np.float32(1e-3)}

    assert terrain_hash(layers, meta) != terrain_hash(nudged, meta)

    # ...and so does a parameter that actually determines the content.
    other_params = {**meta, "params": {"n_azimuth": 16}}
    assert terrain_hash(layers, meta) != terrain_hash(layers, other_params)
