"""The served graph must be O(N.k), and identical to the sparse GNS it stands in for (#24).

The first version of this export reduced the radius graph to a **dense masked adjacency**: message
passing over ``(N, N)`` tensors, chosen so ONNX would see no data-dependent edge count. The numerics
were right and the reasoning was sound, but the cost was not — it ran the edge encoder and every
message MLP across *all* ``N^2`` pairs and then masked ~99% of them away. That made the served tier
``O(N^2)``: the same asymptotics as the DEM solver it exists to replace, with a far worse constant
per pair. The measured speedup sat at ~2x and **did not improve with bed size**, which is the one
thing a surrogate has to do.

The fix gathers each receiver's ``k`` nearest senders instead. ``(N, k)`` is still a static shape —
the property the dense form was chosen for — but the network never touches the pairs it was going to
discard. These tests pin both halves: the numbers must not move, and the cost must.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astro_mine.surrogate.models.gns import build_edges, edge_features
from astro_mine.surrogate.models.train import _CUTOFF_RADII, _PARTICLE_RADIUS_M
from astro_mine.surrogate.serve.export import _NEIGHBOUR_K, _gathered_gns, _neighbourhood

CUTOFF = _CUTOFF_RADII * _PARTICLE_RADIUS_M


def _bed(surrogate, config: int = 0, frame: int = 0) -> np.ndarray:
    return surrogate._dataset.states[config, frame]


def test_the_gathered_neighbourhood_is_the_radius_graph(surrogate) -> None:
    """Same edges as `build_edges` — no neighbour gained, none lost."""
    for frame in (0, 5, 10):
        pos = _bed(surrogate, frame=frame)[:, :2]
        sender, _, mask = _neighbourhood(
            torch.tensor(pos, dtype=torch.float64), CUTOFF, _NEIGHBOUR_K
        )

        # what the gathered form says the graph is: {(sender, receiver)} over the unmasked entries
        gathered = {
            (int(sender[j, m]), j)
            for j in range(pos.shape[0])
            for m in range(_NEIGHBOUR_K)
            if mask[j, m] > 0
        }
        # what the sparse trainer says it is
        edge_index = build_edges(pos, CUTOFF)
        sparse = {(int(s), int(d)) for s, d in zip(edge_index[0], edge_index[1], strict=True)}

        assert gathered == sparse, (
            f"frame {frame}: the gathered graph and the radius graph disagree on "
            f"{len(gathered ^ sparse)} edges"
        )


def test_the_gathered_gns_reproduces_the_sparse_gns_exactly(surrogate) -> None:
    """The whole point: a cheaper graph, not a different one.

    Runs one trained GNS member through both formulations on the same bed and demands the
    accelerations agree to float precision. If they ever diverge, the served tier is quietly
    predicting different physics from the model that was trained and calibrated.
    """
    gns = surrogate._models[0]
    ds = surrogate._dataset
    normalizer = surrogate._normalizer

    for frame in (0, 7):
        state = _bed(surrogate, frame=frame)
        pos = state[:, :2]
        from astro_mine.surrogate.models.gns import node_features

        config_norm = (ds.params[0][:2] - normalizer.config_mean) / normalizer.config_std
        node = node_features(state, float(ds.tool_x[0, frame]), config_norm, ds.bed_width_m)
        node = (node - normalizer.node_mean) / normalizer.node_std

        # --- sparse (the trained path)
        edge_index = build_edges(pos, CUTOFF)
        efeat = edge_features(pos, edge_index)
        with torch.no_grad():
            sparse_out = gns(
                torch.tensor(node, dtype=torch.float32),
                torch.tensor(edge_index, dtype=torch.int64),
                torch.tensor(efeat, dtype=torch.float32),
            ).numpy()

        # --- gathered (the served path)
        position = torch.tensor(pos, dtype=torch.float32)
        sender, gathered_ef, mask = _neighbourhood(position, CUTOFF, _NEIGHBOUR_K)
        with torch.no_grad():
            gathered_out = _gathered_gns(
                gns, torch.tensor(node, dtype=torch.float32), gathered_ef, mask, sender
            ).numpy()

        np.testing.assert_allclose(gathered_out, sparse_out, rtol=1e-5, atol=1e-6)


def test_k_has_real_headroom_over_the_beds_the_tier_will_see(surrogate) -> None:
    """`k` bounds a *packing density*, not a bed size — so it must not be a close-run thing.

    TopK keeps the nearest k, so an over-full neighbourhood would silently drop the most distant
    senders: graceful, but wrong, and wrong in a way nothing would report. Hexagonal packing admits
    6 neighbours; soft-sphere overlap buys a little more.
    """
    ds = surrogate._dataset
    worst = 0
    for config in range(ds.states.shape[0]):
        for frame in range(ds.states.shape[1]):
            pos = ds.states[config, frame, :, :2]
            dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dist, np.inf)
            worst = max(worst, int((dist < CUTOFF).sum(axis=1).max()))

    assert worst <= _NEIGHBOUR_K, f"a particle has {worst} neighbours but k={_NEIGHBOUR_K}"
    assert worst * 2 <= _NEIGHBOUR_K, (
        f"k={_NEIGHBOUR_K} leaves only {_NEIGHBOUR_K - worst} spare over the worst bed seen "
        f"({worst}); that is too thin a margin for a silent truncation"
    )


@pytest.mark.parametrize("n", [64, 128, 256])
def test_the_served_cost_is_linear_in_N_not_quadratic(surrogate, n: int) -> None:
    """The regression that matters. Cost must scale with N.k, not N^2.

    Counts the *edges the network actually evaluates* rather than timing anything, so it is a
    deterministic assertion rather than a flaky benchmark. The dense form evaluated N^2 of them; the
    gathered form evaluates N.k. At N=256 that is a 16x difference, and it widens with N.
    """
    rng = np.random.default_rng(0)
    pos = np.column_stack(
        [rng.uniform(0.0, 0.8, n), rng.uniform(-0.15, 0.0, n)]
    )  # a bed-shaped scatter
    position = torch.tensor(pos, dtype=torch.float32)
    sender, edge_feat, mask = _neighbourhood(position, CUTOFF, _NEIGHBOUR_K)

    evaluated = edge_feat.shape[0] * edge_feat.shape[1]  # what the MLPs are run over
    assert evaluated == n * _NEIGHBOUR_K
    assert evaluated < n * n, "the served graph is still evaluating O(N^2) pairs"
    # ...and the shapes stay static in k, which is what keeps the ONNX export free of a
    # data-dependent edge count.
    assert sender.shape == (n, _NEIGHBOUR_K)
    assert mask.shape == (n, _NEIGHBOUR_K)
