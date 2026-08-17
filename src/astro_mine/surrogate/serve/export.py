# SPDX-License-Identifier: Apache-2.0
"""ONNX export of the excavation surrogate as a raw-state self-contained graph (RM-P1-SURR-04).

Exports the **whole** query — featurization *and* prediction — into one ONNX graph: from the raw
particle state (``position``/``velocity``/``tool_x``/``config``) through the radius graph, node/edge
features, the deep-ensemble GNS, denormalization, integration, ensemble mean/std, the conformal
half-width, the trust-region box, and the OOD inflation. The served artifact therefore takes **raw
state in** and emits ``next_state`` plus calibrated ``uncertainty``/``in_domain``/``ood_margin``
**from ONNX Runtime alone** — so a consumer (Sim's scheduler RM-P1-SIM-03, Worlds RM-P1-WORLDS-10)
runs the tier with **no featurization code and no import of this package** (surrogate.md §6, §11
"both error channels"; the narrow waist).

The radius graph is reduced to each receiver's **k nearest senders** and the message passing runs
over ``(N, k)`` tensors — ``O(N.k)``, with ``k`` a compile-time constant, so the graph still has no
data-dependent edge count (the property ONNX needs, and the reason the first version of this export
went dense). It is numerically identical to the sparse
:class:`~astro_mine.surrogate.models.gns.GNS`: the same neighbours summed into the same receiver.

That reduction is the difference between a tier that pays for itself and one that does not. The
dense ``(N, N)`` form it replaces ran the edge encoder and every message MLP across **all** ``N^2``
pairs and then masked ~99% of them away — 175x wasted work at N=1000 — which made the served tier
``O(N^2)``, the same asymptotics as the DEM solver it exists to replace, and pinned the speedup at
~2x at every bed size (surrogate#24).

torch is imported here (an export-time build tool); the served
:mod:`~astro_mine.surrogate.serve.runtime` runs the graph through ONNX Runtime and never imports
torch.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import Tensor, nn

from astro_mine.surrogate.enums import ServedBackend
from astro_mine.surrogate.models.excavation import (
    _CHANNELS,
    _OOD_INFLATION,
    _UNITS,
    ExcavationSurrogate,
)
from astro_mine.surrogate.serve.bundle import SERVE_META_FORMAT_VERSION, OnnxBundle

__all__ = ["OnnxExportError", "export_excavation_surrogate"]

# The one opset the dynamo exporter targets for this graph. Pinned so the exported bytes and the
# ops used stay stable.
_OPSET = 18
# Distance floor under the sqrt so self-distances stay finite and the radius test is stable.
_EPS = 1e-12
# A distance no bed can produce, used to push self-pairs out of TopK's reach. A literal `inf` would
# also work but leaves NaNs in the traced graph's gradients; a large finite value keeps the export
# clean and is unreachable for a bed measured in metres.
_FAR = 1.0e6

#: Senders gathered per receiver in the served graph — the ``k`` of the ``O(N.k)`` message passing.
#:
#: **Why a fixed k, and why 16.** The radius graph's true neighbour count is set by *packing
#: geometry*, not by bed size: at the trained cutoff (``2.6 r``) a settled, actively-ploughed bed
#: measures a mean of ~5 neighbours and a **worst case of 7**, unchanged from N=90 to N=1000 across
#: every seed tried. Hexagonal packing admits 6; soft-sphere overlap buys a little more. 16 is ~2.3x
#: that worst case.
#:
#: The point of fixing it is that ``(N, k)`` is a **static** shape. That is precisely the property
#: the old dense ``(N, N)`` formulation was chosen for — ONNX handles a data-dependent edge count
#: poorly — except that the dense form paid ``O(N^2)`` for it, running the edge encoder and every
#: message MLP over all ``N^2`` pairs and then masking ~99% of them away (surrogate#24):
#:
#:     N        real edges   N^2 pairs    useful      work discarded
#:     90              450       8,100      5.6%                 18x
#:     1000          5,716   1,000,000      0.57%               175x
#:     2000         11,494   4,000,000      0.29%               348x
#:
#: which made the served tier ``O(N^2)`` — the same asymptotics as the DEM solver it replaces, with
#: a far worse constant per pair — and pinned the speedup at ~2x at *every* bed size.
#:
#: ``TopK`` keeps the **nearest** k, so if a bed ever did exceed k neighbours the senders dropped
#: would be the most distant (weakest) ones: the failure is graceful, not a cliff. Even so it must
#: not happen silently, so :func:`_max_neighbours` asserts it at export time against the fixture the
#: tier was trained on, and a test asserts it against freshly settled beds at several N.
_NEIGHBOUR_K = 16


class OnnxExportError(RuntimeError):
    """An op in the surrogate's forward could not be expressed in ONNX.

    The manifest's ``native_graph_fallback`` flag exists for exactly this case (a native-graph
    *serving* runtime is deferred, surrogate.md §11) — the excavation graph exports cleanly, so this
    is raised only if a future op regresses the export.
    """


def _gathered_gns(
    gns: nn.Module, node_feat: Tensor, edge_feat: Tensor, mask: Tensor, sender: Tensor
) -> Tensor:
    """The sparse :meth:`GNS.forward` over gathered ``(N, k)`` neighbourhoods — ``O(N.k)``.

    ``node_feat`` is ``(N, F)`` (normalized); ``edge_feat`` the ``(N, k, 3)`` features of each
    receiver's k nearest senders; ``mask`` the ``(N, k)`` 0/1 flag for "this sender is really inside
    the cutoff"; ``sender`` the ``(N, k)`` indices of those senders. Returns the per-particle
    acceleration ``(N, 2)``.

    This is the same computation the sparse :class:`GNS` does — for receiver ``j``, the messages
    from exactly the senders ``i`` with ``|xi - xj| < cutoff``, summed into ``j`` — and therefore
    the same computation the old dense ``(N, N)`` form did. What it does *not* do is evaluate the
    MLPs on the ``N^2 - N.k`` pairs the dense mask was about to zero anyway (:data:`_NEIGHBOUR_K`).
    """
    h = gns.node_encoder(node_feat)  # (N, H)
    e = gns.edge_encoder(edge_feat)  # (N, k, H)  <- was (N, N, H)
    hidden = h.shape[1]
    k = sender.shape[1]
    for step in range(gns.steps):
        h_src = h[sender]  # (N, k, H) — the sender's embedding, gathered
        h_dst = h[:, None, :].expand(-1, k, hidden)  # (N, k, H) — the receiver's own
        messages = gns.message_mlps[step](torch.cat([h_src, h_dst, e], dim=-1))  # (N, k, H)
        aggregated = (mask[:, :, None] * messages).sum(
            dim=1
        )  # sum this receiver's senders -> (N,H)
        h = h + gns.update_mlps[step](torch.cat([h, aggregated], dim=-1))
    out: Tensor = gns.decoder(h)
    return out


def _neighbourhood(position: Tensor, cutoff: float, k: int) -> tuple[Tensor, Tensor, Tensor]:
    """Each receiver's ``k`` nearest senders: ``(sender_idx, edge_feat, mask)``, all static-shaped.

    The pairwise distance matrix stays ``O(N^2)`` — but it is ~one flop per pair, not several MLP
    layers, and it is not what the dense formulation was paying for. ``TopK`` then reduces the graph
    to ``(N, k)`` *before* any network touches it.
    """
    n = position.shape[0]
    diff = position[:, None, :] - position[None, :, :]  # (N, N, 2) = pos[j] - pos[i]
    dist = torch.sqrt((diff**2).sum(-1) + _EPS)  # (N, N), symmetric
    index = torch.arange(n)
    # Exclude self by pushing the diagonal beyond every cutoff, so TopK never selects it.
    dist = torch.where(index[:, None] == index[None, :], torch.full_like(dist, _FAR), dist)

    nearest, sender = torch.topk(dist, k, dim=1, largest=False)  # (N, k) both
    mask = (nearest < cutoff).to(position.dtype)  # (N, k) — beyond the radius contributes nothing

    # ef[j, m] = [pos[j] - pos[i], |.|] for i = sender[j, m] — the sparse `edge_features` convention
    # (`rel = pos[dst] - pos[src]`), with receiver j as dst.
    rel = position[:, None, :] - position[sender]  # (N, k, 2)
    edge_feat = torch.cat([rel, nearest[:, :, None]], dim=-1)  # (N, k, 3)
    return sender, edge_feat, mask


class _RawStatePredictor(nn.Module):  # type: ignore[misc]  # nn.Module is Any under the torch override
    """The full query as one exportable graph: raw particle state in, calibrated prediction out.

    Wraps the trained ensemble and folds every constant (radius cutoff, bed width, normalizer stats,
    conformal ``q``/``floor``, the trust-region box, the OOD inflation) into buffers. ``forward``
    featurizes the raw state (radius graph → node/edge features → normalize), runs the dense GNS
    ensemble, and returns ``(mean, half_width, in_domain, ood_margin)``.
    """

    def __init__(self, surrogate: ExcavationSurrogate) -> None:
        super().__init__()
        from astro_mine.surrogate.models.train import _CUTOFF_RADII, _PARTICLE_RADIUS_M

        self.members = nn.ModuleList(surrogate._models)
        self.dt = float(surrogate._dataset.dt_s)
        self.cutoff = _CUTOFF_RADII * _PARTICLE_RADIUS_M
        self.bed_width = float(surrogate._dataset.bed_width_m)
        norm, conf, tr = surrogate._normalizer, surrogate._conformal, surrogate._trust_region
        for name, array in (
            ("node_mean", norm.node_mean),
            ("node_std", norm.node_std),
            ("config_mean", norm.config_mean),
            ("config_std", norm.config_std),
            ("accel_mean", norm.accel_mean),
            ("accel_std", norm.accel_std),
            ("q", conf.quantiles),
            ("floor", conf.floor),
            ("lower", tr.lower),
            ("upper", tr.upper),
        ):
            self.register_buffer(name, torch.tensor(array, dtype=torch.float32))

    def forward(
        self, position: Tensor, velocity: Tensor, tool_x: Tensor, config: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state = torch.cat([position, velocity], dim=1)  # (N, 4)
        n = position.shape[0]
        # The radius graph, reduced to each receiver's k nearest senders *before* the network sees
        # it: O(N.k) message passing, static shapes (surrogate#24). The old form built a dense
        # (N, N) adjacency and ran every MLP across all N^2 pairs to mask ~99% of them away.
        sender, edge_feat, mask = _neighbourhood(position, self.cutoff, _NEIGHBOUR_K)
        # Node features [vel(2), floor/left/right/blade(4), config_norm(2)], then normalize.
        config_norm = (config[:2] - self.config_mean) / self.config_std
        boundary = torch.stack(
            [
                position[:, 1],
                position[:, 0],
                self.bed_width - position[:, 0],
                position[:, 0] - tool_x[0],
            ],
            dim=1,
        )
        node_feat = torch.cat([velocity, boundary, config_norm[None, :].expand(n, 2)], dim=1)
        node_feat = (node_feat - self.node_mean) / self.node_std  # (N, 8)
        preds = []
        for gns in self.members:
            accel = (
                _gathered_gns(gns, node_feat, edge_feat, mask, sender) * self.accel_std
                + self.accel_mean
            )
            next_vel = state[:, 2:] + accel * self.dt
            next_pos = state[:, :2] + next_vel * self.dt
            preds.append(torch.cat([next_pos, next_vel], dim=1))
        stacked = torch.stack(preds, dim=0)  # (K, N, 4)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)  # population std, matching numpy .std()
        half = (std + self.floor) * self.q  # conformal half-width
        # Trust-region membership + signed margin (ExcavationTrustRegion.contains / .margin).
        width = self.upper - self.lower
        in_domain = torch.all((config >= self.lower) & (config <= self.upper))
        safe = torch.where(width > 0, width, torch.ones_like(width))
        per_param = torch.minimum((config - self.lower) / safe, (self.upper - config) / safe)
        active = width > 0
        margin = torch.where(active, per_param, torch.full_like(per_param, float("inf"))).min()
        margin = torch.where(active.any(), margin, torch.zeros_like(margin))
        # Out-of-domain: inflate the interval, never a confident extrapolation (principle 3).
        half = torch.where(in_domain, half, half * _OOD_INFLATION)
        return mean, half, in_domain.to(torch.int64), margin


def _sample_inputs(surrogate: ExcavationSurrogate) -> tuple[Tensor, ...]:
    """Representative raw-state tensors (the fixture's first frame) to trace the export."""
    ds = surrogate._dataset
    state = ds.states[0, 0]
    return (
        torch.tensor(state[:, :2], dtype=torch.float32),
        torch.tensor(state[:, 2:], dtype=torch.float32),
        torch.tensor([float(ds.tool_x[0, 0])], dtype=torch.float32),
        torch.tensor(ds.params[0], dtype=torch.float32),
    )


def _serve_meta(surrogate: ExcavationSurrogate) -> dict[str, object]:
    """Channel/field layout the served runtime maps the graph outputs onto.

    Raw-state featurization now lives *in the graph*, so no normalizer/cutoff params are needed
    here — only the query's field vocabulary and the output layout.
    """
    tr = surrogate._trust_region
    return {
        "format_version": SERVE_META_FORMAT_VERSION,
        "domain": surrogate.error_report.domain.value,
        "ensemble_size": len(surrogate._models),
        "channels": list(_CHANNELS),
        "units": list(_UNITS),
        "param_names": list(tr.param_names),
        "input_fields": ["position", "velocity", "tool_x", "config"],
        # (field_name, dim) — how the runtime splits the (N,4) mean/half into per-field arrays.
        "output_field_layout": [["position", 2], ["velocity", 2]],
    }


def _assert_k_covers_the_fixture(surrogate: ExcavationSurrogate) -> None:
    """Refuse to export a graph whose ``k`` the tier's own training beds already exceed.

    ``TopK`` keeps the *nearest* k senders, so an over-full neighbourhood degrades gracefully (the
    dropped edges are the most distant, weakest ones) rather than corrupting outright. Graceful is
    not the same as acceptable: it would be a silent, permanent under-count of the physics, and this
    package has shipped enough numbers that were quietly wrong. So it is checked, on the very beds
    the tier learned from, and the export fails loudly if ``k`` is too small.
    """
    from astro_mine.surrogate.models.train import _CUTOFF_RADII, _PARTICLE_RADIUS_M

    cutoff = _CUTOFF_RADII * _PARTICLE_RADIUS_M
    states = surrogate._dataset.states  # (C, T+1, N, 4)
    worst = 0
    for config in range(states.shape[0]):
        for frame in range(states.shape[1]):
            pos = states[config, frame, :, :2]
            dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dist, np.inf)
            worst = max(worst, int((dist < cutoff).sum(axis=1).max()))
    if worst > _NEIGHBOUR_K:
        raise OnnxExportError(
            f"a particle in the training fixture has {worst} neighbours inside the {cutoff:.4f} m "
            f"cutoff, but the served graph gathers only k={_NEIGHBOUR_K}. TopK would silently drop "
            f"the {worst - _NEIGHBOUR_K} most distant of them on every step. Raise `_NEIGHBOUR_K`."
        )


def export_excavation_surrogate(surrogate: ExcavationSurrogate) -> OnnxBundle:
    """Export a trained :class:`ExcavationSurrogate` to a raw-state served :class:`OnnxBundle`.

    Traces the self-contained graph (dynamic in the particle count ``N``) and packages it with its
    field-layout ``serve_meta`` and calibrated ``ErrorReport``. Raises :class:`OnnxExportError` if
    the graph cannot be expressed (the ``native_graph_fallback`` manifest case).
    """
    _assert_k_covers_the_fixture(surrogate)
    predictor = _RawStatePredictor(surrogate).eval()
    args = _sample_inputs(surrogate)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "model.onnx"
        try:
            with warnings.catch_warnings():
                # The dynamo exporter emits benign deprecation/user warnings; the export is verified
                # correct (ORT parity to float precision, generalizes across N).
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    predictor,
                    args,
                    str(out_path),
                    input_names=["position", "velocity", "tool_x", "config"],
                    output_names=["mean", "half", "in_domain", "margin"],
                    dynamic_axes={
                        "position": {0: "N"},
                        "velocity": {0: "N"},
                        "mean": {0: "N"},
                        "half": {0: "N"},
                    },
                    opset_version=_OPSET,
                    dynamo=True,
                )
        except Exception as exc:
            raise OnnxExportError(
                "failed to export the surrogate forward to ONNX; the native-graph serving fallback "
                "(ServedBackend.NATIVE_GRAPH) is not implemented in Phase 1"
            ) from exc
        # dynamo may write large initializers to a `model.onnx.data` sidecar; load the model back
        # (resolving that sidecar) and re-serialize so the bundle carries the weights inline.
        onnx_model = onnx.load(str(out_path)).SerializeToString()
    return OnnxBundle(
        onnx_model=onnx_model,
        serve_meta=_serve_meta(surrogate),
        error_report=surrogate.error_report,
    )


# The served backend this exporter produces — the manifest records it (ServedBackend.ONNX, with
# native_graph_fallback=False since the graph exports cleanly).
SERVED_BACKEND = ServedBackend.ONNX
