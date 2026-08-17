# SPDX-License-Identifier: Apache-2.0
"""Render a PolicyExport into a Core PolicyPackage (RM-P1-LEARN-05; learn.md §5, §9, §10).

The map from Learn's framework-native :class:`~astro_mine.learn.algos.PolicyExport` intermediate
to the **Core** :class:`~astro_mine.core.policy.PolicyPackage` — the wire-facing artifact
(ONNX graph + typed sidecar) Mind hosts, Guard wraps, and Bench scores. Learn reuses Core's
schema rather than widening the waist: it rebuilds the decentralized actor for one agent,
traces it to an ONNX graph, **runs the equivalence gate before returning** (so a divergent
graph is never emitted), and fills the Core IoSignature / PolicyAssumptions / Provenance from
the export. The graph's content hash is the artifact identity (ModelRef.digest).

Honest provenance for Guard (learn.md §9): the comms/observability regime (serialized from the
declared comms assumption), the box-action bounds, and the surrogate-fidelity caveats all land
in the Core :class:`~astro_mine.core.policy.PolicyAssumptions` so Guard knows the envelope.

Needs the ``[export]`` extra (``onnx``/``onnxruntime``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import nn

from astro_mine.core.env.model import AgentId
from astro_mine.core.hashing import canonical_json, content_hash, content_hash_json
from astro_mine.core.policy import validate_policy_package
from astro_mine.core.policy.enums import TensorDType
from astro_mine.core.policy.model import (
    POLICY_PACKAGE_VERSION,
    ModelRef,
    PolicyPackage,
    PolicyPackageDocument,
    TensorSpec,
)
from astro_mine.core.policy.model import (
    IoSignature as CoreIoSignature,
)
from astro_mine.core.policy.model import (
    PolicyAssumptions as CorePolicyAssumptions,
)
from astro_mine.core.policy.model import (
    Provenance as CoreProvenance,
)
from astro_mine.learn.algos._contract import POLICY_MANIFEST_INTERFACES, PolicyExport
from astro_mine.learn.export.equivalence import (
    assert_onnx_equivalence,
    assert_onnx_stateful_equivalence,
)
from astro_mine.learn.export.onnx import (
    DEFAULT_OPSET,
    input_specs_for,
    is_recurrent_graph,
    output_specs_for,
    to_onnx_bytes,
)
from astro_mine.learn.export.tensors import HIDDEN_INPUT, OBS_INPUT, STATE_TENSORS
from astro_mine.learn.models.mlp import AgentQNet, DictActorCritic

__all__ = ["ExportedPolicy", "content_id", "export_policy_package", "export_policy_packages"]


@dataclass(frozen=True)
class ExportedPolicy:
    """One agent's exported policy: the ONNX bytes + the typed Core sidecar + graph digest.

    Both the document *and* the bytes are returned because publish (export/publish.py) writes
    the bytes to the content-addressed store while Mind/Guard load the sidecar; the digest
    (``sha256:…`` of the graph) is the artifact identity carried on ``ModelRef.digest``."""

    agent: AgentId
    document: PolicyPackageDocument
    onnx_bytes: bytes
    digest: str


def _rebuild_export_net(net_kind: str, arch: Mapping[str, Any], weights: Any) -> nn.Module:
    """Rebuild one agent's decentralized actor from its serialized architecture + weights.

    Recurrent (``use_rnn``) and comms-learning (``comms_dim``) actors rebuild unchanged: their
    GRU state and aggregated peer-message context are *explicit graph tensors* on the export
    path (export/onnx.py), so neither needs a special-cased net — they only widen the declared
    IoSignature."""
    if net_kind == "actor_critic":
        net: nn.Module = DictActorCritic(
            arch["obs_dim"],
            arch["discrete_heads"],
            arch["box_heads"],
            arch["hidden_sizes"],
            use_rnn=bool(arch.get("use_rnn", False)),
            comms_dim=int(arch.get("comms_dim", 0)),
        )
    elif net_kind == "q_net":
        net = AgentQNet(arch["obs_dim"], arch["n_kinds"], arch["hidden_sizes"])
    else:  # pragma: no cover - net_kind is a closed Literal
        raise ValueError(f"unknown net_kind {net_kind!r}")
    net.load_state_dict(weights)
    net.eval()
    return net


def _tensor_kind(name: str, discrete_heads: set[str]) -> str:
    """Classify a declared graph output for the Core ``action_space`` descriptor.

    ``state`` marks a policy-state tensor (the recurrent ``hidden_out``) that a host carries
    across calls — it is **not** an action component, so it must not appear in the declared
    action bounds Guard enforces."""
    if name in STATE_TENSORS:
        return "state"
    return "discrete" if name in discrete_heads else "box"


def _comms_observability(export: PolicyExport) -> str | None:
    """Serialize the declared comms assumption into Core's ``str`` field (canonical JSON), so
    Guard can parse the exact regime back — ``None`` when the policy was trained unconstrained."""
    comms = export.assumptions.comms_observability
    if comms is None:
        return None
    return canonical_json(comms).decode("utf-8")


def export_policy_package(
    export: PolicyExport,
    agent: AgentId,
    *,
    version: str,
    name: str | None = None,
    opset: int = DEFAULT_OPSET,
    obs_batch: Any = None,
    env_lockfile: str | None = None,
) -> ExportedPolicy:
    """Render one agent's decentralized actor into a Core PolicyPackage document.

    Rebuilds the actor from ``export.net_arch`` + weights, traces it to a pinned-opset ONNX
    graph, **runs the ONNX-Runtime equivalence gate** (raises
    :class:`~astro_mine.learn.export.equivalence.EquivalenceError` on divergence — so a bad
    graph is never returned), and assembles the Core sidecar (IoSignature, honest
    PolicyAssumptions, Provenance). The returned document passes Core
    :func:`~astro_mine.core.policy.validate_policy_package`. ``env_lockfile`` is the lockfile
    content hash the caller (run.py / Cloud RunContext) supplies.

    **Recurrent (``use_rnn``) and comms-learning actors export too.** Their GRU hidden state
    and aggregated peer-message context are declared as explicit graph tensors
    (``hidden_in``/``hidden_out``, ``msg``) in the Core IoSignature, so the host knows the full
    contract it must bind. A recurrent export additionally passes the **multi-step stateful**
    equivalence gate (the hidden state fed back across steps), which a single-shot check cannot
    see."""
    if agent not in export.net_arch:
        raise KeyError(f"no net architecture for agent {agent!r} in the export")
    arch = export.net_arch[agent]
    net = _rebuild_export_net(export.net_kind, arch, export.weights[agent])
    obs_dim = export.io_signature.per_agent[agent].obs_dim

    onnx_bytes = to_onnx_bytes(net, obs_dim, opset=opset)
    # The gate: the graph must reproduce the Torch source before it can become an artifact —
    # single-shot, and (for a stateful actor) across a multi-step rollout that carries the
    # hidden state, the divergence a one-shot check is blind to.
    assert_onnx_equivalence(net, onnx_bytes, obs_dim, obs_batch=obs_batch)
    assert_onnx_stateful_equivalence(net, onnx_bytes, obs_dim)

    in_specs = [(name, obs_dim if name == OBS_INPUT else dim) for name, dim in input_specs_for(net)]
    out_specs = output_specs_for(net)
    discrete_heads = set(export.io_signature.per_agent[agent].discrete_heads)
    recurrent = is_recurrent_graph(net)
    comms_dim = int(arch.get("comms_dim", 0)) if export.net_kind == "actor_critic" else 0

    io_signature = CoreIoSignature(
        inputs=[
            TensorSpec(name=name, dtype=TensorDType.FLOAT32, shape=[-1, dim])
            for name, dim in in_specs
        ],
        outputs=[
            TensorSpec(name=name, dtype=TensorDType.FLOAT32, shape=[-1, dim])
            for name, dim in out_specs
        ],
        observation_space={
            "flat_dim": obs_dim,
            "dtype": "float32",
            # The honest declaration a host binds against: a recurrent policy needs its state
            # carried across calls, a comms-learning one needs the peer-message aggregate fed in
            # (zeros for an isolated agent — the MessageModule semantics).
            "recurrent": recurrent,
            "hidden_dim": dict(in_specs).get(HIDDEN_INPUT, 0),
            "comms_dim": comms_dim,
            "stateful_inputs": [name for name, _dim in in_specs if name != OBS_INPUT],
        },
        action_space={
            "outputs": [
                {"name": name, "dim": dim, "kind": _tensor_kind(name, discrete_heads)}
                for name, dim in out_specs
            ]
        },
    )

    assumptions = CorePolicyAssumptions(
        comms_observability=_comms_observability(export),
        action_bounds={
            name: {"low": -1.0, "high": 1.0, "dim": dim}
            for name, dim in out_specs
            if _tensor_kind(name, discrete_heads) == "box"
        },
        surrogate_fidelity_caveats=list(export.assumptions.surrogate_fidelity_caveats),
        deterministic=True,
    )

    learn_prov = export.provenance
    provenance = CoreProvenance(
        input_hashes=list(learn_prov.input_hashes),
        code_version=learn_prov.code_version,
        toolchain_version=learn_prov.toolchain_version,
        env_lockfile=env_lockfile or learn_prov.env_lockfile,
        seed=learn_prov.seed,
    )

    digest = content_hash(onnx_bytes)
    package = PolicyPackage(
        name=name if name is not None else f"{export.algorithm}.{agent}",
        version=version,
        onnx_model=ModelRef(digest=digest, opset=opset),
        io_signature=io_signature,
        core_interfaces=dict(POLICY_MANIFEST_INTERFACES),
        assumptions=assumptions,
        provenance=provenance,
    )
    document = PolicyPackageDocument(
        policy_package_version=POLICY_PACKAGE_VERSION,  # type: ignore[arg-type]
        policy_package=package,
    )
    validate_policy_package(document)  # structural check against Core's shipped JSON Schema
    return ExportedPolicy(agent=agent, document=document, onnx_bytes=onnx_bytes, digest=digest)


def export_policy_packages(
    export: PolicyExport,
    *,
    version: str,
    opset: int = DEFAULT_OPSET,
    env_lockfile: str | None = None,
) -> dict[AgentId, ExportedPolicy]:
    """Export every heterogeneous agent's decentralized actor as its own PolicyPackage (one
    ONNX graph per agent; each gated by the equivalence check)."""
    return {
        agent: export_policy_package(
            export, agent, version=version, opset=opset, env_lockfile=env_lockfile
        )
        for agent in export.net_arch
    }


def content_id(document: PolicyPackageDocument) -> str:
    """The content-addressed identity of the sidecar document (``sha256:…``).

    The graph's own digest (``ModelRef.digest``) is the primary artifact identity; this hashes
    the whole typed sidecar so a change to any declared metadata is itself content-addressed."""
    return content_hash_json(document.model_dump(mode="json"))
