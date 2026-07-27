"""MARL algorithm plugins + the Learn-internal registry (RM-P1-LEARN-03; learn.md §3, §11).

The reference baselines — **IPPO** (independent, the control), **MAPPO** and **QMIX** (the
CTDE default), and **comms_ppo** (the comms-learning research track: a differentiable-message
CTDE learner over the CommsModel's delivered links) — as registered, reproducible
:class:`Algorithm` / :class:`Trainer` plugins. They register through a **Learn-internal**
:class:`AlgorithmRegistry` (adding nothing to the Core waist); only the *policies* they produce
cross into the **Core** plugin registry as ``POLICY`` manifests (:func:`policy_manifest` /
:func:`manifest_from_export`). The :class:`PolicyExport` typed intermediate is the
RM-P1-LEARN-05 ONNX-export seam.

The concrete Torch-backed algorithms (``ippo``/``mappo``/``qmix``/``comms_ppo``) import the
``[rllib]`` extra and are lazy-loaded by the registry; the contract/registry surface here is
Torch-free.
"""

from __future__ import annotations

from astro_mine.learn.algos._contract import (
    AgentIoSignature,
    Algorithm,
    AlgorithmSpec,
    CentralizedCriticSpec,
    IoSignature,
    PolicyAssumptions,
    PolicyExport,
    Trainer,
)
from astro_mine.learn.algos._specs import COMMS_PPO_SPEC, IPPO_SPEC, MAPPO_SPEC, QMIX_SPEC
from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.algos.policy import (
    ActionHeads,
    InferFn,
    LearnedPolicy,
    action_heads,
    agent_io_signature,
    flat_obs_dim,
    flatten_obs,
    make_reference_policy,
)
from astro_mine.learn.algos.registry import (
    ALGORITHM_ENTRY_POINT_GROUP,
    AlgorithmRegistry,
    comms_learning_specs,
    default_registry,
    manifest_from_export,
    policy_manifest,
)

__all__ = [
    "ALGORITHM_ENTRY_POINT_GROUP",
    "COMMS_PPO_SPEC",
    "IPPO_SPEC",
    "MAPPO_SPEC",
    "QMIX_SPEC",
    "ActionHeads",
    "AgentIoSignature",
    "Algorithm",
    "AlgorithmRegistry",
    "AlgorithmSpec",
    "CentralizedCriticSpec",
    "InferFn",
    "IoSignature",
    "LearnedPolicy",
    "PolicyAssumptions",
    "PolicyExport",
    "TrainConfig",
    "Trainer",
    "action_heads",
    "agent_io_signature",
    "comms_learning_specs",
    "default_registry",
    "flat_obs_dim",
    "flatten_obs",
    "make_reference_policy",
    "manifest_from_export",
    "policy_manifest",
]
