# SPDX-License-Identifier: Apache-2.0
"""The ``Algorithm`` / ``Trainer`` plugin contract for MARL baselines (RM-P1-LEARN-03).

Learn keeps a **Learn-internal** algorithm/trainer contract — it adds *nothing* to the
Core waist. Only the *produced policies* register in the **Core** plugin registry (as
:class:`~astro_mine.core.registry.PluginManifest` of kind ``POLICY``); the algorithm that
made them is discovered through this contract and Learn's own registry
(:mod:`astro_mine.learn.algos.registry`). A concrete algorithm (IPPO/MAPPO/QMIX) is a
:class:`Algorithm` that builds a :class:`Trainer`; the trainer runs reproducible
iterations, hands back a Core :class:`~astro_mine.core.policy.Policy`, and emits a typed
:class:`PolicyExport` intermediate.

The :class:`PolicyExport` is the **LEARN-05 seam**: a *typed* description of a trained
policy (weights + IO signature + declared comms assumptions + Core
:class:`~astro_mine.core.registry.Provenance`) — **not** raw ONNX. The
:mod:`astro_mine.learn.export` module renders this intermediate into a Core
:class:`~astro_mine.core.policy.PolicyPackage` (ONNX graph + typed sidecar); Learn does not
run ONNX in the trainer package. These IO/assumption types stay defined **here**, Learn-side,
as the framework-native intermediate: they carry the Torch ``state_dict`` and the raw env
spaces the export path derives the ONNX graph and Core ``IoSignature`` from — deliberately
richer than, and mapped onto, Core's wire-facing
:class:`~astro_mine.core.policy.PolicyPackage` (which Learn now builds against; RM-P1-CORE-01)
rather than duplicated into the waist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from astro_mine.core.env.model import AgentId
from astro_mine.core.registry.enums import CapabilityTag, PluginKind
from astro_mine.core.registry.model import Provenance

if TYPE_CHECKING:
    from astro_mine.core.policy import Policy
    from astro_mine.learn.algos.config import TrainConfig
    from astro_mine.learn.envs import SwarmEnv
    from astro_mine.learn.train.executor import RewardFn, RolloutExecutor

__all__ = [
    "POLICY_MANIFEST_INTERFACES",
    "AgentIoSignature",
    "Algorithm",
    "AlgorithmSpec",
    "Backend",
    "CentralizedCriticSpec",
    "IoSignature",
    "Paradigm",
    "PolicyAssumptions",
    "PolicyExport",
    "Trainer",
]

#: MARL paradigm (learn.md §11): fully independent learners vs. centralized-training /
#: decentralized-execution. IPPO is ``independent`` (the simple control); MAPPO and QMIX
#: are ``ctde`` (the default) and declare a :class:`CentralizedCriticSpec`.
Paradigm = Literal["independent", "ctde"]

#: The deep-learning backend an algorithm declares (learn.md §4). Only Torch is shipped in
#: RM-P1-LEARN-03; a JAX backend is a later throughput alternative (learn.md §11).
Backend = Literal["torch"]

#: The Core interface versions a produced-policy manifest is built against — the same waist
#: Learn consumes (``env``/``messages``/``sadf``/``policy``), pinned to the frozen Core
#: v0.1.0. Registered against a :class:`~astro_mine.core.registry.PluginRegistry`.
POLICY_MANIFEST_INTERFACES: dict[str, str] = {
    "env": "0.1.0",
    "messages": "0.1.0",
    "sadf": "0.1.0",
    "policy": "0.1.0",
}


@dataclass(frozen=True, slots=True)
class CentralizedCriticSpec:
    """A CTDE algorithm's declared centralized-critic input spec (issue AC; learn.md §3).

    References the global state the critic consumes — ``SwarmEnv.state()`` (a fixed-order
    concat of every possible agent's ``self_state`` block, zero-filled for absent agents) —
    plus the per-agent local observation dims the decentralized actors see. This is the
    honest CTDE declaration: training uses ``global_state_dim`` information the execution-time
    policy (which sees only ``per_agent_obs_dims``) does not.
    """

    global_state_dim: int
    per_agent_obs_dims: Mapping[AgentId, int]
    source: str = "SwarmEnv.state()"


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """The declarative, env-independent identity of a MARL algorithm plugin.

    ``capability_tag`` is the Learn-internal discovery tag the registry resolves algorithms
    by (entry-point group ``astro_mine.learn.algorithms``); ``paradigm`` documents the CTDE
    contract (``ctde`` algorithms build a :class:`CentralizedCriticSpec` per env, exposed on
    the trainer as :attr:`Trainer.centralized_critic`); ``comms_learning`` flags an algorithm
    that learns messages over the :class:`~astro_mine.learn.envs.CommsModel` channel (the
    first-class research track, learn.md §11).
    """

    name: str
    capability_tag: str
    paradigm: Paradigm
    backend: Backend = "torch"
    comms_learning: bool = False
    description: str = ""

    @property
    def is_ctde(self) -> bool:
        return self.paradigm == "ctde"


@dataclass(frozen=True, slots=True)
class AgentIoSignature:
    """One agent's flattened IO shape — the LEARN-05 ONNX-export descriptor for that head.

    ``obs_dim`` is the flattened observation width; ``discrete_heads`` maps each discrete
    action selector (``kind``/``mode``) to its cardinality; ``box_heads`` maps each
    continuous action block (``goto``/``hop``) to its width."""

    obs_dim: int
    discrete_heads: Mapping[str, int]
    box_heads: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class IoSignature:
    """The typed observation/action signature of an exported policy (LEARN-05 seam).

    Derived from the :class:`~astro_mine.learn.envs.SwarmEnv` spaces so RM-P1-LEARN-05 can
    build the ONNX graph's IO without re-deriving it, and Guard/Mind know the exact tensor
    contract. ``global_state_dim`` records the CTDE critic's ``state()`` width (0 for a
    fully decentralized policy)."""

    agent_ids: tuple[AgentId, ...]
    per_agent: Mapping[AgentId, AgentIoSignature]
    global_state_dim: int = 0


@dataclass(frozen=True, slots=True)
class PolicyAssumptions:
    """The declared comms/observability envelope a policy was trained under (LEARN-05).

    ``comms_observability`` is the JSON-serializable comms provenance from
    ``SwarmEnv.comms_provenance()`` (``None`` when unconstrained) — the honest metadata
    Guard enforces as a floor (learn.md §9). ``partial_observability`` records that the
    policy saw masked observations. ``surrogate_fidelity_caveats`` are the honest
    fidelity-tier caveats (RM-P1-LEARN-04) — e.g. "mostly surrogate-trained → needs a
    high-fidelity validation pass" — that RM-P1-LEARN-05 folds into the Core
    ``PolicyAssumptions.surrogate_fidelity_caveats`` so Guard knows the envelope to enforce."""

    comms_observability: Mapping[str, Any] | None = None
    partial_observability: bool = True
    surrogate_fidelity_caveats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyExport:
    """A trained policy as a typed intermediate — the RM-P1-LEARN-05 ONNX-export seam.

    Carries the framework-native ``weights`` (kept internal to Learn — never the
    cross-component artifact, learn.md §5), the :class:`IoSignature` derived from the env
    spaces, the :class:`PolicyAssumptions` (comms/observability envelope), and the Core
    :class:`~astro_mine.core.registry.Provenance` (seed, toolchain, lockfile). It is **not**
    ONNX: RM-P1-LEARN-05 renders the ONNX graph + ``PolicyPackage`` from this, runs the
    ONNX-Runtime equivalence check, and signs it."""

    algorithm: str
    backend: Backend
    io_signature: IoSignature
    assumptions: PolicyAssumptions
    provenance: Provenance
    #: Opaque framework-native weights (Torch ``state_dict`` tensors). Internal to Learn.
    weights: Mapping[str, Any] = field(default_factory=dict)
    #: The final reported metrics (mean return, seed variance, throughput) for the sidecar.
    metrics: Mapping[str, float] = field(default_factory=dict)
    #: The exported actor net kind — ``actor_critic`` (PPO ``DictActorCritic``) or ``q_net``
    #: (QMIX ``AgentQNet``) — the discriminator the export path dispatches the ONNX trace on.
    net_kind: Literal["actor_critic", "q_net"] = "actor_critic"
    #: Serializable per-agent net architecture (obs dim + head/hidden layout) — what the export
    #: path rebuilds the actor from to trace its ONNX graph (RM-P1-LEARN-05).
    net_arch: Mapping[AgentId, Mapping[str, Any]] = field(default_factory=dict)

    def policy_kind(self) -> PluginKind:
        """The Core registry kind a produced policy registers under — always ``POLICY``.

        There is deliberately no ``ALGORITHM``/``TRAINER`` kind: adding one would widen the
        (RFC-only) Core :class:`~astro_mine.core.registry.enums.PluginKind`. The algorithm
        stays Learn-internal; only its *policy* is a Core plugin."""
        return PluginKind.POLICY

    def capability_tags(self) -> list[CapabilityTag]:
        """Capability tags to stamp on the produced-policy manifest (none gated by default)."""
        return []


@runtime_checkable
class Trainer(Protocol):
    """A running training process for one algorithm on one env (learn.md §3).

    Deterministic under the config seed: ``learning_curve`` is byte-reproducible across
    two trainers built with the same config + env (the CX-REPRO determinism gate). Produces
    a Core :class:`~astro_mine.core.policy.Policy` (passes ``check_policy``) and a typed
    :class:`PolicyExport`.
    """

    @property
    def spec(self) -> AlgorithmSpec: ...

    @property
    def centralized_critic(self) -> CentralizedCriticSpec | None:
        """The per-env centralized-critic input spec for a CTDE algorithm; ``None`` for an
        independent (IPPO) learner."""
        ...

    def train_iteration(self) -> dict[str, float]:
        """Run one reproducible train iteration; return its scalar metrics."""
        ...

    def learning_curve(self) -> list[float]:
        """The per-iteration headline metric so far (the determinism-gate golden series)."""
        ...

    def policy(self) -> Policy:
        """The current trained policy as a Core Policy (decentralized-execution)."""
        ...

    def export(self) -> PolicyExport:
        """The typed export intermediate for the RM-P1-LEARN-05 ONNX path."""
        ...


@runtime_checkable
class Algorithm(Protocol):
    """A registered MARL algorithm plugin: a factory for :class:`Trainer`\\ s.

    Reference baselines ship as *replaceable examples*, not privileged internals
    (charter §10.2): a contributor registers a new :class:`Algorithm` through the Learn
    registry's entry-point group, discovered by :attr:`AlgorithmSpec.capability_tag`.
    """

    @property
    def spec(self) -> AlgorithmSpec: ...

    def make_trainer(
        self,
        env: SwarmEnv,
        config: TrainConfig,
        *,
        reward_fn: RewardFn | None = None,
        executor: RolloutExecutor | None = None,
    ) -> Trainer:
        """Build a :class:`Trainer` for ``env`` under ``config``.

        ``executor`` optionally overrides the rollout executor — the RM-P1-LEARN-04 seam that
        swaps the in-process :class:`~astro_mine.learn.train.executor.LocalExecutor` for a
        distributed/batched one without forking algorithm code — and ``reward_fn`` overrides
        the default reward shaping."""
        ...
