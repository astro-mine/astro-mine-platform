"""Flattening helpers + the trained policy as a Core ``Policy`` (RM-P1-LEARN-03).

The bridge between the RL tensor world and the Core Policy/Planner contract, kept
**Torch-free** so the registry/contract tests run without the ``[rllib]`` extra:

- :func:`flatten_obs` / :func:`flat_obs_dim` collapse an agent's capability-keyed Dict
  observation sample into the flat vector a policy net consumes (deterministic key order,
  via Gymnasium's own :mod:`gymnasium.spaces` flattening);
- :func:`action_heads` decomposes an agent's Dict action space into its discrete
  (``kind``/``mode``) and continuous (``goto``/``hop``) heads;
- :class:`LearnedPolicy` adapts a per-agent ``infer`` callable (flat obs → action sample)
  into a Core :class:`~astro_mine.core.policy.Policy` that passes ``check_policy``. The
  ``infer`` is host-supplied — a trained Torch net at runtime, or a seeded reference policy
  in tests — mirroring the LEARN-05 ``OnnxPolicy`` "host-supplied infer" shape without
  running ONNX in-package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim, flatten
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy import DecisionContext, Policy
from astro_mine.learn.algos._contract import AgentIoSignature
from astro_mine.learn.envs.adapter.encode import (
    decode_action,
    encode_observation,
    zero_observation,
)
from astro_mine.learn.envs.adapter.spaces import AgentSpaceSpec

__all__ = [
    "ActionHeads",
    "InferFn",
    "LearnedPolicy",
    "action_heads",
    "agent_io_signature",
    "flat_obs_dim",
    "flatten_obs",
    "make_reference_policy",
]

#: A per-agent inference function: a flat float32 observation vector → an action sample dict
#: (the ``kind``/``mode`` indices + any ``goto``/``hop`` blocks) that ``decode_action``
#: consumes. Torch nets, ONNX runtimes (LEARN-05), or a seeded reference policy all fit.
InferFn = Callable[[NDArray[np.float32]], Mapping[str, Any]]


class ActionHeads:
    """The discrete + continuous heads an agent's Dict action space decomposes into."""

    __slots__ = ("box", "discrete")

    def __init__(self, discrete: dict[str, int], box: dict[str, int]) -> None:
        #: Selector name (``kind``/``mode``) → number of choices.
        self.discrete = discrete
        #: Continuous block name (``goto``/``hop``) → width (always 3).
        self.box = box


def action_heads(space: spaces.Dict) -> ActionHeads:
    """Decompose a Dict action space into its :class:`ActionHeads`."""
    discrete: dict[str, int] = {}
    box: dict[str, int] = {}
    for key, subspace in space.spaces.items():
        if isinstance(subspace, spaces.Discrete):
            discrete[key] = int(subspace.n)
        elif isinstance(subspace, spaces.Box):
            box[key] = int(np.prod(subspace.shape))
    return ActionHeads(discrete, box)


def flat_obs_dim(space: spaces.Dict) -> int:
    """The flattened width of a Dict observation space."""
    return int(flatdim(space))


def flatten_obs(
    sample: Mapping[str, NDArray[np.float32]], space: spaces.Dict
) -> NDArray[np.float32]:
    """Flatten one observation sample into a float32 vector (Gymnasium key order)."""
    return np.asarray(flatten(space, sample), dtype=np.float32)


def agent_io_signature(spec: AgentSpaceSpec) -> AgentIoSignature:
    """Derive one agent's :class:`AgentIoSignature` from its space spec (LEARN-05 seam)."""
    heads = action_heads(spec.action_space)
    return AgentIoSignature(
        obs_dim=flat_obs_dim(spec.observation_space),
        discrete_heads=dict(heads.discrete),
        box_heads=dict(heads.box),
    )


class LearnedPolicy(Policy):
    """A trained (or reference) policy as a Core :class:`~astro_mine.core.policy.Policy`.

    Holds one :class:`~astro_mine.learn.envs.adapter.spaces.AgentSpaceSpec` and one
    :data:`InferFn` per agent. :meth:`decide` encodes each Core observation into the agent's
    space, flattens it, runs ``infer``, and decodes the result back into a Core ``Action`` —
    so ``env.step(policy.decide(obs, ctx))`` closes the loop and the batch passes
    ``validate_action_batch`` (hence ``check_policy``). A masked/unobservable agent decides
    on the neutral zero observation (no sensor leak)."""

    def __init__(
        self,
        specs: Mapping[AgentId, AgentSpaceSpec],
        infer: Mapping[AgentId, InferFn],
    ) -> None:
        self._specs = dict(specs)
        self._infer = dict(infer)

    def act(
        self, flat_obs: Mapping[AgentId, NDArray[np.float32]]
    ) -> dict[AgentId, Mapping[str, Any]]:
        """Greedy per-agent action samples from *flat* observations — the eval-harness path
        (bench/reference.py), which drives the policy through a SwarmEnv rollout rather than
        the Core observation loop that :meth:`decide` serves."""
        return {a: self._infer[a](obs) for a, obs in flat_obs.items() if a in self._infer}

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        actions = []
        for agent_id, obs in observations.items():
            spec = self._specs.get(agent_id)
            infer = self._infer.get(agent_id)
            if spec is None or infer is None:  # pragma: no cover - observed agents are known
                continue
            sample = encode_observation(obs, spec) if obs.observable else zero_observation(spec)
            flat = flatten_obs(sample, spec.observation_space)
            actions.append(decode_action(infer(flat), spec))
        return ActionBatch(actions=actions)


def make_reference_policy(
    specs: Mapping[AgentId, AgentSpaceSpec], *, seed: int = 0
) -> LearnedPolicy:
    """A seeded, Torch-free reference policy — the honest-eval baseline a trained policy
    must beat, and the ``check_policy`` fixture for the no-extra CI job.

    Samples each agent's action heads from a per-agent NumPy RNG (deterministic under
    ``seed``): a categorical draw per discrete head, a uniform ``[-1, 1]`` draw per box
    head."""
    infer: dict[AgentId, InferFn] = {}
    for i, (agent_id, spec) in enumerate(specs.items()):
        infer[agent_id] = _reference_infer(spec, seed + i)
    return LearnedPolicy(specs, infer)


def _reference_infer(spec: AgentSpaceSpec, seed: int) -> InferFn:
    heads = action_heads(spec.action_space)
    rng = np.random.default_rng(seed)

    def infer(_obs: NDArray[np.float32]) -> Mapping[str, Any]:
        sample: dict[str, Any] = {
            name: int(rng.integers(0, n)) for name, n in heads.discrete.items()
        }
        for name, width in heads.box.items():
            sample[name] = rng.uniform(-1.0, 1.0, size=width).astype(np.float32)
        return sample

    return infer
