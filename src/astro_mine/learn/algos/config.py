# SPDX-License-Identifier: Apache-2.0
"""Declarative training config — the reproducibility key for a baseline run (LEARN-03).

A Pydantic v2 document (conventions.md §3) validated and recorded with every run: the same
``TrainConfig`` + the same ``CommsModelConfig`` + the same seed reproduce the identical
learning curve (the CX-REPRO determinism gate; learn.md §2.4, conventions.md §5, §11). It
is intentionally *serializable* — no callables — so it round-trips through JSON and lands
verbatim in the :class:`~astro_mine.core.registry.Provenance` of the produced policy.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Fidelity", "TrainConfig"]

#: Rollout fidelity tier, selectable per curriculum stage (learn.md §8; surrogate.md §2).
#: The tier is *Sim's* decision behind the Core Environment contract — Learn only **selects**
#: it (a config axis) and records the choice; it never imports Sim/Surrogate. ``sim_high`` is
#: full-fidelity Sim; ``surrogate`` is a surrogate fidelity tier honoring its tracked error
#: bounds (and flagged for a high-fidelity validation pass when it dominates training);
#: ``gpu_vectorized`` is a batched vector-env rollout (Brax/MJX/surrogate exposing the
#: Gymnasium vector API) via :class:`~astro_mine.learn.envs.vector.VectorExecutor`.
Fidelity = Literal["sim_high", "surrogate", "gpu_vectorized"]

_Pos = Annotated[int, Field(gt=0)]
_Prob = Annotated[float, Field(ge=0.0, le=1.0)]
_Pos_f = Annotated[float, Field(gt=0.0)]
_NonNeg_f = Annotated[float, Field(ge=0.0)]


class TrainConfig(BaseModel):
    """The declarative, replayable configuration of a baseline training run.

    Every field has a tier-1-friendly default so ``TrainConfig()`` trains a tiny model on a
    CPU workstation (learn.md §7). ``seed`` seeds Torch, NumPy, and the env together;
    ``hidden_sizes`` shapes the shared MLP trunk; the PPO knobs (``clip``/``gae_lambda``/
    ``entropy_coef``) are ignored by the value-based QMIX baseline, which reads ``gamma``,
    ``lr``, and ``hidden_sizes`` only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 0
    iterations: _Pos = 2
    rollout_steps: _Pos = 32
    hidden_sizes: tuple[int, ...] = (32, 32)
    lr: _Pos_f = 3.0e-3
    gamma: _Prob = 0.99
    gae_lambda: _Prob = 0.95
    clip: _Pos_f = 0.2
    update_epochs: _Pos = 1
    entropy_coef: _NonNeg_f = 0.0
    value_coef: _Pos_f = 0.5
    #: Recurrent (GRU) policy core for partial observability instead of a feed-forward MLP.
    use_rnn: bool = False
    #: QMIX mixer: ``vdn`` sums per-agent Q-values; ``qmix`` uses a monotonic hypernetwork
    #: mixer conditioned on the global ``state()``. Ignored by the PPO baselines.
    mixer: Literal["vdn", "qmix"] = "qmix"
    #: QMIX exploration rate (ε-greedy over the discrete task selector). Ignored by PPO.
    epsilon: _Prob = 0.1
    #: Rollout fidelity tier for this run / curriculum stage (RM-P1-LEARN-04). ``sim_high`` is
    #: the tier-1 default; ``surrogate``/``gpu_vectorized`` select cheaper tiers behind the
    #: Core Environment contract. Recorded in provenance and mapped to the honest
    #: surrogate-fidelity caveats a policy carries for Guard (learn.md §9).
    fidelity: Fidelity = "sim_high"
    #: Opaque world-provider selector threaded through to the injected env factory — Core's
    #: Environment/WorldProvider resolves it to a concrete fidelity tier; it is all the same
    #: SwarmEnv to Learn. JSON-serializable so it round-trips into provenance verbatim.
    world_provider: dict[str, Any] = Field(default_factory=dict)
    #: KubeRay rollout-worker / vector-env count (RM-P1-LEARN-04). ``1`` is the tier-1
    #: in-process default; recorded so a distributed/batched run is reproducible per topology.
    num_workers: _Pos = 1
    #: Fraction of surrogate-fidelity rollout steps above which the exported policy is flagged
    #: "mostly surrogate-trained → needs a high-fidelity validation pass" (issue AC; §9).
    surrogate_validation_threshold: _Prob = 0.5
