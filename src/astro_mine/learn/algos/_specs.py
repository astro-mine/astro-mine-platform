# SPDX-License-Identifier: Apache-2.0
"""Env-independent :class:`AlgorithmSpec`\\ s for the reference baselines (RM-P1-LEARN-03).

Kept in a **Torch-free** module so the registry can list algorithms, resolve them by
capability tag, and emit Core ``POLICY`` manifests *without* importing the ``[rllib]``
extra. The concrete (Torch-backed) :class:`~astro_mine.learn.algos._contract.Algorithm`
classes in ``ippo``/``mappo``/``qmix`` import these specs and are themselves lazy-loaded by
the registry only when a trainer is actually built.
"""

from __future__ import annotations

from astro_mine.learn.algos._contract import AlgorithmSpec

__all__ = ["COMMS_PPO_SPEC", "IPPO_SPEC", "MAPPO_SPEC", "QMIX_SPEC"]

IPPO_SPEC = AlgorithmSpec(
    name="ippo",
    capability_tag="marl.independent.ppo",
    paradigm="independent",
    description=(
        "Independent PPO — each agent optimises its own actor/critic with no shared "
        "information (the simple control; learn.md §11)."
    ),
)

MAPPO_SPEC = AlgorithmSpec(
    name="mappo",
    capability_tag="marl.ctde.mappo",
    paradigm="ctde",
    description=(
        "Multi-Agent PPO — decentralised actors with a centralised critic over the global "
        "SwarmEnv.state() (the CTDE default; learn.md §11)."
    ),
)

QMIX_SPEC = AlgorithmSpec(
    name="qmix",
    capability_tag="marl.ctde.qmix",
    paradigm="ctde",
    description=(
        "QMIX/VDN — shared per-agent Q-nets over the discrete task selector, mixed by a "
        "monotonic (QMIX) or additive (VDN) mixer conditioned on the global state (CTDE)."
    ),
)

COMMS_PPO_SPEC = AlgorithmSpec(
    name="comms_ppo",
    capability_tag="marl.ctde.comms_ppo",
    paradigm="ctde",
    comms_learning=True,
    description=(
        "Differentiable-message CTDE PPO — MAPPO's centralized critic plus a learned message "
        "channel: each agent encodes a message from its own observation and conditions on the "
        "mean-pooled messages of the peers the CommsModel actually DELIVERED this tick "
        "(gate → budget → drop → delay). The message encoder is trained end-to-end by the team "
        "objective, so what to say is learned under the same comms constraint every other "
        "baseline is scored on — the first-class research track (learn.md §11; charter §8)."
    ),
)
