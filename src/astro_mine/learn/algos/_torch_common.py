# SPDX-License-Identifier: Apache-2.0
"""Torch-side reproducibility + RL math shared by the baselines (RM-P1-LEARN-03).

Imported only by the Torch-backed trainers (never by the registry/contract path), this
module centralises the CX-REPRO machinery — global + generator seeding, GAE, the default
deterministic reward shaping for a reward-free env — and the Core
:class:`~astro_mine.core.registry.Provenance` every produced policy carries. Same seed +
same config ⇒ identical learning curve (conventions.md §5, §11; learn.md §2.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

from astro_mine.core.registry.model import Provenance
from astro_mine.learn.algos.config import TrainConfig

__all__ = [
    "MESSAGE_DIM",
    "MESSAGE_FEAT_DIM",
    "compute_gae",
    "fidelity_caveats",
    "make_generator",
    "provenance",
    "seed_everything",
    "toolchain_version",
]

#: The learned-message width for the comms-learning entry point (models/comms.py) — the size of
#: the aggregated peer-message context that widens a comms-learning actor's trunk and is declared
#: as the exported graph's explicit ``msg`` input (export/tensors.py).
MESSAGE_DIM = 8

#: The shared message-*feature* width every heterogeneous agent projects its own (differently
#: sized) observation into before the one shared ``MessageModule`` encodes it — what lets a
#: single message channel span SADF-heterogeneous agents (models/comms.py::CommsEncoder).
MESSAGE_FEAT_DIM = 16


def fidelity_caveats(config: TrainConfig) -> tuple[str, ...]:
    """Honest surrogate-fidelity caveats for the exported policy (RM-P1-LEARN-04 AC; §9).

    A run configured on the ``surrogate`` tier is trained (by construction) entirely on the
    surrogate fidelity tier — above ``surrogate_validation_threshold`` — so it is flagged for
    a high-fidelity (Sim) validation pass, the caveat Guard reads off the ``PolicyPackage``.
    ``sim_high`` and ``gpu_vectorized`` add no surrogate caveat (a GPU-vectorized env may be
    full-fidelity Brax/MJX); a per-episode surrogate-step fraction read off env/ErrorReport
    metadata (surrogate.md §2) refines this when a surrogate-backed world reports it."""
    if config.fidelity == "surrogate":
        return (
            "trained on the surrogate fidelity tier "
            f"(>= {config.surrogate_validation_threshold:.0%} of rollout steps); "
            "needs a high-fidelity (Sim) validation pass before operational use",
        )
    return ()


def seed_everything(seed: int, *, device: str = "cpu") -> None:
    """Seed Torch + NumPy and force deterministic, single-threaded CPU execution.

    The determinism gate (``tests/algos/test_determinism.py``) requires two trainers built
    with the same config to produce byte-identical learning curves; single-threaded
    deterministic Torch on CPU is what makes that hold — the CPU tier-1 default (learn.md §7).
    ``device`` is the optional CUDA-placement hook (RM-P1-LEARN-04): a ``cuda`` device also
    seeds the CUDA RNGs while keeping deterministic algorithms on, so the single-GPU overnight
    run is reproducible; the CPU single-thread pin (the byte-identity gate) is unconditional
    and harmless on GPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    if torch.device(device).type == "cuda":  # pragma: no cover - GPU path is CI-deselected
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int) -> torch.Generator:
    """A dedicated, seeded Torch generator for action sampling (independent of the global
    RNG so training stays reproducible regardless of other Torch ops)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Generalized Advantage Estimation over one (agent or team) trajectory.

    Bootstraps 0 past the trajectory end (the episode terminated or truncated). Returns
    ``(advantages, returns)`` with ``returns = advantages + values``."""
    n = len(rewards)
    advantages = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        next_value = 0.0 if (t + 1 >= n or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * next_value * (0.0 if dones[t] else 1.0) - values[t]
        gae = delta + gamma * gae_lambda * (0.0 if dones[t] else 1.0) * gae
        advantages[t] = gae
    returns = [advantages[t] + values[t] for t in range(n)]
    return advantages, returns


def toolchain_version() -> str:
    """A compact record of the training toolchain versions for provenance."""
    parts = []
    for pkg in ("torch", "ray", "gymnasium", "numpy"):
        try:
            parts.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:  # pragma: no cover - all present under the [rllib] extra
            continue
    return ";".join(parts)


def _learn_version() -> str:
    try:
        return version("astro-mine-platform")
    except PackageNotFoundError:  # pragma: no cover - editable installs always resolve a version
        return "0.0.0"


def provenance(
    config: TrainConfig, *, comms_provenance: dict[str, object] | None = None
) -> Provenance:
    """The Core :class:`~astro_mine.core.registry.Provenance` for a produced policy.

    Records the seed, the training toolchain, and the code version — the reproducibility
    chain Bench re-derives from. RM-P1-LEARN-05 attaches the ONNX-graph digest and the
    lockfile hash to complete it."""
    return Provenance(
        code_version=_learn_version(),
        toolchain_version=toolchain_version(),
        seed=config.seed,
        input_hashes=[] if comms_provenance is None else ["comms:" + str(sorted(comms_provenance))],
    )
