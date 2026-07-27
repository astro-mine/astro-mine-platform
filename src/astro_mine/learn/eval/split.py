"""Held-out evaluation seeds, separated from training **by construction** (RM-P1-LEARN-06).

Honest evaluation begins with an eval set the policy has never trained on. Learn enforces the
separation *structurally* rather than by convention (learn.md §10; issue AC "separated from
training envs by construction"):

- :class:`HeldOutSplit` — a frozen Pydantic v2 document holding the ``train_seeds`` and the
  ``held_out_seeds`` with a validator that **raises** if the two intersect, so an overlapping
  split can never be constructed.
- :func:`partition` — draws the two disjoint sets from **separate**
  :class:`numpy.random.SeedSequence` namespaces (a distinct salt per role), decorrelated from
  the ``SeedSequence([seed, index])`` stream :func:`~astro_mine.learn.train.executor.derive_seeds`
  uses for distributed rollout slices — so a training run's per-worker seeds can never collide
  with a held-out seed. The held-out set is then filtered to be provably disjoint from the
  training set before the :class:`HeldOutSplit` is built.

A held-out *env* is a distinct declared factory (a different world seed / scenario) from the
training env; :meth:`HeldOutSplit.assert_holds_out` additionally asserts, against the seed a
run actually trained on (``TrainConfig.seed`` / the produced policy's ``Provenance``), that it
is not in the held-out set — the honest-eval guard the harness applies before scoring.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["HeldOutSplit", "partition"]

#: Distinct :class:`numpy.random.SeedSequence` salts that namespace the training and held-out
#: seed streams away from each other **and** from ``derive_seeds``'s ``SeedSequence([seed,
#: index])`` (a two-int entropy vector with a small ``index``). Bump only on an intentional
#: seed-realization change.
_TRAIN_SEED_SALT = 0x5EED_7241
_HELD_OUT_SEED_SALT = 0x5EED_E7A1

_PosInt = Annotated[int, Field(gt=0)]


class HeldOutSplit(BaseModel):
    """A training / held-out seed split whose two sets are disjoint by construction.

    Constructing a split whose ``train_seeds`` intersect ``held_out_seeds`` raises — the
    honest-eval separation is a validated invariant, not a convention."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_seeds: frozenset[int]
    held_out_seeds: tuple[int, ...]

    @model_validator(mode="after")
    def _check_disjoint(self) -> HeldOutSplit:
        if not self.held_out_seeds:
            raise ValueError("a held-out split needs at least one held-out seed")
        if len(set(self.held_out_seeds)) != len(self.held_out_seeds):
            raise ValueError(f"held_out_seeds must be unique, got {self.held_out_seeds}")
        overlap = self.train_seeds & set(self.held_out_seeds)
        if overlap:
            raise ValueError(
                f"held-out seeds must be separated from training seeds by construction; "
                f"overlap: {sorted(overlap)}"
            )
        return self

    def assert_holds_out(self, *trained_seeds: int) -> None:
        """Raise if a seed a run *actually trained on* leaks into the held-out set.

        Applied by the harness before scoring so a reported held-out number can never have
        been trained on (issue AC; learn.md §10)."""
        leaked = set(int(s) for s in trained_seeds) & set(self.held_out_seeds)
        if leaked:
            raise ValueError(
                f"trained-on seed(s) {sorted(leaked)} appear in the held-out set — the "
                "evaluation would not be held out"
            )


def _draw_distinct(
    base_seed: int, salt: int, count: int, *, avoid: frozenset[int]
) -> tuple[int, ...]:
    """Deterministically draw ``count`` distinct 32-bit seeds from the ``(base_seed, salt)``
    :class:`numpy.random.SeedSequence` namespace, skipping any in ``avoid``.

    Same ``(base_seed, salt, count, avoid)`` ⇒ same seeds (the reproducibility contract);
    the distinct ``salt`` keeps this stream decorrelated from other roles' streams."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    seq = np.random.SeedSequence([int(base_seed), int(salt)])
    pool = seq.generate_state(max(count * 8, 16), dtype=np.uint32)
    seen = set(avoid)
    drawn: list[int] = []
    for value in pool:
        candidate = int(value)
        if candidate not in seen:
            seen.add(candidate)
            drawn.append(candidate)
            if len(drawn) == count:
                return tuple(drawn)
    raise RuntimeError(  # pragma: no cover - the pool is 8x oversized, exhaustion is unreachable
        f"could not draw {count} distinct seeds from the {salt:#x} namespace"
    )


def partition(base_seed: int, n_train: int, n_eval: int) -> HeldOutSplit:
    """Build a :class:`HeldOutSplit` with disjoint training and held-out seed sets.

    The two sets are drawn from **separate** :class:`numpy.random.SeedSequence` namespaces
    (distinct salts, both decorrelated from ``derive_seeds``'s slice stream), and the held-out
    draw explicitly avoids every training seed — so the split is disjoint by construction and
    reproducible from ``(base_seed, n_train, n_eval)`` (conventions.md §11)."""
    train = _draw_distinct(base_seed, _TRAIN_SEED_SALT, n_train, avoid=frozenset())
    held_out = _draw_distinct(base_seed, _HELD_OUT_SEED_SALT, n_eval, avoid=frozenset(train))
    return HeldOutSplit(train_seeds=frozenset(train), held_out_seeds=held_out)
