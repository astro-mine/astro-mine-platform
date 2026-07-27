"""The Optuna :class:`~.search.SearchBackend` adapter (RM-P1-STUDIO-02; studio.md §3, §11).

studio.md §11 names four external trade-study engines — Ax/BoTorch, **Optuna**, pymoo, and
Ray Tune — behind "one internal interface; pick per study". This module lands the first of
them. Optuna leads because it covers both halves of that recommendation with one dependency:
**NSGA-II** for cheap/large-population evolutionary MO and **MOTPE** (multi-objective
Tree-structured Parzen Estimator) for sample-efficient Bayesian-flavored MO, with a far
lighter footprint than Ax/BoTorch or Ray Tune.

The adapter is a **drop-in** ``SearchBackend``, not a parallel code path: ``TradeStudy`` picks
it by name (``backend="optuna"``) and the study loop, the fidelity ladder, the Pareto math, and
the provenance envelope are all unchanged.

Stateless replay — why the study is rebuilt every call
-----------------------------------------------------
Optuna's samplers are stateful across a ``Study``; :class:`~.search.SearchBackend` is a *pure*
function of ``(ranked, senses, seed)`` — the built-in ``NSGAII`` derives every proposal from
its arguments, and ``search.py`` promises "a re-run reproduces the same proposals". Holding a
long-lived ``Study`` on the backend instance would break that: a second ``evolve`` call with
identical arguments would see the first call's trials and answer differently.

It would also be *wrong*. The population handed to ``evolve`` is not the population Optuna
proposed: ``study._evaluate_generation`` drops Guard-rejected candidates, prunes the cheap
fidelity tier, and de-duplicates by candidate id. A long-lived study would accumulate a history
of trials it asked for but was never told about, and results it was told about but never asked
for. So each ``evolve`` **rebuilds** a study, replays the *observed* population into it with
:func:`optuna.trial.create_trial`, and asks for the next batch. "Here is what was really
evaluated; propose the next generation" is the honest model for this loop, and it keeps the
backend referentially transparent.

Determinism & seed policy
-------------------------
Studio's convention (``search.py``): every proposal is a deterministic function of an explicit
integer ``seed``, and ``run_trade_study`` advances it per generation (``base_seed + generation
+ 1``). Optuna carries its own internal RNG, so the two are reconciled as follows.

1. **The sampler is constructed per call, seeded with Studio's seed** — ``NSGAIISampler(seed=…)``
   / ``TPESampler(seed=…)``. Optuna's RNG is therefore re-derived from Studio's seed on every
   call rather than drifting with process-global state.
2. **No global RNG is touched.** Optuna seeds a sampler-local ``numpy`` ``RandomState``; this
   module never calls ``numpy.random.seed`` and never reads the global RNG, so an unrelated
   ``numpy`` consumer cannot perturb a proposal (and vice versa).
3. **The study is in-memory and fresh per call** (no storage URL, no study name), so no state
   survives across calls or processes to make a re-run diverge.
4. **The DoE seeding is Studio's, not Optuna's.** ``initial`` reuses the same ``scipy.stats.qmc``
   Sobol/LHS sampler the built-in NSGA-II uses (studio.md §11: "Sobol/LHS space-filling to seed
   the optimizer"), so both backends start from the same space-filling design and a study is
   comparable across backends.

The result: ``evolve(ranked, senses, space, seed=s, n=k)`` returns the same ``k`` vectors for the
same ``s`` — the STUDIO-02 CI determinism gate holds for ``backend="optuna"`` exactly as it does
for ``nsga2``. This is asserted directly in ``tests/test_search_optuna.py``.

A note on Optuna internals
--------------------------
Replaying a population into a *fresh* NSGA-II study requires telling the sampler which
generation the replayed trials belong to; Optuna keeps that in a private per-sampler system-attr
key (``BaseGASampler._get_generation_key()``). Without it the sampler sees an empty parent
population and **silently degrades to uniform random sampling** — the failure is invisible, so
it is pinned by a CI canary (``test_nsga2_breeds_children_from_elite_parents``) that asserts
children are genuinely bred from the elite front. The ``[optuna]`` extra is capped below the
next major accordingly.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import optuna

from ..models import DecisionSpace
from .search import Ranked, Vector, _doe_sample

__all__ = ["OptunaBackend"]

# Optuna logs a line per created study at INFO; a trade study creates one per generation.
# Scoped to this module's import, which only happens when an Optuna backend is instantiated.
optuna.logging.set_verbosity(optuna.logging.WARNING)

#: Decision variables are positional (one integer count per ``AssetChoice``), so parameters are
#: named by index rather than by ``sadf_ref`` — a space may legitimately carry the same asset
#: kind twice, and an index is total.
_PARAM_PREFIX = "x"

#: The samplers this adapter exposes, by the name they register under in the backend registry.
SAMPLERS = ("nsga2", "tpe")


Distributions = dict[str, optuna.distributions.BaseDistribution]


def _param(index: int) -> str:
    return f"{_PARAM_PREFIX}{index}"


def _distributions(bounds: Sequence[tuple[int, int]]) -> Distributions:
    """The decision space as Optuna integer distributions (the codec is lossless: one integer
    count per asset choice, exactly as ``designspace.encode`` defines it)."""
    return {
        _param(index): optuna.distributions.IntDistribution(low=low, high=high)
        for index, (low, high) in enumerate(bounds)
    }


def _directions(senses: Sequence[bool]) -> list[str]:
    """Studio's per-objective sense (``True`` = higher-is-better) as Optuna directions."""
    return ["maximize" if sense else "minimize" for sense in senses]


def _generation_key(sampler: optuna.samplers.BaseSampler) -> str | None:
    """The GA generation system-attr key, or ``None`` for a non-GA sampler (e.g. TPE).

    Private Optuna surface (``BaseGASampler._get_generation_key``) — see the module docstring:
    a replayed trial without this attr is invisible to NSGA-II's parent selection, which then
    falls back to random sampling *silently*. The CI canary is what keeps this honest.
    """
    getter = getattr(sampler, "_get_generation_key", None)
    return None if getter is None else str(getter())


class OptunaBackend:
    """An Optuna sampler behind Studio's one search interface.

    ``sampler`` selects ``"nsga2"`` (evolutionary MO) or ``"tpe"`` (MOTPE). ``doe`` selects the
    space-filling design that seeds generation 0. ``elite_fraction`` sets NSGA-II's population
    size as a fraction of the observed population: selection pressure comes from keeping only
    the best ``elite_fraction`` of the observed points as parents, mirroring the built-in
    ``NSGAII``'s top-half mating pool. ``crossover_prob``/``mutation_prob`` default to Optuna's
    own (``mutation_prob=None`` → ``1/n_params``).
    """

    def __init__(
        self,
        *,
        sampler: str = "nsga2",
        doe: str = "sobol",
        elite_fraction: float = 0.5,
        crossover_prob: float | None = None,
        mutation_prob: float | None = None,
    ) -> None:
        if sampler not in SAMPLERS:
            raise ValueError(f"unknown Optuna sampler {sampler!r}; have {list(SAMPLERS)}")
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError(f"elite_fraction must be in (0, 1], got {elite_fraction}")
        self._sampler = sampler
        self._doe = doe
        self._elite_fraction = elite_fraction
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob

    # -- SearchBackend ------------------------------------------------------ #

    def initial(self, space: DecisionSpace, *, seed: int, n: int) -> list[Vector]:
        """Seed the space with the same Sobol/LHS design the built-in NSGA-II uses, so a study
        is comparable across backends (studio.md §11)."""
        return _doe_sample(space.bounds(), self._doe, seed=seed, n=n)

    def evolve(
        self, ranked: Ranked, senses: Sequence[bool], space: DecisionSpace, *, seed: int, n: int
    ) -> list[Vector]:
        """Replay the observed population into a fresh seeded study and ask for ``n`` proposals."""
        if len(ranked) < 2:
            # Too small to breed — reseed the space (still deterministic in `seed`), matching
            # the built-in NSGA-II's fallback so the two backends share one contract.
            return _doe_sample(space.bounds(), self._doe, seed=seed, n=n)

        bounds = space.bounds()
        distributions = _distributions(bounds)
        study = self._replay(ranked, senses, distributions, seed=seed)

        children: list[Vector] = []
        for _ in range(n):
            # `ask` without a matching `tell`: the proposals have not been evaluated yet — that
            # is precisely what the study loop is about to do. NSGA-II breeds each child from
            # the elite parent front; MOTPE relies on `constant_liar` (below) so that a batch
            # asked in one go does not collapse onto a single argmax point.
            trial = study.ask(distributions)
            children.append(tuple(int(trial.params[_param(i)]) for i in range(len(bounds))))
        return children

    # -- internals ---------------------------------------------------------- #

    def _build_sampler(self, *, seed: int, population: int) -> optuna.samplers.BaseSampler:
        if self._sampler == "tpe":
            with warnings.catch_warnings():
                # `constant_liar` is flagged experimental by Optuna, but it is load-bearing here
                # (see below) — not an incidental default we could drop.
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                return optuna.samplers.TPESampler(
                    seed=seed,
                    n_startup_trials=0,  # the DoE already seeded the space; model it immediately
                    # Without `constant_liar` an un-told batch of asks collapses onto one point,
                    # because nothing in the study changes between successive asks.
                    constant_liar=True,
                )
        return optuna.samplers.NSGAIISampler(
            seed=seed,
            population_size=max(2, int(population * self._elite_fraction)),
            crossover_prob=self._crossover_prob if self._crossover_prob is not None else 0.9,
            mutation_prob=self._mutation_prob,
        )

    def _replay(
        self,
        ranked: Ranked,
        senses: Sequence[bool],
        distributions: Mapping[str, optuna.distributions.BaseDistribution],
        *,
        seed: int,
    ) -> optuna.study.Study:
        """A fresh in-memory study holding the evaluated population as generation-0 trials."""
        sampler = self._build_sampler(seed=seed, population=len(ranked))
        study = optuna.create_study(directions=_directions(senses), sampler=sampler)

        key = _generation_key(sampler)
        system_attrs: dict[str, Any] = {} if key is None else {key: 0}
        study.add_trials(
            [
                optuna.trial.create_trial(
                    params={_param(i): value for i, value in enumerate(vector)},
                    distributions=dict(distributions),
                    values=list(objectives),
                    system_attrs=dict(system_attrs),
                )
                for vector, objectives in ranked
            ]
        )
        return study
