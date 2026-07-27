"""Pluggable multi-objective search (studio.md §3 ``designspace/search``, §4, §11).

**One internal interface** (:class:`SearchBackend`) with adapters selectable per study —
the study loop never changes when the backend does. Phase-1 ships a working built-in
**NSGA-II** (evolutionary MO) seeded by a **space-filling DoE** (Sobol / Latin-Hypercube,
via ``scipy.stats.qmc``), plus the first *external* engine of the four studio.md §11 names:
**Optuna** (NSGA-II + MOTPE), behind the optional ``[optuna]`` extra.

Seed policy
-----------
Everything here is seeded and deterministic: a proposal is a pure function of its arguments
and an explicit integer ``seed``, so a re-run reproduces it. The built-in ``NSGAII`` drives
evolution and the QMC samplers from ``numpy.random.default_rng(seed)``. An **external** backend
brings its own RNG, and the adapter's job is to reconcile it with this contract rather than let
it drift — see :mod:`.optuna_backend` ("Determinism & seed policy") for how Optuna's
sampler-local RNG is re-derived from Studio's seed on every call, touching no global state.
``run_trade_study`` advances the seed per generation (``base_seed + generation + 1``).

Registered vs. deferred
-----------------------
``optuna`` / ``optuna-nsga2`` / ``optuna-tpe`` are **registered** and instantiate lazily: the
name is always discoverable via :func:`registered_backends`, and only *using* one imports
``optuna``, so the base wheel keeps ``numpy``/``scipy`` as its only search dependencies. Without
the extra, :func:`get_backend` raises :class:`MissingBackendExtra` naming the install.

Ax/BoTorch, pymoo, and Ray Tune remain **deferred** — documented seams, not silent omissions.
:data:`DEFERRED_BACKENDS` records what each one is for and how it plugs in; asking
:func:`get_backend` for one raises a pointer rather than a bare "unknown backend". Each lands
the same way Optuna did: an adapter module implementing :class:`SearchBackend`, an optional
extra, and a :func:`register_backend` call with a lazy factory.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import qmc

from ..models import DecisionSpace
from .pareto import crowding_distance, non_dominated_sort

Vector = tuple[int, ...]
Ranked = Sequence[tuple[Vector, tuple[float, ...]]]


class MissingBackendExtra(ImportError):
    """A registered backend was requested whose optional extra is not installed."""

    def __init__(self, backend: str, extra: str) -> None:
        super().__init__(
            f"the {backend!r} search backend requires the optional [{extra}] extra; "
            f"install it with `uv sync --extra {extra}` "
            f"(or `pip install astro-mine-studio[{extra}]`)"
        )
        self.backend = backend
        self.extra = extra


#: The remaining studio.md §11 engines, deferred behind this same interface. Each entry is the
#: seam a future issue fills: an adapter module + an optional extra + a `register_backend` call,
#: exactly as `optuna_backend` did. Documented, not dropped (RM-P1-STUDIO-02).
DEFERRED_BACKENDS: Mapping[str, str] = {
    "ax": (
        "Ax/BoTorch — sample-efficient Bayesian MO (qNEHVI); studio.md §11's recommendation for "
        "*expensive* candidate evaluation, where each candidate is a full Sim rollout. Plugs in "
        "as an `[ax]` extra; the ask/tell replay in `optuna_backend` maps onto Ax's "
        "`AxClient.get_next_trial`/`complete_trial` one-for-one."
    ),
    "pymoo": (
        "pymoo — a reference implementation of the evolutionary MO family (NSGA-II/III, MOEA/D) "
        "for cross-checking the built-in `NSGAII` and for algorithms Optuna does not carry. "
        "Plugs in as a `[pymoo]` extra via its `Problem`/`minimize` API with a fixed seed."
    ),
    "ray-tune": (
        "Ray Tune — for when the *search itself* must scale across Cloud (studio.md §7, §11), "
        "rather than only the candidate evaluations (which already fan out through "
        "`orchestrate.cloud.CloudDispatcher`). Plugs in as a `[raytune]` extra; note it inverts "
        "control (Tune owns the loop), so the adapter drives a Tune run per generation."
    ),
}


@runtime_checkable
class SearchBackend(Protocol):
    """A multi-objective search strategy. ``initial`` seeds the space; ``evolve`` proposes
    the next batch from the evaluated population."""

    def initial(self, space: DecisionSpace, *, seed: int, n: int) -> list[Vector]: ...

    def evolve(
        self, ranked: Ranked, senses: Sequence[bool], space: DecisionSpace, *, seed: int, n: int
    ) -> list[Vector]: ...


def _doe_sample(
    bounds: Sequence[tuple[int, int]], method: str, *, seed: int, n: int
) -> list[Vector]:
    """Space-filling integer samples in the bounded box via a QMC sequence."""
    rng = np.random.default_rng(seed)
    dimension = len(bounds)
    sampler: qmc.QMCEngine
    if method == "sobol":
        sampler = qmc.Sobol(d=dimension, seed=rng)
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=dimension, seed=rng)
    else:  # pragma: no cover - guarded by the registry
        raise ValueError(f"unknown DoE method {method!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sobol balance warning for non-power-of-two n
        unit = sampler.random(n)
    low = np.array([lo for lo, _ in bounds])
    high = np.array([hi for _, hi in bounds])
    scaled = np.floor(low + unit * (high - low + 1)).astype(int)
    scaled = np.clip(scaled, low, high)
    return [tuple(int(value) for value in row) for row in scaled]


def _breed(
    a: Vector, b: Vector, bounds: Sequence[tuple[int, int]], rng: np.random.Generator
) -> Vector:
    """Uniform crossover + random-reset mutation, clamped to bounds."""
    child: list[int] = []
    for ai, bi, (lo, hi) in zip(a, b, bounds, strict=True):
        gene = ai if rng.random() < 0.5 else bi
        if rng.random() < 0.2:
            gene = int(rng.integers(lo, hi + 1))
        child.append(int(min(max(gene, lo), hi)))
    return tuple(child)


class NSGAII:
    """Built-in NSGA-II: non-dominated sort + crowding select the mating pool; uniform
    crossover + mutation produce the next generation. DoE-seeded and deterministic."""

    def __init__(self, *, doe: str = "sobol") -> None:
        self._doe = doe

    def initial(self, space: DecisionSpace, *, seed: int, n: int) -> list[Vector]:
        return _doe_sample(space.bounds(), self._doe, seed=seed, n=n)

    def evolve(
        self, ranked: Ranked, senses: Sequence[bool], space: DecisionSpace, *, seed: int, n: int
    ) -> list[Vector]:
        vectors = [vector for vector, _ in ranked]
        objectives = [objective for _, objective in ranked]
        if len(vectors) < 2:
            # Too small to breed — reseed the space (still deterministic in `seed`).
            return _doe_sample(space.bounds(), self._doe, seed=seed, n=n)

        order: list[int] = []
        for front in non_dominated_sort(objectives, senses):
            crowding = crowding_distance(objectives, front, senses)
            order.extend(sorted(front, key=lambda i: -crowding[i]))
        pool = [vectors[i] for i in order[: max(2, len(order) // 2)]]

        rng = np.random.default_rng(seed)
        bounds = space.bounds()
        children: list[Vector] = []
        while len(children) < n:
            a = pool[int(rng.integers(len(pool)))]
            b = pool[int(rng.integers(len(pool)))]
            children.append(_breed(a, b, bounds, rng))
        return children[:n]


_BACKENDS: dict[str, Callable[[], SearchBackend]] = {}


def register_backend(name: str, factory: Callable[[], SearchBackend]) -> None:
    """Register a search backend behind the one interface — the extension seam every optimizer
    plugs into (studio.md §3 "Extension points"). The factory is called on each
    :func:`get_backend`, so an adapter whose engine lives behind an optional extra can import it
    lazily and keep the base wheel free of the dependency."""
    _BACKENDS[name] = factory


def get_backend(name: str) -> SearchBackend:
    """Instantiate a registered backend by name.

    Raises :class:`MissingBackendExtra` if the backend is registered but its optional extra is
    not installed, and ``ValueError` for an unknown name — pointing at :data:`DEFERRED_BACKENDS`
    when the name is a known-but-deferred seam rather than a typo.
    """
    factory = _BACKENDS.get(name)
    if factory is None:
        if name in DEFERRED_BACKENDS:
            raise ValueError(
                f"search backend {name!r} is a documented but deferred seam "
                f"(RM-P1-STUDIO-02): {DEFERRED_BACKENDS[name]}"
            )
        raise ValueError(f"unknown search backend {name!r}; have {registered_backends()}")
    return factory()


def registered_backends() -> list[str]:
    """Every backend name :func:`get_backend` accepts (including those behind an optional extra —
    the name is discoverable even when the extra is absent; instantiating is what fails)."""
    return sorted(_BACKENDS)


def deferred_backends() -> Mapping[str, str]:
    """The documented, not-yet-implemented seams (see :data:`DEFERRED_BACKENDS`)."""
    return DEFERRED_BACKENDS


def _optuna(sampler: str) -> Callable[[], SearchBackend]:
    """A lazy factory for an Optuna sampler — `optuna` is imported only when the backend is
    actually instantiated, so it stays an optional extra."""

    def factory() -> SearchBackend:
        try:
            from .optuna_backend import OptunaBackend
        except ImportError as exc:  # the [optuna] extra is not installed
            raise MissingBackendExtra("optuna", "optuna") from exc
        return OptunaBackend(sampler=sampler)

    return factory


register_backend("nsga2", lambda: NSGAII(doe="sobol"))
register_backend("nsga2-lhs", lambda: NSGAII(doe="lhs"))
# The external engine (RM-P1-STUDIO-02). `optuna` is the friendly default (NSGA-II — the same
# algorithm family as the built-in, but Optuna's implementation); `optuna-tpe` is MOTPE.
register_backend("optuna", _optuna("nsga2"))
register_backend("optuna-nsga2", _optuna("nsga2"))
register_backend("optuna-tpe", _optuna("tpe"))
