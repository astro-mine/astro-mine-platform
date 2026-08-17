# SPDX-License-Identifier: Apache-2.0
"""Design-space exploration engine (RM-P1-STUDIO-02).

A pluggable multi-objective search that proposes, evaluates, and **Pareto-ranks**
``DesignCandidate``s (studio.md §3, §11). Studio computes nothing but the search and the
Pareto math: every candidate *evaluation* is delegated to the STUDIO-03 design loop on
content-addressed artifacts, under a multi-fidelity ladder (studio.md §8). Backends are
swappable behind one interface (:mod:`.search`): the built-in NSGA-II + Sobol/LHS DoE is the
working default, **Optuna** (NSGA-II / MOTPE) is the first external engine behind the optional
``[optuna]`` extra, and Ax/BoTorch·pymoo·Ray Tune stay documented seams
(``search.DEFERRED_BACKENDS``).

:class:`~.optuna_backend.OptunaBackend` is deliberately **not** re-exported here — importing it
imports ``optuna``, and this package must stay importable without the extra. Reach it through
``get_backend("optuna")``, or import :mod:`.optuna_backend` directly.
"""

from __future__ import annotations

from .encode import decode, encode
from .pareto import (
    crowding_distance,
    dominates,
    hypervolume,
    non_dominated_sort,
    pareto_front,
)
from .search import (
    DEFERRED_BACKENDS,
    NSGAII,
    MissingBackendExtra,
    SearchBackend,
    deferred_backends,
    get_backend,
    register_backend,
    registered_backends,
)
from .study import FidelityLadder, build_trade_study, run_trade_study

__all__ = [
    "DEFERRED_BACKENDS",
    "NSGAII",
    "FidelityLadder",
    "MissingBackendExtra",
    "SearchBackend",
    "build_trade_study",
    "crowding_distance",
    "decode",
    "deferred_backends",
    "dominates",
    "encode",
    "get_backend",
    "hypervolume",
    "non_dominated_sort",
    "pareto_front",
    "register_backend",
    "registered_backends",
    "run_trade_study",
]
