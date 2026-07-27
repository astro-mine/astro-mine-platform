"""The solver registry — name → backend factory, built-in or plugin (RM-P1-ALLOC-02).

The dispatch table :class:`~astro_mine.allocate.AllocationPlanner` resolves a backend id through
(allocate.md §3, ``registry`` / §11: solver backends are plugins behind one strategy). A backend
reaches it one of two ways, and :func:`resolve_solver` treats them identically:

* **Built-in** — CP-SAT and the trivial stub, seeded in :data:`_LOADERS` so the base package
  works from a raw checkout with nothing installed (CX-LOCAL);
* **Plugin** — any distribution advertising :data:`SOLVER_ENTRY_POINT_GROUP`, discovered through
  ``importlib.metadata`` (conventions.md §7: in-process plugins use Python entry points).

The *public* plugin surface is still the Core manifest a solver advertises itself through; this
is the in-process map from the id recorded in a plan's provenance to the
:class:`~astro_mine.allocate.solvers.base.Solver` factory that produces it. What changed is that
a third party can reach the table without editing it — the extension point allocate.md §11
advertises is now actually open.

**Laziness is a feature.** Listing backends never imports one: :func:`known_backends` reads
entry-point *names* from installed metadata and never calls ``load()``. Importing Allocate
therefore never requires OR-Tools, and never pays for whatever plugins happen to be installed on
the machine — the import happens only in :func:`resolve_solver`, for the one id asked for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, cast

from astro_mine.allocate.solvers._common import Pair
from astro_mine.core.messages.enums import TaskKind

if TYPE_CHECKING:
    from astro_mine.allocate.solvers.base import Solver

__all__ = [
    "CPSAT_BACKEND",
    "SOLVER_ENTRY_POINT_GROUP",
    "TRIVIAL_STUB_BACKEND",
    "available_backends",
    "known_backends",
    "resolve_solver",
]

#: The CP-SAT backend id (the primary engine; RM-P1-ALLOC-02).
CPSAT_BACKEND = "cp-sat"

#: The no-dependency greedy backend id (the local no-solver path).
TRIVIAL_STUB_BACKEND = "trivial-stub"

#: The entry-point group a third-party solver backend advertises itself under. The entry point's
#: **name** is the backend id (the string recorded in a plan's provenance); its value resolves to
#: a :data:`SolverFactory` — the same shape a built-in loader returns, so one
#: :func:`resolve_solver` call handles both.
#:
#: .. code-block:: toml
#:
#:     [project.entry-points."astro_mine.allocate.solvers"]
#:     my-solver = "my_pkg.solver:MySolver"
SOLVER_ENTRY_POINT_GROUP = "astro_mine.allocate.solvers"

#: A backend factory: builds a Solver from the plan-decode context (task kinds + durations).
SolverFactory = Callable[..., "Solver"]


def _cpsat_factory() -> SolverFactory:
    from astro_mine.allocate.solvers.cpsat import CpSatSolver

    return CpSatSolver


def _trivial_factory() -> SolverFactory:
    from astro_mine.allocate.solvers.trivial import TrivialStubSolver

    return TrivialStubSolver


#: The known backend ids → a loader that lazily imports the factory (so OR-Tools stays optional).
_LOADERS: dict[str, Callable[[], SolverFactory]] = {
    CPSAT_BACKEND: _cpsat_factory,
    TRIVIAL_STUB_BACKEND: _trivial_factory,
}


def _advertised() -> dict[str, EntryPoint]:
    """Plugin backends advertised in this environment, by id — **nothing is loaded**.

    Reading installed metadata is what keeps :func:`known_backends` cheap and import-free: a
    machine with ten solver plugins installed pays ten dictionary entries, not ten imports."""
    return {ep.name: ep for ep in entry_points(group=SOLVER_ENTRY_POINT_GROUP)}


def _describe(entry: EntryPoint) -> str:
    """Name a plugin's provider precisely enough to act on — distribution *and* target."""
    dist = getattr(entry.dist, "name", None)
    return f"{entry.value!r} (from {dist!r})" if dist else repr(entry.value)


def _load_factory(name: str) -> SolverFactory:
    """Resolve one backend id to its factory — built-in or plugin, one path.

    Raises :class:`ValueError` for an unknown id or a built-in/plugin id collision, and lets the
    underlying :class:`ImportError` escape for a backend whose dependency is missing."""
    builtin = _LOADERS.get(name)
    advertised = _advertised().get(name)
    if builtin is not None:
        if advertised is not None:
            # Never silently let a third party take over the primary engine's id: which solver
            # produced a plan is provenance (RM-P1-ALLOC-07), so an ambiguous id is a hard error
            # naming both claimants, not a precedence rule the user has to know.
            raise ValueError(
                f"solver backend id {name!r} is claimed by both the built-in backend and the "
                f"plugin {_describe(advertised)}; rename the plugin's entry point"
            )
        return builtin()
    if advertised is None:
        raise ValueError(f"unknown solver backend {name!r}; known backends: {known_backends()}")
    factory = advertised.load()
    if not callable(factory):
        raise TypeError(
            f"solver backend {name!r} entry point {_describe(advertised)} is not callable; it "
            f"must resolve to a factory building a Solver"
        )
    return cast("SolverFactory", factory)


def backend_provider(name: str) -> str | None:
    """Identify the distribution providing plugin backend ``name``, for a plan's provenance.

    ``None`` for a built-in (the caller pins its own solver version) and for an id no plugin
    advertises. Best-effort by construction: a plugin may advertise no resolvable distribution (a
    local entry point, a namespace package), and an unidentifiable provider is reported honestly
    rather than silently attributed to Allocate itself (RM-P1-ALLOC-07)."""
    if name in _LOADERS:
        return None
    entry = _advertised().get(name)
    if entry is None:
        return None
    dist_name = getattr(entry.dist, "name", None)
    dist_version = getattr(entry.dist, "version", None)
    if dist_name and dist_version:
        return f"{dist_name} {dist_version}"
    return f"plugin {entry.value}"


def known_backends() -> tuple[str, ...]:
    """Every backend id the registry knows — built-ins plus advertised plugins — whether or not
    its dependency is installed.

    Never imports a backend, so a broken or heavyweight plugin cannot make listing fail or slow."""
    return tuple(sorted({*_LOADERS, *_advertised()}))


def available_backends() -> tuple[str, ...]:
    """The backend ids that actually resolve in this environment (a subset of
    :func:`known_backends`).

    This one *does* import, since importability is the question being asked. Any failure — a
    missing dependency, a plugin that raises on load, an id collision — excludes that backend
    rather than propagating, so one broken plugin cannot deny the list to every other backend."""
    available: list[str] = []
    for name in known_backends():
        try:
            _load_factory(name)
        except Exception:  # a probe: any failure means "not available", never a crash
            continue
        available.append(name)
    return tuple(sorted(available))


def resolve_solver(
    name: str,
    *,
    task_kinds: Mapping[str, TaskKind],
    durations: Mapping[Pair, float] | None = None,
) -> Solver:
    """Build the backend named ``name``, wired with the plan-decode context.

    Resolves built-in and plugin backends through one path. Raises :class:`ValueError` for an
    unknown backend id or an id claimed by both a built-in and a plugin, and a clear
    :class:`ImportError` when a known backend's solver dependency (OR-Tools, for ``"cp-sat"``) is
    not installed.
    """
    try:
        factory = _load_factory(name)
    except ImportError as exc:  # pragma: no cover - only hit in an environment without OR-Tools
        raise ImportError(
            f"solver backend {name!r} requires an optional dependency that is not installed "
            f"({exc}); install it or use the {TRIVIAL_STUB_BACKEND!r} backend"
        ) from exc
    return factory(task_kinds=task_kinds, durations=durations)
