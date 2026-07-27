"""The solver entry-point group — a third-party backend without a PR to Allocate (G2.9).

``allocate.md`` §11 calls solver backends "plugins behind one strategy" and the README advertised
``solvers/`` as backend plugins, but the dispatch table was a hardcoded dict with no
``entry_points()`` call: a community solver needed a patch to Allocate to be reachable. These
prove the advertised extension point is now real, and — the half that is easy to miss — that
``AllocationPlanner`` actually *routes* to it, since the planner used to send every non-CP-SAT id
to its own internal greedy and still stamp the id into the plan's provenance.

The fake plugin delegates to :class:`TrivialStubSolver` on purpose: what is under test is the
registry seam and the planner routing, not a search algorithm.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from astro_mine.allocate import AllocationPlanner
from astro_mine.allocate.api.model import SolveBudget
from astro_mine.allocate.model.ir.model import AllocationIR
from astro_mine.allocate.solvers import registry as registry_mod
from astro_mine.allocate.solvers._common import Pair
from astro_mine.allocate.solvers.base import Incumbent, Solver
from astro_mine.allocate.solvers.registry import (
    CPSAT_BACKEND,
    TRIVIAL_STUB_BACKEND,
    available_backends,
    backend_provider,
    known_backends,
    resolve_solver,
)
from astro_mine.allocate.solvers.trivial import TrivialStubSolver
from astro_mine.core.messages.enums import TaskKind
from tests.allocate.factories import unwindowed_request

PLUGIN_BACKEND = "acme-solver"


class FakeThirdPartySolver:
    """A stand-in for a community backend: a distinct type reached only through the entry point."""

    def __init__(
        self,
        *,
        task_kinds: Mapping[str, TaskKind],
        durations: Mapping[Pair, float] | None = None,
    ) -> None:
        self._inner = TrivialStubSolver(task_kinds=task_kinds, durations=durations)

    def solve(
        self,
        ir: AllocationIR,
        budget: SolveBudget,
        *,
        hints: Mapping[str, float] | None = None,
    ) -> Iterator[Incumbent]:
        yield from self._inner.solve(ir, budget, hints=hints)


@dataclass
class _FakeDist:
    name: str
    version: str


class _FakeEntryPoint:
    """The importlib.metadata surface the registry reads: a name, a value, a dist, and load()."""

    def __init__(
        self,
        name: str,
        *,
        loads: Any = FakeThirdPartySolver,
        raises: Exception | None = None,
        dist: _FakeDist | None = None,
        value: str = "acme_solver.backend:AcmeSolver",
    ) -> None:
        self.name = name
        self.value = value
        self.dist = dist
        self._loads = loads
        self._raises = raises

    def load(self) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._loads


def _advertise(monkeypatch: pytest.MonkeyPatch, *entries: _FakeEntryPoint) -> None:
    """Advertise ``entries`` under the solver group, as an installed distribution would.

    Patches the module-scope ``entry_points`` symbol — the same mechanism Learn's registry tests
    use — so a plugin author can test discovery without installing anything."""
    monkeypatch.setattr(registry_mod, "entry_points", lambda *, group: list(entries))


# --- discovery ---------------------------------------------------------------------------


def test_a_plugin_backend_is_listed_alongside_the_builtins(monkeypatch) -> None:
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND))
    assert set(known_backends()) == {CPSAT_BACKEND, TRIVIAL_STUB_BACKEND, PLUGIN_BACKEND}


def test_listing_backends_never_loads_one(monkeypatch) -> None:
    """Laziness is the contract: a machine with plugins installed must not pay for them on import.

    A plugin whose ``load()`` explodes is still *listed* — because listing reads names from
    installed metadata and never calls ``load()``."""
    exploding = _FakeEntryPoint(PLUGIN_BACKEND, raises=RuntimeError("must not be loaded"))
    _advertise(monkeypatch, exploding)
    assert PLUGIN_BACKEND in known_backends()


def test_available_backends_reports_only_what_resolves(monkeypatch) -> None:
    _advertise(
        monkeypatch,
        _FakeEntryPoint(PLUGIN_BACKEND),
        _FakeEntryPoint("broken", raises=ImportError("no acme_solver")),
    )
    available = available_backends()
    assert PLUGIN_BACKEND in available
    assert "broken" not in available
    # AC5: one broken plugin must not deny the list to everyone else.
    assert TRIVIAL_STUB_BACKEND in available


def test_a_broken_plugin_does_not_break_the_listings(monkeypatch) -> None:
    """A plugin that raises something other than ImportError is still contained."""
    _advertise(monkeypatch, _FakeEntryPoint("hostile", raises=RuntimeError("boom")))
    assert "hostile" in known_backends()  # advertised, so listed
    assert "hostile" not in available_backends()  # but not usable
    assert set(available_backends()) >= {TRIVIAL_STUB_BACKEND}


# --- resolution --------------------------------------------------------------------------


def test_resolve_solver_builds_a_plugin_backend(monkeypatch) -> None:
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND))
    solver = resolve_solver(PLUGIN_BACKEND, task_kinds={})
    assert isinstance(solver, FakeThirdPartySolver)
    assert isinstance(solver, Solver)  # it satisfies the strategy contract


def test_a_plugin_may_not_hijack_a_builtin_id(monkeypatch) -> None:
    """AC4: an id claimed by both fails loudly, naming both claimants.

    Which solver produced a plan is provenance, so an ambiguous id is a hard error rather than a
    silent precedence rule the user would have to know about."""
    _advertise(
        monkeypatch,
        _FakeEntryPoint(CPSAT_BACKEND, dist=_FakeDist("acme-solver", "9.9.9")),
    )
    with pytest.raises(ValueError, match="claimed by both") as excinfo:
        resolve_solver(CPSAT_BACKEND, task_kinds={})
    message = str(excinfo.value)
    assert "built-in" in message
    assert "acme-solver" in message  # the other claimant is named, so it is actionable


def test_an_entry_point_that_is_not_callable_is_rejected(monkeypatch) -> None:
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND, loads="not-a-factory"))
    with pytest.raises(TypeError, match="not callable"):
        resolve_solver(PLUGIN_BACKEND, task_kinds={})


def test_unknown_backend_error_still_lists_what_is_known(monkeypatch) -> None:
    """AC6: the pre-existing error is unchanged — and now names plugins too."""
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND))
    with pytest.raises(ValueError, match="unknown solver backend"):
        resolve_solver("gurobi", task_kinds={})


# --- end-to-end through the planner ------------------------------------------------------


def test_the_planner_routes_to_a_plugin_backend(monkeypatch) -> None:
    """AC1: `AllocationPlanner(backend=<their-id>)` reaches the plugin, with no change to Allocate.

    Before this, any id that was not ``cp-sat`` fell through to Allocate's internal greedy."""
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND, dist=_FakeDist("acme-solver", "1.2.3")))
    allocation = AllocationPlanner(backend=PLUGIN_BACKEND).solve(unwindowed_request())
    assert allocation.plan  # a real, feasible plan came back
    assert allocation.provenance.backend == PLUGIN_BACKEND


def test_a_plugin_plan_records_its_provider_in_provenance(monkeypatch) -> None:
    """AC8: the recorded backend must identify what actually ran, not just repeat the id."""
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND, dist=_FakeDist("acme-solver", "1.2.3")))
    prov = AllocationPlanner(backend=PLUGIN_BACKEND).solve(unwindowed_request()).provenance
    assert prov.backend == PLUGIN_BACKEND
    assert "acme-solver 1.2.3" in prov.backend_version
    assert backend_provider(PLUGIN_BACKEND) == "acme-solver 1.2.3"
    assert backend_provider(CPSAT_BACKEND) is None  # a built-in pins its own solver version
    assert backend_provider("nobody-advertises-this") is None


def test_a_plugin_with_no_resolvable_distribution_is_recorded_honestly(monkeypatch) -> None:
    """A local entry point may advertise no distribution — say so rather than inventing one.

    The failure to avoid is attributing an unidentifiable third-party solver to Allocate itself."""
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND, dist=None))
    assert backend_provider(PLUGIN_BACKEND) == "plugin acme_solver.backend:AcmeSolver"
    prov = AllocationPlanner(backend=PLUGIN_BACKEND).solve(unwindowed_request()).provenance
    assert "acme_solver.backend:AcmeSolver" in prov.backend_version
    assert "astro-mine-allocate" in prov.backend_version  # the compiler is still pinned too


def test_a_plugin_backed_plan_is_deterministic(monkeypatch) -> None:
    """AC8: RM-P1-ALLOC-07 determinism must hold for a plugin backend too."""
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND))
    request = unwindowed_request()
    first = AllocationPlanner(backend=PLUGIN_BACKEND).solve(request)
    second = AllocationPlanner(backend=PLUGIN_BACKEND).solve(request)
    assert first.content_hash() == second.content_hash()


def test_an_unknown_backend_no_longer_yields_a_mislabelled_greedy_plan() -> None:
    """The regression this issue exposed.

    `AllocationPlanner(backend="totally-made-up-solver").solve(...)` used to return a *feasible*
    plan produced by Allocate's internal greedy, with `provenance.backend` reporting the backend
    that never ran — a plan attributing itself to a solver that does not exist."""
    with pytest.raises(ValueError, match="unknown solver backend"):
        AllocationPlanner(backend="totally-made-up-solver")
