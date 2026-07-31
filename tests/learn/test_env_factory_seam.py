"""The ``--env-factory`` resolution seam (#24).

A producer that must not depend on Learn cannot hand over a :class:`SwarmEnv`, and a Core
``Environment`` alone is not enough to build one — the per-agent spaces come from the SADF. So the
seam accepts the Core-typed ``(Environment, assets)`` pair as well, and normalizes here rather
than at the nine ``env_factory()`` call sites downstream.

These tests use Learn's own fakes as the stand-in producer: ``FakeSwarmWorld`` is a Core
``Environment`` and ``build_assets()`` returns Core ``Asset``\\ s, so the seam is exercised end to
end **without astro-mine-sim installed** — which is the point, since Learn must never depend on it.
"""

from __future__ import annotations

import pickle
import sys

import pytest

from astro_mine.core.sadf.model import Asset
from astro_mine.learn.envs import SwarmEnv, make_swarm_env
from astro_mine.learn.train.run import resolve_env_factory
from tests.learn.fakes import FakeSwarmWorld, build_assets

_THIS = "tests.learn.test_env_factory_seam"


# --- module-level factories: resolvable by dotted path, and picklable ------------------


def make_pair() -> tuple[FakeSwarmWorld, dict[str, Asset]]:
    """A Learn-free producer: Core types only, exactly what a simulator would hand over."""
    return FakeSwarmWorld(), build_assets()


def make_swarm_env_directly() -> SwarmEnv:
    """A producer that already depends on Learn and builds the wrapper itself."""
    return make_swarm_env(FakeSwarmWorld(), build_assets())


def make_wrong_shape() -> str:
    return "not an env"


not_callable = object()


# --- both accepted shapes -------------------------------------------------------------


def test_a_core_typed_pair_is_wrapped_into_a_swarm_env() -> None:
    """The shape that lets a producer stay Learn-free."""
    env = resolve_env_factory(f"{_THIS}:make_pair")()
    assert isinstance(env, SwarmEnv)
    assert set(env.possible_agents) == set(build_assets())


def test_a_swarm_env_factory_still_works_unchanged() -> None:
    """The pre-existing shape must not regress."""
    env = resolve_env_factory(f"{_THIS}:make_swarm_env_directly")()
    assert isinstance(env, SwarmEnv)


def test_both_shapes_yield_equivalent_environments() -> None:
    direct = resolve_env_factory(f"{_THIS}:make_swarm_env_directly")()
    wrapped = resolve_env_factory(f"{_THIS}:make_pair")()
    assert list(direct.possible_agents) == list(wrapped.possible_agents)
    assert direct.observation_spaces.keys() == wrapped.observation_spaces.keys()


def test_each_call_builds_a_fresh_env() -> None:
    """The factory contract is 'builds a fresh env', not 'returns a shared one'."""
    factory = resolve_env_factory(f"{_THIS}:make_pair")
    assert factory() is not factory()


# --- the Ray constraint ---------------------------------------------------------------


def test_the_resolved_factory_is_picklable() -> None:
    """KubeRayExecutor ships the factory to each rollout worker.

    The adapter holds the dotted *string* and re-resolves, so it survives the hop. A closure over
    the resolved target would not — and the failure would only ever appear on the distributed
    path, never in a local run.
    """
    factory = resolve_env_factory(f"{_THIS}:make_pair")
    revived = pickle.loads(pickle.dumps(factory))
    assert isinstance(revived(), SwarmEnv)


# --- actionable failures, never a bare traceback --------------------------------------


def test_a_malformed_dotted_path_is_rejected_eagerly() -> None:
    with pytest.raises(ValueError, match="module:attr"):
        resolve_env_factory("no_colon_here")


def test_an_uninstalled_producer_names_the_package_to_install() -> None:
    """A missing producer is the common first failure; it must not surface as an ImportError.

    Deliberately uses a module that cannot exist rather than ``astro_mine.sim``: Learn does not
    depend on Sim, but a developer may well have it installed alongside, and a test that asserts
    a real package is *absent* passes or fails on the state of the machine.
    """
    factory = resolve_env_factory("astro_mine_no_such_producer.reference:make_env")
    with pytest.raises(ValueError, match="astro-mine-platform"):
        factory()


def test_a_missing_attribute_names_the_module_and_the_attribute() -> None:
    factory = resolve_env_factory(f"{_THIS}:no_such_attribute")
    with pytest.raises(ValueError, match="no_such_attribute"):
        factory()


def test_a_non_callable_target_is_rejected() -> None:
    factory = resolve_env_factory(f"{_THIS}:not_callable")
    with pytest.raises(TypeError, match="not callable"):
        factory()


def test_an_unexpected_return_shape_names_both_accepted_shapes() -> None:
    factory = resolve_env_factory(f"{_THIS}:make_wrong_shape")
    with pytest.raises(TypeError, match="SwarmEnv or an"):
        factory()


# --- the waist-purity claim, asserted rather than commented ---------------------------


def test_learn_never_imports_sim() -> None:
    """AC4. The claim was previously only a comment (``tests/algos/test_registry_contract.py``).

    Static *and* dynamic: no Learn source names the module, and importing the training entrypoint
    does not pull it in. The static half is what catches a lazy, function-local import that a
    ``sys.modules`` check would miss because the function was never called.
    """
    import astro_mine.learn.train.run  # noqa: F401

    assert "astro_mine.sim" not in sys.modules

    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src" / "astro_mine" / "learn"
    offenders = [
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if "import astro_mine.sim" in path.read_text(encoding="utf-8")
        or "from astro_mine.sim" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"Learn must not import Sim (conventions.md §1.1): {offenders}"
