# SPDX-License-Identifier: Apache-2.0
"""The OMPL + FCL motion backend (RM-P1-MIND-03) — the ``[native]`` extra.

The production realization of the geometric half of TAMP that mind.md §4 names: **OMPL**
sampling-based planning (RRT* / PRM* / BIT* — asymptotically optimal, unlike the reference RRT's
first-feasible-path) over an **FCL** collision check, behind the same
:class:`~astro_mine.mind.tamp.motion.protocol.MotionPlanner` contract the pure-Python reference
fills. Swapping the two is a stack-spec edit, not a code change (principle 2).

**Geometry.** The ground plane is an OMPL ``RealVectorStateSpace(2)`` bounded by ``bound_m``. Each
:class:`~astro_mine.mind.tamp.motion.reference.Obstacle` (a circular keep-out — the Guard/Worlds
keep-out primitive) becomes an ``fcl.Sphere`` collision object; the agent is a sphere of
``agent_radius_m``, so clearance is a real swept-body check rather than the reference's
point-to-segment distance. OMPL's discrete motion validator interpolates between states, so a
whole edge — not just its endpoints — is checked.

**Determinism (read this before trusting a seed).** OMPL owns a **process-global** RNG that can
only be seeded once, *before any sampling happens* (``ompl.util.RNG.setSeed`` errors afterwards).
The adapter therefore seeds it at first use from the caller's :class:`random.Random` and cannot
honour a *different* seed later in the same process. Sampling planners are also
wall-clock-budgeted (``solve_time_s``), so a loaded machine can yield a different path. Both
facts are why this backend's manifest declares ``determinism_class: tolerance``, why its tests
assert *path validity* (endpoints + clearance) rather than byte-equality, and why the bit-exact
reference RRT — not this — is the CI-tested default and the golden-trace baseline
(conventions.md §11).

**Degrade-not-collapse.** A solve that finds no exact path returns the direct ``[goal]`` — a
defined best-effort result, never an exception (principle 4), matching the reference's contract.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from importlib import resources
from random import Random
from typing import Any

from astro_mine.core.registry.loader import load_manifest
from astro_mine.core.registry.model import PluginManifest
from astro_mine.core.registry.tier import TierPlugin
from astro_mine.mind.tamp.motion.reference import Obstacle, Point

__all__ = ["OmplMotionPlanner", "ompl_tamp_plugin"]

#: The OMPL geometric planners exposed by name (mind.md §4 names RRT*, PRM*, BIT*).
_PLANNERS = ("rrtstar", "prmstar", "bitstar")

#: OMPL's RNG is process-global and seedable exactly once, before any sampling. Guard the
#: one-shot seeding so concurrent planners in one process do not race on it.
_SEED_LOCK = threading.Lock()
_SEEDED = False


class MotionPlanningError(Exception):
    """Raised when the ``[native]`` extra (OMPL / FCL) is not installed."""


def _seed_once(seed: int) -> None:
    """Seed OMPL's global RNG the first time any planner runs in this process."""
    global _SEEDED
    from ompl import util as ou

    with _SEED_LOCK:
        if _SEEDED:
            return
        ou.RNG.setSeed(seed)
        _SEEDED = True


class OmplMotionPlanner:
    """A sampling-based motion planner over OMPL with FCL collision checking.

    ``planner`` selects the OMPL algorithm (``rrtstar`` / ``prmstar`` / ``bitstar``);
    ``solve_time_s`` is the anytime budget (an optimizing planner keeps improving until it
    expires); ``agent_radius_m`` inflates the keep-outs by the agent's own body.
    """

    def __init__(
        self,
        *,
        planner: str = "rrtstar",
        solve_time_s: float = 1.0,
        bound_m: float = 64.0,
        agent_radius_m: float = 0.25,
    ) -> None:
        if planner not in _PLANNERS:
            raise ValueError(f"unknown OMPL planner {planner!r}; expected one of {_PLANNERS}")
        self._planner = planner
        self._solve_time_s = solve_time_s
        self._bound_m = bound_m
        self._agent_radius_m = agent_radius_m

    def plan(
        self, start: Point, goal: Point, obstacles: Sequence[Obstacle], rng: Random
    ) -> list[Point]:
        """A collision-free waypoint path from ``start`` to ``goal`` (see module docstring)."""
        try:
            from ompl import base as ob
            from ompl import geometric as og
            from ompl import util as ou
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MotionPlanningError(
                "the OMPL motion backend needs the [native] extra: "
                "pip install 'astro-mine-platform[mind-native]'"
            ) from exc

        ou.setLogLevel(ou.LogLevel.LOG_ERROR)  # OMPL is chatty; keep stdout the decision trace
        _seed_once(rng.getrandbits(32))

        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-self._bound_m)
        bounds.setHigh(self._bound_m)
        space.setBounds(bounds)

        is_clear = self._collision_checker(obstacles)
        setup = og.SimpleSetup(space)
        setup.setStateValidityChecker(lambda state: is_clear(state[0], state[1]))

        start_state = space.allocState()
        start_state[0], start_state[1] = start
        goal_state = space.allocState()
        goal_state[0], goal_state[1] = goal
        setup.setStartAndGoalStates(start_state, goal_state)
        setup.setPlanner(self._algorithm(og, setup.getSpaceInformation()))

        setup.solve(self._solve_time_s)
        if not setup.haveExactSolutionPath():
            return [goal]  # defined best-effort result, never a failure (principle 4)
        setup.simplifySolution()
        path = setup.getSolutionPath()
        waypoints = [
            (path.getState(i)[0], path.getState(i)[1]) for i in range(path.getStateCount())
        ]
        return waypoints[1:]  # drop the start node; the caller already sits there

    def _algorithm(self, og: Any, space_information: Any) -> Any:
        factories = {
            "rrtstar": og.RRTstar,
            "prmstar": og.PRMstar,
            "bitstar": og.BITstar,
        }
        return factories[self._planner](space_information)

    def _collision_checker(self, obstacles: Sequence[Obstacle]) -> Any:
        """An ``(x, y) -> bool`` clearance test over ``obstacles``, backed by FCL."""
        try:
            import fcl
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MotionPlanningError(
                "the OMPL motion backend needs the [native] extra (python-fcl): "
                "pip install 'astro-mine-platform[mind-native]'"
            ) from exc

        keep_outs = [
            fcl.CollisionObject(
                fcl.Sphere(obstacle.radius),
                fcl.Transform([obstacle.center[0], obstacle.center[1], 0.0]),
            )
            for obstacle in obstacles
        ]
        agent_shape = fcl.Sphere(max(self._agent_radius_m, 1e-6))
        request = fcl.CollisionRequest()

        def is_clear(x: float, y: float) -> bool:
            agent = fcl.CollisionObject(agent_shape, fcl.Transform([x, y, 0.0]))
            return not any(
                fcl.collide(agent, keep_out, request, fcl.CollisionResult())
                for keep_out in keep_outs
            )

        return is_clear


def _manifest(filename: str) -> PluginManifest:
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("manifests", filename)
        .read_text(encoding="utf-8")
    )
    return load_manifest(text).manifest


def _params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "planner" in params:
        out["planner"] = str(params["planner"])
    if "solve_time_s" in params:
        out["solve_time_s"] = float(params["solve_time_s"])
    if "bound_m" in params:
        out["bound_m"] = float(params["bound_m"])
    if "agent_radius_m" in params:
        out["agent_radius_m"] = float(params["agent_radius_m"])
    return out


def ompl_tamp_plugin() -> TierPlugin:
    """Provider for the TAMP tier with OMPL/FCL motion behind it (entry point).

    Only the *motion* backend changes: the symbolic task selection and the PDDLStream-style
    interleaving stay the reference TAMP planner's, which is the point of the ``MotionPlanner``
    seam. The factory defers construction (and the ``ompl``/``fcl`` imports) until a stack binds
    this plugin.
    """
    from astro_mine.mind.tamp.reference import ReferenceTampPlanner

    return TierPlugin(
        manifest=_manifest("ompl_tamp.yaml"),
        factory=lambda params: ReferenceTampPlanner(motion=OmplMotionPlanner(**_params(params))),
    )
