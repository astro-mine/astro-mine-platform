"""Runner discovery — the ``astro_mine.bench.runners`` entry-point group (bench.md §2.2, §11).

A runner is an **in-process plugin** (conventions.md §7: *"in-process plugins use Python entry
points"*), discovered by name — never imported. Bench seeds the dependency-clean ``fixture`` runner
as a built-in (always available, no metadata refresh — the base ``score`` command is sacred,
conventions.md §7 tier 1 / CX-LOCAL) and overlays third-party runners discovered through the
entry-point group, so an injected Sim runner from ``astro-mine-sim[bench]`` resolves **without
importing** ``astro_mine.sim`` (conventions.md §1.1 — no private side-channels; the base package
stays core + pydantic, and ``Bench MUST NOT import Sim``, bench.md §2.2). Reference implementations
ship as replaceable examples (conventions.md §1.3); this mirrors Learn's registry — built-ins
seeded in code, entry points discovered on top.

A provider bundles the two runner protocols a benchmark drives — the scoring-path
:class:`~astro_mine.bench.baseline.EpisodeRunner` (``score``) and the harness
:class:`~astro_mine.bench.harness.Runner` (the determinism gate) — plus the ``runner_id`` recorded
on the produced artifact, so a single ``--runner`` selection threads through both. A provider MAY
also offer :class:`DefaultPolicyProvider` — the policy it wants scored when the caller names none.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

from astro_mine.bench.baseline._policy import BaselinePolicy
from astro_mine.bench.baseline._runner import (
    REFERENCE_EPISODE_RUNNER_ID,
    EpisodeRunner,
    reference_episode_runner,
)
from astro_mine.bench.harness import Runner, reference_runner
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.core.policy import Policy

__all__ = [
    "RUNNER_ENTRYPOINT_GROUP",
    "BenchRunnerProvider",
    "DefaultPolicyProvider",
    "RunnerNotAvailableError",
    "default_policy_for",
    "fixture_runner_provider",
    "load_runner_provider",
]

#: The setuptools entry-point group in-process runner plugins register into (conventions.md §7).
RUNNER_ENTRYPOINT_GROUP = "astro_mine.bench.runners"

#: Shown when ``--runner sim`` is requested but the Sim runner is not registered (CX-LOCAL: an
#: actionable install line, never a traceback). Mirrors Sim's own ``_BENCH_HINT``.
_SIM_INSTALL_HINT = (
    "the 'sim' runner is provided by astro-mine-sim; install it with "
    "`pip install 'astro-mine-sim[bench]'` (or `uv pip install 'astro-mine-sim[bench]'`), "
    "then fetch the anchor content so a Sim-backed run has a store to read"
)


@runtime_checkable
class BenchRunnerProvider(Protocol):
    """A named runner registered into :data:`RUNNER_ENTRYPOINT_GROUP`.

    ``runner_id`` is the identity stamped on a :class:`~astro_mine.bench.metrics.Scorecard` and the
    harness ``Result``. ``episode_runner`` returns the scoring-path
    :class:`~astro_mine.bench.baseline.EpisodeRunner`; ``harness_runner`` the determinism-gate
    :class:`~astro_mine.bench.harness.Runner`. ``store`` is the content store an engine-backed
    runner reads (a Sim bundle store, resolved by ``astro-mine-bench fetch``); the fixture ignores
    it. Typed as ``object`` so Bench never names a Sim type.
    """

    @property
    def runner_id(self) -> str: ...

    def episode_runner(self, store: object | None = None) -> EpisodeRunner: ...

    def harness_runner(self, store: object | None = None) -> Runner: ...


@runtime_checkable
class DefaultPolicyProvider(Protocol):
    """**Optional** companion to :class:`BenchRunnerProvider` — a runner's own default policy.

    A capability-aware policy has to read each asset's SADF (its capability tags and declared
    ``loads_by_mode``) to know what to command. Bench cannot supply that: a
    :class:`~astro_mine.bench.scenario.ResolvedScenario` carries content *hashes*, never
    materialized bundles, and Core's ``Policy`` protocol is handed observations, which carry no mode
    vocabulary. A runner that already resolves the content — Sim does — can build one; Bench asks
    for it by name and never learns what SADF is (astro-mine-sim#61).

    **Both arguments are load-bearing.** ``store`` is where the bundles live; ``spec`` is *which*
    of them this run pins. A provider given only a store would have to find each asset some other
    way — by OCI tag, in practice — and would then build a policy against a different asset version
    than the run scores, silently breaking content-addressing (CX-REPRO). The spec's
    ``content.fleet`` pins are the digests the episode actually runs on.

    Kept **separate from** :class:`BenchRunnerProvider` on purpose. ``load_runner_provider`` gates
    on ``isinstance(provider, BenchRunnerProvider)``, and a ``runtime_checkable`` Protocol matches
    on method *presence* — so folding this member into that Protocol would make every already-
    published provider (Sim's included) stop resolving. Providers opt in instead, and callers check
    with :func:`isinstance` and fall back.
    """

    def default_policy(self, spec: ScenarioSpec, store: object | None = None) -> Policy: ...


def default_policy_for(
    provider: BenchRunnerProvider, spec: ScenarioSpec, store: object | None = None
) -> Policy:
    """The policy ``provider`` wants scored on ``spec``, else :class:`BaselinePolicy`.

    The fallback is the honest floor, not a stand-in for a missing feature: a runner that has no
    opinion about what to score gets Bench's API-conformance baseline, exactly as before.
    """
    if isinstance(provider, DefaultPolicyProvider):
        return provider.default_policy(spec, store)
    return BaselinePolicy()


class _FixtureRunnerProvider:
    """The dependency-clean built-in — a deterministic trace fixture, not a physics engine.

    Seeded into :data:`_BUILTINS` so it always resolves, and needs no content store. It declares no
    ``default_policy``: the fixture path's baseline is :class:`BaselinePolicy`, reached through
    :func:`default_policy_for`'s fallback rather than restated here.
    """

    runner_id = REFERENCE_EPISODE_RUNNER_ID

    def episode_runner(self, store: object | None = None) -> EpisodeRunner:
        return reference_episode_runner

    def harness_runner(self, store: object | None = None) -> Runner:
        return reference_runner


#: The built-in ``fixture`` provider — always available, no installed-metadata dependency.
fixture_runner_provider = _FixtureRunnerProvider()

#: Runners seeded in code so the base ``score`` command works from a raw checkout (CX-LOCAL);
#: third-party runners (Sim's ``sim``) overlay via the entry-point group. Mirrors Learn's registry.
_BUILTINS: dict[str, BenchRunnerProvider] = {"fixture": fixture_runner_provider}

#: The runner ids this package provides itself, as names only.
#:
#: Public because a *scaffold* needs them and nothing else about the registry: `astro-mine
#: plugin new runner` warns when the id a user picked would shadow a built-in, and it should
#: not have to import the providers -- or reach into `_BUILTINS` -- to find that out. Exposing
#: the keys keeps the providers themselves private (astro-mine-cli#12).
BUILTIN_RUNNERS: frozenset[str] = frozenset(_BUILTINS)


class RunnerNotAvailableError(RuntimeError):
    """A ``--runner`` name is not registered — e.g. ``sim`` without ``astro-mine-sim[bench]``."""


class ScoringRefused(RuntimeError):
    """A runner declined to score, deliberately — part of the runner contract, not a failure.

    An engine-backed runner may find that scoring this scenario would produce a claim it cannot
    support: the canonical case is a pin that resolved by digest but rebuilt no provider, so the
    run would report metrics for content it never modelled (``astro-mine-sim#67``). A scorecard is
    a published claim, and there is no honest use for one made against a world that was never
    loaded — so the runner **raises** rather than returning a degraded trace, because a refusal
    that can be ignored will be.

    It is a distinct type so a caller can present it as an error a *user* can act on — the message
    names what is missing and which package supplies it — while a genuine engine bug keeps its
    traceback. Bench's own CLI does exactly that. Matching on message text would work today and rot
    tomorrow (``#79``).

    Raised by the runner and caught by whoever drives it; Bench never raises it itself.
    """


def load_runner_provider(name: str) -> BenchRunnerProvider:
    """Resolve a runner ``name`` to its provider — a built-in, else the entry-point group.

    Never imports ``astro_mine.sim`` (conventions.md §1.1): a third-party runner resolves by
    entry-point name only. Raises :class:`RunnerNotAvailableError` with an actionable message — the
    Sim install hint for ``sim``, otherwise the set of registered names — rather than a traceback
    (CX-LOCAL).
    """
    builtin = _BUILTINS.get(name)
    if builtin is not None:
        return builtin
    found = {ep.name: ep for ep in entry_points(group=RUNNER_ENTRYPOINT_GROUP)}
    entry = found.get(name)
    if entry is None:
        if name == "sim":
            raise RunnerNotAvailableError(_SIM_INSTALL_HINT)
        available = ", ".join(sorted({*_BUILTINS, *found})) or "none"
        raise RunnerNotAvailableError(f"unknown runner {name!r}; registered runners: {available}")
    provider = entry.load()
    if not isinstance(provider, BenchRunnerProvider):
        raise RunnerNotAvailableError(
            f"runner {name!r} entry point {entry.value!r} is not a BenchRunnerProvider"
        )
    return provider
