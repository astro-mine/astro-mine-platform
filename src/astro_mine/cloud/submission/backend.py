"""Execution backends and the backend registry.

A :class:`Backend` runs a :class:`~astro_mine.cloud.submission.jobspec.JobSpec` against a
content-addressed store and returns a :class:`~astro_mine.cloud.submission.result.RunResult`.
Backends register under a name; :func:`~astro_mine.cloud.submission.submit` looks one up by
name, so swapping local for a future cluster backend is a name change, not a code fork
(``cloud.md`` §2 principle 2, §3). Phase 0 registers ``"local"`` (subprocess) and
``"docker"`` (container); the cluster backend is Phase 1.

Backlog: RM-P0-CLOUD-02 -- astro-mine-cloud#2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from astro_mine.cloud.runs.events import RunObserver
    from astro_mine.cloud.submission.jobspec import JobSpec
    from astro_mine.cloud.submission.result import RunResult
    from astro_mine.core.artifacts import ArtifactStore

__all__ = [
    "Backend",
    "get_backend",
    "register_backend",
    "registered_backends",
]


@runtime_checkable
class Backend(Protocol):
    """Runs a job against a store and returns its result.

    *observer* (default: none) carries the lifecycle-event publisher + status store threaded from
    :func:`~astro_mine.cloud.submission.submit`; a backend forwards it to
    :func:`~astro_mine.cloud.submission._run.execute` (RM-P1-CLOUD-06).
    """

    def run(
        self, job: JobSpec, *, store: ArtifactStore, observer: RunObserver | None = None
    ) -> RunResult: ...


_REGISTRY: dict[str, Backend] = {}


def register_backend(name: str, backend: Backend, *, replace: bool = False) -> None:
    """Register *backend* under *name*; raise on a duplicate unless *replace* is set."""
    if name in _REGISTRY and not replace:
        raise ValueError(f"backend {name!r} is already registered")
    _REGISTRY[name] = backend


def get_backend(name: str) -> Backend:
    """Return the backend registered under *name*, or raise with the known names."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(registered_backends()) or "(none)"
        raise ValueError(f"unknown backend {name!r}; registered: {known}") from None


def registered_backends() -> tuple[str, ...]:
    """Return the registered backend names, sorted."""
    return tuple(sorted(_REGISTRY))
