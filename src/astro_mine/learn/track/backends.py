# SPDX-License-Identifier: Apache-2.0
"""Experiment-tracking backends — MLflow default, W&B optional (learn.md §4, §11).

learn.md §11: *"**MLflow default** (OSS, self-host), **W&B optional**"*; learn.md §4: "Runs link
to Bench results and Hub artifacts by content hash."

The backend is a **seam**, not a dependency: :class:`TrackingBackend` is a five-method Protocol,
:class:`MlflowBackend` is the default realization behind the optional ``[mlflow]`` extra (a
**lazy** ``import mlflow``, so nothing here needs MLflow to import), and
:class:`InMemoryBackend` is the dependency-free realization used off-line, in tests, and by
anyone who just wants the provenance record without a tracking server. A W&B backend is the same
five methods.

This module is Torch-free and MLflow-free at import time — it is the tracking surface every
Learn run can reach for, whether or not a server exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["InMemoryBackend", "MlflowBackend", "TrackingBackend"]


@runtime_checkable
class TrackingBackend(Protocol):
    """Where a tracked run's params, metrics, and artifacts go (learn.md §11).

    Deliberately minimal — params once, metrics per step, JSON artifacts, tags — because that is
    all Learn's reproducibility contract needs, and a small surface is one any backend (MLflow,
    W&B, a file, a test double) can satisfy honestly."""

    def start_run(self, name: str | None, tags: Mapping[str, str]) -> str:
        """Open a run; return its backend-assigned id."""
        ...

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record the run's immutable inputs (config, seeds, versions) — written once."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Record one step's scalar metrics (the learning / eval curves)."""
        ...

    def log_dict(self, payload: Mapping[str, Any], artifact: str) -> None:
        """Attach a JSON artifact (the full configs, the provenance record, the curve manifest)."""
        ...

    def end_run(self) -> None:
        """Close the run."""
        ...


@dataclass
class InMemoryBackend:
    """A dependency-free :class:`TrackingBackend` that just remembers everything.

    The default when no tracking server is configured — a tier-1 run still gets its full,
    inspectable provenance record without MLflow, Postgres, or a network (learn.md §7 tier 1
    "MUST always work"). Also the deterministic test double for the tracking surface."""

    run_id: str = "in-memory"
    name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    #: Every logged metric point as ``(key, value, step)`` — the curve, replayable.
    metrics: list[tuple[str, float, int]] = field(default_factory=list)
    artifacts: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    ended: bool = False

    def start_run(self, name: str | None, tags: Mapping[str, str]) -> str:
        self.name = name
        self.tags.update(tags)
        return self.run_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        self.metrics.extend((key, float(value), step) for key, value in metrics.items())

    def log_dict(self, payload: Mapping[str, Any], artifact: str) -> None:
        self.artifacts[artifact] = dict(payload)

    def end_run(self) -> None:
        self.ended = True

    # --- inspection -----------------------------------------------------------------

    def curve(self, metric: str) -> list[float]:
        """The recorded series for one metric, in step order."""
        points = sorted(
            ((step, value) for key, value, step in self.metrics if key == metric),
            key=lambda point: point[0],
        )
        return [value for _step, value in points]


class MlflowBackend:
    """The **MLflow** :class:`TrackingBackend` — Learn's default tracking store (learn.md §11).

    MLflow is **not** a hard dependency: the ``import mlflow`` is lazy, in ``__init__``, so this
    raises loudly (``ImportError``) only if the backend is actually constructed without the
    optional ``[mlflow]`` extra installed. Nothing else in Learn imports it.

    Unlike a bare ``mlflow.start_run()`` context, this holds the run open across the whole
    training loop (:meth:`start_run` … :meth:`end_run`), which is what lets a
    :class:`~astro_mine.learn.track.run.TrackedRun` stream per-iteration curves into the *same*
    run the config and provenance were logged to."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment: str | None = None,
    ) -> None:
        # Lazy on purpose: mlflow is the optional [mlflow] extra, never a hard dep.
        import mlflow

        self._mlflow = mlflow
        self._active: Any = None
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment is not None:
            mlflow.set_experiment(experiment)

    def start_run(self, name: str | None, tags: Mapping[str, str]) -> str:
        self._active = self._mlflow.start_run(run_name=name, tags=dict(tags))
        run_id: str = self._active.info.run_id
        return run_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        self._mlflow.log_metrics({key: float(v) for key, v in metrics.items()}, step=step)

    def log_dict(self, payload: Mapping[str, Any], artifact: str) -> None:
        self._mlflow.log_dict(dict(payload), artifact)

    def end_run(self) -> None:
        self._mlflow.end_run()
        self._active = None
