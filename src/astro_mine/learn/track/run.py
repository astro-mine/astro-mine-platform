# SPDX-License-Identifier: Apache-2.0
"""The tracked run — config, seeds, provenance, and curves in one record (learn.md §3, §5).

The ``track/`` module of learn.md §3's tree: *"experiment tracking (MLflow default / W&B
option), **provenance capture**"*. Tracking here is not "nice dashboards" — it is the
**reproducibility contract** made concrete (learn.md §2.4: *"Reproducibility is non-negotiable.
Same scenario + same seed + same pinned environment ⇒ same learning curve. Every run records its
inputs, code version, lockfile, and seeds; results are content-addressed so Bench can re-derive
them"*).

:class:`TrackedRun` is that record:

- **captured at open** — the :class:`TrainConfig`, the :class:`CommsModelConfig`, the
  :class:`CurriculumSpec`, every seed, the toolchain versions, the code version, and the
  environment lockfile hash. Their **content hash** (:attr:`TrackedRun.run_hash`) is the
  reproducibility key: two runs with the same hash are the same experiment and MUST produce the
  same curve;
- **streamed during** — per-iteration training metrics and, at the end, the honest-eval
  :class:`~astro_mine.learn.eval.CurveTable` (comms-stress curves, seed variance) into the *same*
  run;
- **linked at close** — the produced policy's ONNX digests, so the tracked run and the
  :class:`~astro_mine.core.policy.PolicyPackage` Bench scores and Hub distributes are joined
  **by content hash** (learn.md §4: "Runs link to Bench results and Hub artifacts by content
  hash").

The backend is pluggable (:mod:`~astro_mine.learn.track.backends`): MLflow by default,
:class:`InMemoryBackend` when there is no server — so a tier-1 run still gets the full provenance
record with no infrastructure at all.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any

from astro_mine.core.hashing import content_hash_json
from astro_mine.learn.algos._contract import PolicyExport
from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.curriculum.spec import CurriculumSpec
from astro_mine.learn.envs.comms import CommsModelConfig
from astro_mine.learn.track.backends import InMemoryBackend, TrackingBackend

if TYPE_CHECKING:
    from astro_mine.learn.eval.aggregate import CurveTable

__all__ = ["RUN_RECORD_VERSION", "TrackedRun", "run_provenance", "tracked_run"]

#: Bumped when the shape of the captured run record changes.
RUN_RECORD_VERSION = "0.1.0"


def _toolchain() -> dict[str, str]:
    """The pinned toolchain versions, for the provenance record (best-effort, never fatal)."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    # astro-mine-learn/-core now ship from the consolidated platform distribution.
    for package in ("astro-mine-platform", "torch", "ray", "gymnasium", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover - depends on which extras are installed
            continue
    return versions


def run_provenance(
    config: TrainConfig,
    *,
    algorithm: str | None = None,
    comms: CommsModelConfig | None = None,
    curriculum: CurriculumSpec | None = None,
    env_lockfile: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The canonical, JSON-serializable provenance record of a training run.

    Everything needed to re-derive the run: the full training config (its ``seed`` included), the
    comms regime, the curriculum, the toolchain versions, and the environment lockfile hash
    (conventions.md §5). Deliberately a plain dict so its
    :func:`~astro_mine.core.hashing.content_hash_json` is stable and Bench can re-derive a
    leaderboard entry from it."""
    record: dict[str, Any] = {
        "record_version": RUN_RECORD_VERSION,
        "algorithm": algorithm,
        "seed": config.seed,
        "train_config": config.model_dump(mode="json"),
        "comms": None if comms is None else comms.model_dump(mode="json"),
        "curriculum": None if curriculum is None else curriculum.model_dump(mode="json"),
        "toolchain": _toolchain(),
        "env_lockfile": env_lockfile,
    }
    if extra:
        record["extra"] = dict(extra)
    return record


class TrackedRun:
    """One tracked training run: provenance in, curves through, artifact digests out.

    Use it as a context manager — the backend run is opened (and the immutable inputs logged) on
    entry and closed on exit, even if training raises::

        with TrackedRun(config, algorithm="mappo", comms=comms_cfg) as run:
            for step in range(config.iterations):
                run.log_iteration(trainer.train_iteration())
            run.log_export(trainer.export(), digests={"rover": "sha256:…"})

    With no ``backend`` it records into an :class:`InMemoryBackend` (no MLflow, no server, no
    network); pass :class:`~astro_mine.learn.track.backends.MlflowBackend` to mirror the same
    calls into MLflow."""

    def __init__(
        self,
        config: TrainConfig,
        *,
        algorithm: str | None = None,
        comms: CommsModelConfig | None = None,
        curriculum: CurriculumSpec | None = None,
        env_lockfile: str | None = None,
        backend: TrackingBackend | None = None,
        name: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.algorithm = algorithm
        self.backend: TrackingBackend = backend if backend is not None else InMemoryBackend()
        self.provenance = run_provenance(
            config,
            algorithm=algorithm,
            comms=comms,
            curriculum=curriculum,
            env_lockfile=env_lockfile,
        )
        #: The content address of the whole run record — THE reproducibility key (learn.md §2.4).
        #: Two runs with the same ``run_hash`` are the same experiment and must produce the same
        #: learning curve; CI's determinism gate is exactly that claim.
        self.run_hash: str = content_hash_json(self.provenance)
        self._name = name if name is not None else f"{algorithm or 'run'}-{config.seed}"
        self._tags = {"run_hash": self.run_hash, **(dict(tags) if tags else {})}
        if algorithm is not None:
            self._tags["algorithm"] = algorithm
        self.run_id: str | None = None
        self._step = 0
        #: Content-addressed identities of everything this run produced (ONNX graph digests),
        #: which is how the run joins to Bench scores and Hub artifacts (learn.md §4).
        self.artifact_digests: dict[str, str] = {}

    # --- lifecycle ------------------------------------------------------------------

    def start(self) -> TrackedRun:
        """Open the backend run and log the immutable inputs (config, seeds, provenance)."""
        self.run_id = self.backend.start_run(self._name, self._tags)
        self.backend.log_params(
            {
                "algorithm": self.algorithm or "",
                "seed": self.config.seed,
                "iterations": self.config.iterations,
                "rollout_steps": self.config.rollout_steps,
                "fidelity": self.config.fidelity,
                "use_rnn": self.config.use_rnn,
                "num_workers": self.config.num_workers,
                "run_hash": self.run_hash,
            }
        )
        # The *full* documents as artifacts — params are flat scalars, provenance is not.
        self.backend.log_dict(self.provenance, "run_provenance.json")
        return self

    def finish(self) -> None:
        """Close the backend run."""
        self.backend.end_run()

    def __enter__(self) -> TrackedRun:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.finish()

    # --- logging --------------------------------------------------------------------

    def log_iteration(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log one training iteration's metrics (reward, losses, entropy, env-steps/s).

        ``step`` defaults to a monotonically increasing counter, so the caller's loop needs no
        bookkeeping and the recorded curve is exactly the trainer's ``learning_curve()``."""
        index = self._step if step is None else step
        self.backend.log_metrics(metrics, step=index)
        self._step = index + 1

    def log_stage(self, stage: Any) -> None:
        """Record a curriculum promotion — the resolved
        :class:`~astro_mine.learn.curriculum.Stage` the run just moved onto.

        The stage lands as an **artifact**, not a metric: its provenance (the sampled comms
        regime, the world_provider draw, the stage's TrainConfig) is what makes that exact world
        re-derivable, and its content hash identifies it. The per-iteration ``curriculum_stage``
        *metric* — which stage each iteration ran in — rides the normal curve
        (:meth:`log_iteration`), so it is not duplicated here."""
        self.backend.log_dict(stage.provenance(), f"curriculum/stage_{stage.index}.json")

    def log_curve(self, table: CurveTable) -> None:
        """Log the honest-eval :class:`CurveTable` (comms-stress / seed-sweep curves).

        Supersedes the standalone ``MlflowSink`` for a *tracked* run: the curves land in the
        **same** backend run as the config and provenance, so the eval numbers and the run that
        produced them are one record rather than two that must be joined by hand."""
        self.backend.log_dict(table.manifest, "curve_manifest.json")
        self.backend.log_metrics({"curve_rows": float(len(table.rows))}, step=self._step)
        for index, row in enumerate(table.rows):
            prefix = f"{row.algorithm}.{row.stress_axis}"
            self.backend.log_metrics(
                {
                    f"{prefix}.episode_return": row.episode_return,
                    f"{prefix}.delivery_ratio": row.delivery_ratio,
                },
                step=index,
            )

    def log_export(self, export: PolicyExport, *, digests: Mapping[str, str] | None = None) -> None:
        """Link the produced policy to this run **by content hash** (learn.md §4).

        ``digests`` maps each agent to its exported ONNX graph digest
        (:attr:`ExportedPolicy.digest`) — the artifact identity Bench scores and Hub
        distributes. Recording it here is what closes the loop from a leaderboard entry back to
        the exact run, seed, and comms regime that produced it."""
        self.backend.log_dict(
            {
                "algorithm": export.algorithm,
                "backend": export.backend,
                "metrics": dict(export.metrics),
                "surrogate_fidelity_caveats": list(export.assumptions.surrogate_fidelity_caveats),
                "comms_observability": (
                    None
                    if export.assumptions.comms_observability is None
                    else dict(export.assumptions.comms_observability)
                ),
                "onnx_digests": dict(digests or {}),
            },
            "policy_export.json",
        )
        if digests:
            self.artifact_digests.update(digests)


@contextmanager
def tracked_run(config: TrainConfig, **kwargs: Any) -> Iterator[TrackedRun]:
    """Context-manager sugar for :class:`TrackedRun` (opens on enter, closes on exit)."""
    run = TrackedRun(config, **kwargs)
    run.start()
    try:
        yield run
    finally:
        run.finish()
