"""MLflow tracking -- every job is a run recording its RunContext + artifacts.

A :class:`RunTracker` turns a
:class:`~astro_mine.cloud.artifacts.runcontext.RunContext` into an MLflow run: it logs the
reproducibility envelope (seed, image digest, Core interface version, lockfile, input hashes)
as params, the produced outputs as **content-addressed artifact refs**, and stamps the MLflow
``run_id`` back onto the context (``cloud.md`` §5, §6). The MLflow backend is injected via the
:class:`TrackingClient` seam -- :class:`MlflowTrackingClient` (behind the ``[mlflow]`` extra)
in production, a fake in tests -- so the local tier tracks nothing heavier than it wants.

Backlog: RM-P1-CLOUD-05 -- https://github.com/astro-mine/astro-mine-cloud/issues/16
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astro_mine.cloud.artifacts.runcontext import RunContext

__all__ = ["MlflowTrackingClient", "RunTracker", "TrackingClient"]


@runtime_checkable
class TrackingClient(Protocol):
    """The experiment-tracking backend seam (MLflow in production, a fake in tests)."""

    def start_run(self, *, experiment: str, tags: Mapping[str, str]) -> str: ...
    def log_params(self, run_id: str, params: Mapping[str, str]) -> None: ...
    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None: ...
    def log_artifact_ref(self, run_id: str, name: str, address: str) -> None: ...
    def end_run(self, run_id: str, status: str) -> None: ...


class RunTracker:
    """Records a run's provenance to a :class:`TrackingClient` and returns the stamped context."""

    def __init__(self, client: TrackingClient, *, experiment: str = "astro-mine") -> None:
        self._client = client
        self._experiment = experiment

    def _envelope_params(self, context: RunContext) -> dict[str, str]:
        params: dict[str, str] = {"schema_version": context.schema_version}
        if context.seed is not None:
            params["seed"] = str(context.seed)
        if context.image_digest is not None:
            params["image_digest"] = context.image_digest
        if context.core_interface_version is not None:
            params["core_interface_version"] = context.core_interface_version
        if context.env_lockfile is not None:
            params["env_lockfile"] = context.env_lockfile
        for name, address in context.source_content_hashes.items():
            params[f"input.{name}"] = address
        return params

    def record(
        self,
        context: RunContext,
        *,
        params: Mapping[str, str] | None = None,
        metrics: Mapping[str, float] | None = None,
        status: str = "FINISHED",
    ) -> RunContext:
        """Log *context* as an MLflow run; return the context stamped with its ``run_id``.

        The run is tagged with the context's content address (its reproducibility pin), so a
        re-run of the same job is findable and reproduces from the recorded envelope.
        """
        run_id = self._client.start_run(
            experiment=self._experiment, tags={"astro-mine.org/run": context.content_address()}
        )
        self._client.log_params(run_id, {**self._envelope_params(context), **(params or {})})
        if metrics:
            self._client.log_metrics(run_id, dict(metrics))
        for name, address in context.outputs.items():
            self._client.log_artifact_ref(run_id, name, address)
        self._client.end_run(run_id, status)
        return context.model_copy(update={"run_id": run_id})


class MlflowTrackingClient:
    """A :class:`TrackingClient` backed by a real MLflow tracking store (``[mlflow]`` extra).

    Wraps ``mlflow.tracking.MlflowClient`` -- MLflow's **explicit**, run-id-addressed API -- not the
    fluent ``mlflow.start_run()`` / ``mlflow.log_params()`` module API. The seam is run-id-addressed
    by contract (every call names the run it applies to), whereas the fluent API writes to a single
    *process-global active run*: a worker recording more than one run at a time (or a second
    tracker in the same process) would mis-attribute params, metrics, and artifact refs to whichever
    run happens to be active, and a second ``start_run`` would fail outright. The explicit client
    has no global state, so it honors the seam exactly and is safe to share across concurrent runs.

    ``tracking_uri`` is any MLflow store URI -- an ``http(s)://`` tracking server (the cluster tier;
    see ``docker-compose.yml``), a ``file://`` store, or ``None`` to defer to MLflow's own default
    resolution (``MLFLOW_TRACKING_URI``, else ``./mlruns``). The local tier needs none of it:
    :class:`RunTracker` is backend-agnostic (``cloud.md`` §2 principle 2, §5).
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        try:
            from mlflow.tracking import MlflowClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MlflowTrackingClient needs the 'mlflow' extra: pip install "
                "'astro-mine-cloud[mlflow]'"
            ) from exc
        self._client: Any = MlflowClient(tracking_uri=tracking_uri)
        self._experiment_ids: dict[str, str] = {}

    def _experiment_id(self, experiment: str) -> str:
        """Resolve (creating on first use) the id of the named experiment.

        Cloud fans a submission out over many workers, so two of them can race to create the same
        experiment: the loser's ``create_experiment`` fails with ``RESOURCE_ALREADY_EXISTS``, and
        the winner's experiment is then the one to use. Resolution is memoized per instance, so
        steady state costs no extra round trip.
        """
        cached = self._experiment_ids.get(experiment)
        if cached is not None:
            return cached
        found = self._client.get_experiment_by_name(experiment)
        if found is None:
            try:
                experiment_id = str(self._client.create_experiment(experiment))
            except Exception:  # a concurrent worker won the create race -- adopt its experiment
                found = self._client.get_experiment_by_name(experiment)
                if found is None:
                    raise
                experiment_id = str(found.experiment_id)
        else:
            experiment_id = str(found.experiment_id)
        self._experiment_ids[experiment] = experiment_id
        return experiment_id

    def start_run(self, *, experiment: str, tags: Mapping[str, str]) -> str:
        run = self._client.create_run(
            experiment_id=self._experiment_id(experiment), tags=dict(tags)
        )
        return str(run.info.run_id)

    def log_params(self, run_id: str, params: Mapping[str, str]) -> None:
        for key, value in params.items():
            self._client.log_param(run_id, key, value)

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            self._client.log_metric(run_id, key, value)

    def log_artifact_ref(self, run_id: str, name: str, address: str) -> None:
        # A *ref*, not the bytes: the artifact itself lives in the content-addressed store, so the
        # run records its address (``cloud.md`` §5) and stays cheap to write and to read back.
        self._client.set_tag(run_id, f"artifact.{name}", address)

    def end_run(self, run_id: str, status: str) -> None:
        self._client.set_terminated(run_id, status=status)
