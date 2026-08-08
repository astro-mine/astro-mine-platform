"""The cluster backend -- compile a JobSpec to an engine object, dispatch it, collect the result.

``ClusterBackend`` closes the local<->cluster loop: the *same* ``submit(job)`` call site runs on a
cluster by only swapping the backend selector (``cloud.md`` §2 principle 2, §3). It routes the
job to an engine (:func:`~astro_mine.cloud.engines.selection.select_engine`), compiles the
manifest, and hands it to an **injectable** :class:`ClusterClient` -- exactly the seam
``DockerBackend`` uses for its runner.

:class:`KubectlClusterClient` (the registered default) is the real one. It applies the manifest,
waits for the object to reach a terminal state, and then **collects the result of the run the
cluster actually did**: the in-pod harness (:mod:`~astro_mine.cloud.submission.harness`) prints
its RunContext's content address, and this client loads that envelope back out of the *shared*
artifact store. Nothing is recomputed host-side, so a cluster ``RunResult`` is the pod's own --
which is what makes "the cluster run reproduces the laptop run" a real assertion rather than a
tautology.

Every ``kubectl`` call goes through a :class:`KubectlRunner` seam, mirroring ``DockerBackend``'s
runner. The wait / collect / parse *logic* is therefore unit-tested hermetically against a fake
runner, and only a ten-line subprocess shim needs a live cluster to exercise.

:class:`DryRunClient` records the compiled manifest and runs the job through the *local* harness.
It proves the call site and the manifest with no cluster at all -- but note what it cannot prove:
it re-runs *locally*, so comparing it against a local run compares local to local. The genuine
local<->cluster equivalence assertion lives in the opt-in ``cluster``-marked tests.

Backlog: RM-P1-CLOUD-02 -- astro-mine-cloud#21
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from astro_mine.cloud.artifacts.runcontext import RunContext
from astro_mine.cloud.engines import get_engine, select_engine
from astro_mine.cloud.runs.events import RunObserver
from astro_mine.cloud.submission._run import pre_run_context
from astro_mine.cloud.submission.backend import register_backend
from astro_mine.cloud.submission.harness import parse_sentinels
from astro_mine.cloud.submission.local import LocalBackend
from astro_mine.cloud.submission.result import RunResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from astro_mine.cloud.k8s import Manifest
    from astro_mine.cloud.runs.events import RunStatus
    from astro_mine.cloud.submission.jobspec import JobSpec
    from astro_mine.core.artifacts import ArtifactStore

__all__ = [
    "ClusterBackend",
    "ClusterClient",
    "ClusterDispatchError",
    "CommandResult",
    "DryRunClient",
    "KubectlClusterClient",
    "KubectlRunner",
]

#: Terminal phases, normalised across object kinds (and the RunResult's own status vocabulary).
_SUCCEEDED: Final[Literal["succeeded"]] = "succeeded"
_FAILED: Final[Literal["failed"]] = "failed"


class ClusterDispatchError(RuntimeError):
    """A dispatched object could not be applied, never finished, or reported no result."""


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one ``kubectl`` invocation: its exit status, its stdout, and its stderr.

    ``stderr`` carries the *reason* a kubectl call failed -- the admission rejection, the RBAC
    denial, the malformed field. Dropping it (as this once did) turns every cluster-side failure
    into a bare non-zero exit, which is the least useful thing a live-cluster harness can say.
    """

    returncode: int
    stdout: str
    stderr: str = ""


@runtime_checkable
class KubectlRunner(Protocol):
    """Runs a ``kubectl`` argv and returns its result -- the injectable cluster seam."""

    def run(self, argv: Sequence[str], *, stdin: bytes | None = None) -> CommandResult: ...


class _SubprocessKubectlRunner:
    """Shells out to the real ``kubectl`` -- the only piece here that needs a live cluster."""

    def run(  # pragma: no cover - requires kubectl + a live cluster
        self, argv: Sequence[str], *, stdin: bytes | None = None
    ) -> CommandResult:
        import subprocess

        completed = subprocess.run(list(argv), input=stdin, capture_output=True, check=False)
        return CommandResult(
            completed.returncode,
            completed.stdout.decode(errors="replace"),
            completed.stderr.decode(errors="replace"),
        )


@runtime_checkable
class ClusterClient(Protocol):
    """Dispatches a compiled manifest and returns the job's result."""

    def dispatch(
        self,
        job: JobSpec,
        manifest: Manifest,
        *,
        store: ArtifactStore,
        observer: RunObserver | None = None,
    ) -> RunResult: ...


class DryRunClient:
    """Records the compiled manifest and runs the job locally -- the no-cluster CI path.

    Proves the ``submit()`` call site works with ``backend="cluster"`` and lets a test inspect the
    manifest it *would* apply, while producing a real
    :class:`~astro_mine.cloud.submission.result.RunResult` via the local backend. It does not --
    and structurally cannot -- prove local<->cluster equivalence: it never leaves the workstation.
    """

    def __init__(self) -> None:
        self.dispatched: list[Manifest] = []

    def dispatch(
        self,
        job: JobSpec,
        manifest: Manifest,
        *,
        store: ArtifactStore,
        observer: RunObserver | None = None,
    ) -> RunResult:
        self.dispatched.append(manifest)
        return LocalBackend().run(job, store=store, observer=observer)


def _status_argv(kind: str, name: str, namespace: str) -> list[str]:
    """The ``kubectl get`` revealing whether *kind*/*name* has reached a terminal state."""
    path = (
        "jsonpath={.status.jobStatus}"
        if kind == "RayJob"
        else "jsonpath={range .status.conditions[*]}{.type}={.status};{end}"
    )
    return ["kubectl", "get", kind.lower(), name, "-n", namespace, "-o", path]


def _terminal_phase(kind: str, stdout: str) -> str | None:
    """Map a status query's output to ``succeeded`` / ``failed`` / ``None`` (still running)."""
    if kind == "RayJob":
        # KubeRay reports the *entrypoint's* outcome in .status.jobStatus -- not the
        # RayCluster's readiness, which succeeds long before the workload does.
        status = stdout.strip().upper()
        if status == "SUCCEEDED":
            return _SUCCEEDED
        return _FAILED if status in {"FAILED", "STOPPED"} else None
    if "Complete=True" in stdout:
        return _SUCCEEDED
    return _FAILED if "Failed=True" in stdout else None


def _pod_selectors(kind: str, name: str) -> list[str]:
    """Label selectors matching every pod that could have run *kind*/*name*'s workload.

    A Job's attempts all carry ``job-name``. A RayJob's entrypoint runs via the submitter Job
    KubeRay creates (named for the RayJob) against the Ray head, so both the submitter's pod and
    the cluster's own pods are candidates -- scan them all rather than guess which one holds the
    driver's stdout.
    """
    if kind == "RayJob":
        return [f"job-name={name}", f"ray.io/originated-from-cr-name={name}"]
    return [f"job-name={name}"]


class KubectlClusterClient:
    """Applies the compiled manifest to a live cluster and collects the run's real result.

    Needs the ``[cluster]`` extra (pyyaml, to render the manifest), ``kubectl`` on ``PATH``, a
    kubeconfig, and a workload image whose entrypoint is
    :mod:`astro_mine.cloud.submission.harness` -- so the whole path is exercised only by the
    opt-in ``cluster``-marked tests. Everything but the subprocess shim runs against an injected
    *runner* in the default suite.
    """

    def __init__(
        self,
        *,
        runner: KubectlRunner | None = None,
        timeout: float = 900.0,
        poll_interval: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runner: KubectlRunner = runner if runner is not None else _SubprocessKubectlRunner()
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._sleep = sleep
        self._monotonic = monotonic

    def dispatch(
        self,
        job: JobSpec,
        manifest: Manifest,
        *,
        store: ArtifactStore,
        observer: RunObserver | None = None,
    ) -> RunResult:
        from astro_mine.cloud.k8s import to_yaml

        observer = observer or RunObserver()
        kind = str(manifest["kind"])
        name = str(manifest["metadata"]["name"])
        namespace = str(manifest["metadata"].get("namespace", "default"))

        applied = self._runner.run(
            ["kubectl", "apply", "-f", "-"], stdin=to_yaml(manifest).encode()
        )
        if applied.returncode != 0:
            # kubectl writes its diagnosis to stderr, not stdout. Reporting only stdout (as this
            # once did) raised this error with an empty message -- a rejected manifest and an
            # unreachable API server were indistinguishable.
            detail = (applied.stderr or applied.stdout).strip()
            raise ClusterDispatchError(f"kubectl apply failed for {kind}/{name}: {detail}")

        # The pre-outputs pin -- the same run identity `submitted` carried -- so the host's event
        # stream stays coherent even though the pod's own observer is inert (RM-P1-CLOUD-06).
        observer.transition(pre_run_context(job, store), "started")

        phase = self._wait(kind=kind, name=name, namespace=namespace)
        collected = self._collect(kind=kind, name=name, namespace=namespace)
        if collected is None:
            raise ClusterDispatchError(
                f"{kind}/{name} reached {phase!r} but no run-context sentinel appeared in any of "
                "its pods' logs -- is the workload image's entrypoint "
                "`python -m astro_mine.cloud.submission.harness`?"
            )
        address, exit_code = collected

        # The result is the *pod's own*, read back from the shared store -- never recomputed here.
        context = RunContext.load(store, address)
        status: RunStatus = "completed" if exit_code == 0 else "failed"
        observer.transition(context, status)
        return RunResult(
            status=_SUCCEEDED if exit_code == 0 else _FAILED,
            exit_code=exit_code,
            outputs=dict(context.outputs),
            run_context_address=address,
            run_context=context,
        )

    def _wait(self, *, kind: str, name: str, namespace: str) -> str:
        """Poll until *kind*/*name* is terminal; return its phase, or raise on timeout."""
        deadline = self._monotonic() + self._timeout
        while True:
            result = self._runner.run(_status_argv(kind, name, namespace))
            if result.returncode == 0:
                phase = _terminal_phase(kind, result.stdout)
                if phase is not None:
                    return phase
            if self._monotonic() >= deadline:
                raise ClusterDispatchError(
                    f"timed out after {self._timeout}s waiting for {kind}/{name} in {namespace}"
                )
            self._sleep(self._poll_interval)

    def _collect(self, *, kind: str, name: str, namespace: str) -> tuple[str, int] | None:
        """Scan the object's pods, oldest first, for the harness sentinels; the last one wins.

        Oldest-first matters for a *resumed* run: the pod killed mid-run printed no sentinel, and
        the retry that finished the job did -- so what survives is the result the cluster actually
        completed with.
        """
        collected: tuple[str, int] | None = None
        for selector in _pod_selectors(kind, name):
            for pod in self._pods(namespace=namespace, selector=selector):
                parsed = parse_sentinels(self._logs(namespace=namespace, pod=pod))
                if parsed is not None:
                    collected = parsed
        return collected

    def _pods(self, *, namespace: str, selector: str) -> list[str]:
        result = self._runner.run(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                selector,
                "--sort-by=.metadata.creationTimestamp",
                "-o",
                "name",
            ]
        )
        return result.stdout.split() if result.returncode == 0 else []

    def _logs(self, *, namespace: str, pod: str) -> str:
        # A pod force-deleted mid-run has no logs left to read. That is not an error -- it is
        # precisely the case the *next* attempt's sentinel covers.
        result = self._runner.run(["kubectl", "logs", "-n", namespace, pod, "--tail=-1"])
        return result.stdout if result.returncode == 0 else ""


class ClusterBackend:
    """Compiles a JobSpec to an engine object and dispatches it via a :class:`ClusterClient`.

    *engine* forces an engine name; the default routes per workload shape. *client* is the
    dispatch seam (default: :class:`KubectlClusterClient`; inject :class:`DryRunClient` to
    exercise the call site without a cluster).
    """

    def __init__(
        self,
        *,
        client: ClusterClient | None = None,
        namespace: str = "default",
        engine: str | None = None,
    ) -> None:
        self._client = client if client is not None else KubectlClusterClient()
        self._namespace = namespace
        self._engine = engine

    def run(
        self, job: JobSpec, *, store: ArtifactStore, observer: RunObserver | None = None
    ) -> RunResult:
        engine_name = self._engine if self._engine is not None else select_engine(job)
        manifest = get_engine(engine_name).compile(job, namespace=self._namespace)
        return self._client.dispatch(job, manifest, store=store, observer=observer)


register_backend("cluster", ClusterBackend())
