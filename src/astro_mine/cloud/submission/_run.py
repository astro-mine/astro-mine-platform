"""The shared run harness -- what makes backends equivalent.

Every backend runs the *same* pipeline: stage content-addressed inputs into an isolated
run directory, hand off to a :class:`Launcher` (subprocess or container), capture declared
outputs back into the store by content address, and record a
:class:`~astro_mine.cloud.artifacts.runcontext.RunContext`. Only the launch step differs
per backend, so the same deterministic job yields identical output addresses and an
identical run-context address regardless of backend -- the CLOUD-02 equivalence contract.

Backlog: RM-P0-CLOUD-02 -- astro-mine-cloud#2
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astro_mine.cloud.artifacts.runcontext import RunContext
from astro_mine.cloud.runs.events import RunObserver, RunStatus
from astro_mine.cloud.submission.result import RunResult

if TYPE_CHECKING:
    from astro_mine.cloud.submission.jobspec import JobSpec
    from astro_mine.core.artifacts import ArtifactStore

__all__ = ["Launcher", "build_env", "execute", "pre_run_context"]

#: Env var that pins the environment lockfile explicitly, overriding cwd discovery.
ENV_LOCKFILE_VAR = "ASTRO_MINE_ENV_LOCKFILE"


def _active_uv_lock() -> Path | None:
    """Locate the ``uv.lock`` pinning this run's environment (``conventions.md`` §5).

    Honors :data:`ENV_LOCKFILE_VAR` as an explicit override, else searches upward from the
    current working directory. Returns ``None`` when no lockfile is discoverable, so
    provenance records ``env_lockfile = None`` rather than a wrong pin. For the local
    subprocess backend this lockfile is the *only* true environment pin (the recorded
    ``image_digest`` is not the actual execution environment).
    """
    override = os.environ.get(ENV_LOCKFILE_VAR)
    if override:
        path = Path(override)
        return path if path.is_file() else None
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        candidate = directory / "uv.lock"
        if candidate.is_file():
            return candidate
    return None


def _env_lockfile_address(store: ArtifactStore) -> str | None:
    """Content-address (and store) the active ``uv.lock``, or ``None`` if none is found."""
    lockfile = _active_uv_lock()
    return None if lockfile is None else store.put(lockfile.read_bytes())


@runtime_checkable
class Launcher(Protocol):
    """Runs a job's command with inputs/outputs bound to the given directories."""

    def launch(self, *, job: JobSpec, inputs_dir: Path, outputs_dir: Path) -> int: ...


def build_env(
    job: JobSpec, inputs_dir: Path, outputs_dir: Path, *, container: bool
) -> dict[str, str]:
    """Build the job's environment: ``ASTRO_MINE_{INPUTS,OUTPUTS,SEED}`` plus ``job.env``.

    ``container`` selects the in-container paths (``/inputs``, ``/outputs``) vs the host
    run-directory paths, so both backends present the workload one stable I/O contract.
    """
    env = dict(job.env)
    env["ASTRO_MINE_INPUTS"] = "/inputs" if container else str(inputs_dir)
    env["ASTRO_MINE_OUTPUTS"] = "/outputs" if container else str(outputs_dir)
    if job.seed is not None:
        env["ASTRO_MINE_SEED"] = str(job.seed)
    return env


def pre_run_context(job: JobSpec, store: ArtifactStore) -> RunContext:
    """The reproducibility envelope known *before* a run produces outputs (RM-P1-CLOUD-06).

    Built from the job's pinned inputs, seed, image digest, interface version, and the active
    lockfile -- everything the final :class:`RunContext` carries except ``outputs``. Its
    :meth:`~astro_mine.cloud.artifacts.runcontext.RunContext.run_pin` therefore equals the final
    context's, so the ``submitted``/``started`` events share a run identity with ``completed``.
    """
    return RunContext(
        source_content_hashes=dict(job.inputs),
        seed=job.seed,
        env_lockfile=_env_lockfile_address(store),
        image_digest=job.image.reference,
        core_interface_version=job.core_interface_version,
    )


def execute(
    job: JobSpec, store: ArtifactStore, launcher: Launcher, *, observer: RunObserver | None = None
) -> RunResult:
    """Run *job* via *launcher* against *store* and return its :class:`RunResult`.

    Emits the ``started`` lifecycle event before launch and ``completed``/``failed`` after (via
    *observer*; the default is inert, so the local tier stays broker-free -- RM-P1-CLOUD-06). Any
    exception during launch/output-capture emits ``failed`` before propagating.
    """
    observer = observer or RunObserver()
    with tempfile.TemporaryDirectory(prefix="astro-mine-run-") as tmp:
        root = Path(tmp)
        inputs_dir = root / "inputs"
        outputs_dir = root / "outputs"
        inputs_dir.mkdir()
        outputs_dir.mkdir()

        for name, address in job.inputs.items():
            destination = inputs_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(store.get(address))

        lockfile_address = _env_lockfile_address(store)
        # The pre-outputs pin: identity for `started` and the fallback address for a `failed` emit.
        pending = RunContext(
            source_content_hashes=dict(job.inputs),
            seed=job.seed,
            env_lockfile=lockfile_address,
            image_digest=job.image.reference,
            core_interface_version=job.core_interface_version,
        )
        observer.transition(pending, "started")

        try:
            exit_code = launcher.launch(job=job, inputs_dir=inputs_dir, outputs_dir=outputs_dir)

            outputs: dict[str, str] = {}
            if exit_code == 0:
                for name in job.outputs:
                    path = outputs_dir / name
                    try:
                        data = path.read_bytes()
                    except FileNotFoundError:
                        raise FileNotFoundError(
                            f"job succeeded but declared output {name!r} was not produced"
                        ) from None
                    outputs[name] = store.put(data)

            context = RunContext(
                source_content_hashes=dict(job.inputs),
                seed=job.seed,
                env_lockfile=lockfile_address,
                image_digest=job.image.reference,
                core_interface_version=job.core_interface_version,
                outputs=outputs,
            )
            address = context.store(store)
        except Exception:
            observer.transition(pending, "failed")
            raise

        status: RunStatus = "completed" if exit_code == 0 else "failed"
        observer.transition(context, status)
        return RunResult(
            status="succeeded" if exit_code == 0 else "failed",
            exit_code=exit_code,
            outputs=outputs,
            run_context_address=address,
            run_context=context,
        )
