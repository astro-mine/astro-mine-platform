# SPDX-License-Identifier: Apache-2.0
"""Tier-1 training entrypoint + the RayJob entrypoint (RM-P1-LEARN-04; learn.md §2.1, §7).

The one command a `pip install`ed researcher runs to train a baseline on a single
workstation with no cloud (learn.md §7 tier 1), and — unchanged — the entrypoint Cloud wraps
in a KubeRay ``RayJob`` for scale-out (cloud.md). It selects the rollout executor from the
``TrainConfig`` fidelity axis (in-process :class:`LocalExecutor`, distributed
:class:`KubeRayExecutor`, or batched :class:`~astro_mine.learn.envs.vector.VectorExecutor`) —
"the same code with a different executor, never a fork" — trains the chosen baseline, and
emits the learning curve + throughput + the produced-policy provenance.

Learn never imports Cloud. Cloud injects its reproducibility envelope — the **RunContext**
(cloud.md: MLflow run id, image digest, Core interface version, lockfile, input hashes) — as
environment variables; this module *reads* that envelope (:class:`RunContext.from_env`) and
folds it into the produced policy's Core :class:`~astro_mine.core.registry.Provenance`
(:func:`apply_run_context`), completing the build-time + run-time reproducibility chain.

The world is supplied as an importable ``module:attr`` zero-arg **env factory** (``--env-factory``)
that yields either a :class:`~astro_mine.learn.envs.SwarmEnv` or the Core-typed
``(Environment, {AgentId: Asset})`` pair a Learn-free producer hands over — so the entrypoint
stays Sim-free (a real run points it at a Sim-backed factory; the CI smoke points it at the fake
world). The pair form is what lets the producing package avoid importing Learn while still
supplying the SADF the per-agent spaces are derived from.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from astro_mine.core.registry.model import Provenance
from astro_mine.learn.algos import PolicyExport, TrainConfig, default_registry
from astro_mine.learn.algos.registry import AlgorithmRegistry
from astro_mine.learn.envs import SwarmEnv, make_swarm_env
from astro_mine.learn.envs.vector import BatchedWorldFactory, VectorExecutor
from astro_mine.learn.train.executor import (
    EnvFactory,
    KubeRayExecutor,
    LocalExecutor,
    RolloutExecutor,
)

__all__ = [
    "DEFAULT_EXPORT_VERSION",
    "RunContext",
    "TrainRunReport",
    "apply_run_context",
    "build_executor",
    "export_trained_policy",
    "resolve_batched_world",
    "resolve_env_factory",
    "train",
]

#: Version stamped on an exported PolicyPackage when ``--export-version`` is not given. Hub
#: rejects re-publishing an existing ``name:version``, so a real publish pins its own.
DEFAULT_EXPORT_VERSION = "0.1.0"

#: Shown when ``--export`` is requested without the optional [export] extra installed. An
#: actionable install line, never a traceback (CX-LOCAL).
_EXPORT_INSTALL_HINT = (
    "--export needs the ONNX toolchain from the optional [export] extra; install it with "
    "`uv sync --extra export` (or `pip install 'astro-mine-platform[learn-export]'`) and re-run"
)

#: RunContext environment-variable names — the Learn<->Cloud contract. Cloud injects the
#: envelope on every execution (cloud.md); Learn only reads it, never calls Cloud.
ENV_RUN_ID = "ASTRO_MINE_RUN_ID"
ENV_IMAGE_DIGEST = "ASTRO_MINE_IMAGE_DIGEST"
ENV_ENV_LOCKFILE = "ASTRO_MINE_ENV_LOCKFILE"
ENV_CORE_INTERFACE_VERSION = "ASTRO_MINE_CORE_INTERFACE_VERSION"
ENV_INPUT_HASHES = "ASTRO_MINE_INPUT_HASHES"


@dataclass(frozen=True)
class RunContext:
    """Cloud's reproducibility envelope, read from the environment (cloud.md "RunContext").

    Distinct from build-time :class:`~astro_mine.core.registry.Provenance` — this is the
    *run-time* execution envelope Cloud auto-attaches; :func:`apply_run_context` folds the two
    into the full chain. Every field is optional so a bare tier-1 run (no Cloud) yields an
    empty context and the CLI still works."""

    run_id: str | None = None
    image_digest: str | None = None
    env_lockfile: str | None = None
    core_interface_version: str | None = None
    input_hashes: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RunContext:
        env = os.environ if environ is None else environ
        raw = env.get(ENV_INPUT_HASHES, "")
        hashes = tuple(part for part in (chunk.strip() for chunk in raw.split(",")) if part)
        return cls(
            run_id=env.get(ENV_RUN_ID),
            image_digest=env.get(ENV_IMAGE_DIGEST),
            env_lockfile=env.get(ENV_ENV_LOCKFILE),
            core_interface_version=env.get(ENV_CORE_INTERFACE_VERSION),
            input_hashes=hashes,
        )

    @property
    def is_empty(self) -> bool:
        return (
            not any(
                (self.run_id, self.image_digest, self.env_lockfile, self.core_interface_version)
            )
            and not self.input_hashes
        )


def apply_run_context(provenance: Provenance, context: RunContext) -> Provenance:
    """Fold Cloud's RunContext into a produced policy's build-time Provenance.

    The RunContext's resolved-input hashes are appended to ``input_hashes`` (the run id and
    image digest as labelled entries), and the lockfile reference lands on ``env_lockfile`` if
    the build did not already record one — so Bench can re-derive the artifact by content hash
    (CX-REPRO)."""
    input_hashes = list(provenance.input_hashes)
    if context.run_id:
        input_hashes.append(f"run_id:{context.run_id}")
    if context.image_digest:
        input_hashes.append(f"image:{context.image_digest}")
    input_hashes.extend(context.input_hashes)
    return provenance.model_copy(
        update={
            "input_hashes": input_hashes,
            "env_lockfile": provenance.env_lockfile or context.env_lockfile,
        }
    )


def _load_dotted(dotted: str, *, flag: str) -> Any:
    """Import a ``module:attr`` target, failing actionably rather than with a bare traceback."""
    module_name, sep, attr = dotted.partition(":")
    if not sep or not attr:
        raise ValueError(f"{flag} must be 'module:attr', got {dotted!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # the common case: the producing package is not installed
        raise ValueError(
            f"{flag} {dotted!r}: cannot import {module_name!r} ({exc}). "
            f"Install the package that provides it — e.g. the reference environment ships in "
            f"astro-mine-platform (`uv pip install astro-mine-platform`)."
        ) from exc
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"{flag} {dotted!r}: {module_name!r} has no attribute {attr!r}") from exc
    if not callable(target):
        raise TypeError(f"{flag} {dotted!r} is not callable")
    return target


@dataclass(frozen=True)
class _ResolvedEnvFactory:
    """A picklable zero-arg :class:`SwarmEnv` factory over a ``module:attr`` target.

    Holds the *dotted string*, not the resolved callable, and re-resolves on each call. That is
    what keeps it picklable for :class:`KubeRayExecutor`, which ships the factory to each rollout
    worker so the worker builds its own env — a closure over the resolved target would not
    survive that hop, and a user factory that is itself unpicklable (a lambda, a bound method)
    now does.

    It accepts **either** shape a producer may yield, and normalizes here so the nine call sites
    downstream keep seeing one type:

    - a :class:`~astro_mine.learn.envs.SwarmEnv` — returned unchanged; or
    - a Core-typed ``(Environment, {AgentId: Asset})`` pair, wrapped via ``make_swarm_env``.

    The pair form is what lets a producer stay Learn-free. A Core ``Environment`` alone is not
    enough to build a ``SwarmEnv`` — the per-agent observation/action spaces are derived from
    SADF — so a producer that returned only an environment could not be wrapped, and one that
    returned a ``SwarmEnv`` would have to import Learn. Handing over both Core-typed halves is
    what avoids a dependency in either direction (conventions.md §1.1).
    """

    dotted: str

    def __call__(self) -> SwarmEnv:
        produced = _load_dotted(self.dotted, flag="--env-factory")()
        if isinstance(produced, SwarmEnv):
            return produced
        try:
            env, assets = produced
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"--env-factory {self.dotted!r} returned {type(produced).__name__}; expected a "
                f"SwarmEnv or an (Environment, {{AgentId: Asset}}) pair"
            ) from exc
        return make_swarm_env(env, assets)


def resolve_env_factory(dotted: str) -> EnvFactory:
    """Resolve a ``module:attr`` dotted path to a zero-arg :class:`SwarmEnv` factory.

    The target may yield a ``SwarmEnv`` directly, or the Core-typed ``(Environment, assets)``
    pair a Learn-free producer hands over; see :class:`_ResolvedEnvFactory`. Resolution is
    deferred to call time so an unimportable target fails when the factory is *used*, with a
    message naming the package to install.
    """
    _, sep, attr = dotted.partition(":")
    if not sep or not attr:
        raise ValueError(f"--env-factory must be 'module:attr', got {dotted!r}")
    return cast(EnvFactory, _ResolvedEnvFactory(dotted))


def resolve_batched_world(dotted: str | None) -> BatchedWorldFactory | None:
    """Resolve a ``module:attr`` dotted path to a zero-arg :class:`BatchedWorld` factory.

    The GPU-vectorized tier's world (RM-P1-LEARN-04): Sim's Brax/MJX GPU tier, a JAX surrogate,
    or Learn's own :func:`~astro_mine.learn.envs.vector.jax_batched_world_factory`. ``None``
    leaves ``--fidelity gpu_vectorized`` on the sequential CPU fallback."""
    if dotted is None:
        return None
    module_name, sep, attr = dotted.partition(":")
    if not sep or not attr:
        raise ValueError(f"--batched-world must be 'module:attr', got {dotted!r}")
    factory = getattr(importlib.import_module(module_name), attr)
    if not callable(factory):
        raise TypeError(f"batched world factory {dotted!r} is not callable")
    return cast(BatchedWorldFactory, factory)


def build_executor(
    config: TrainConfig,
    env_factory: EnvFactory,
    *,
    ray_address: str | None = None,
    batched_world: BatchedWorldFactory | None = None,
) -> RolloutExecutor:
    """Pick the rollout executor from the fidelity axis + topology — the LEARN-04 seam.

    ``gpu_vectorized`` fidelity ⇒ the batched :class:`VectorExecutor`; a multi-worker or
    address-bound run ⇒ the distributed :class:`KubeRayExecutor`; otherwise the tier-1
    in-process :class:`LocalExecutor`. The trainer code is identical across all three.

    ``batched_world`` supplies the GPU-vectorized tier's batched world (Sim's Brax/MJX tier, or
    :func:`~astro_mine.learn.envs.vector.jax_batched_world_factory`). With one, the
    :class:`VectorExecutor` runs its genuinely batched kernel; without one — or without the
    ``[jax]`` extra — it degrades gracefully to the sequential CPU loop, so ``--fidelity
    gpu_vectorized`` never *fails* on a workstation with no GPU, it just runs the slow path."""
    if config.fidelity == "gpu_vectorized":
        return VectorExecutor(env_factory, num_envs=config.num_workers, batched_world=batched_world)
    if config.num_workers > 1 or ray_address is not None:
        return KubeRayExecutor(env_factory, num_workers=config.num_workers, ray_address=ray_address)
    return LocalExecutor()


@dataclass(frozen=True)
class TrainRunReport:
    """The reproducible result of a training run — the JSON the entrypoint emits."""

    algorithm: str
    learning_curve: tuple[float, ...]
    train_throughput_steps_per_s: float
    provenance: Provenance
    surrogate_fidelity_caveats: tuple[str, ...]
    comms_observability: Mapping[str, Any] | None
    config: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "learning_curve": list(self.learning_curve),
            "train_throughput_steps_per_s": self.train_throughput_steps_per_s,
            "provenance": self.provenance.model_dump(mode="json"),
            "surrogate_fidelity_caveats": list(self.surrogate_fidelity_caveats),
            "comms_observability": (
                dict(self.comms_observability) if self.comms_observability is not None else None
            ),
            "config": self.config.model_dump(mode="json"),
        }


def train(
    algorithm_tag: str,
    env_factory: EnvFactory,
    config: TrainConfig,
    *,
    run_context: RunContext | None = None,
    ray_address: str | None = None,
    registry: AlgorithmRegistry | None = None,
    batched_world: BatchedWorldFactory | None = None,
) -> tuple[TrainRunReport, PolicyExport]:
    """Train ``algorithm_tag`` on ``env_factory`` under ``config`` and return the run report
    plus the typed :class:`PolicyExport` (the RM-P1-LEARN-05 export seam)."""
    registry = registry if registry is not None else default_registry()
    algorithm = registry.get(algorithm_tag)
    executor = build_executor(
        config, env_factory, ray_address=ray_address, batched_world=batched_world
    )
    trainer = algorithm.make_trainer(env_factory(), config, executor=executor)

    start = perf_counter()
    metrics = [trainer.train_iteration() for _ in range(config.iterations)]
    elapsed = perf_counter() - start
    total_steps = sum(int(m.get("env_steps", 0.0)) for m in metrics)

    export = trainer.export()
    provenance = export.provenance
    if run_context is not None and not run_context.is_empty:
        provenance = apply_run_context(provenance, run_context)
        # Fold the envelope into the *export* too, not only the report. `export_policy_package`
        # builds the sidecar's Core Provenance from `export.provenance`, so without this the
        # published artifact would silently lose the run id, image digest, and Cloud's resolved
        # input hashes that the report sitting beside it records (CX-REPRO).
        export = replace(export, provenance=provenance)

    report = TrainRunReport(
        algorithm=trainer.spec.name,
        learning_curve=tuple(trainer.learning_curve()),
        train_throughput_steps_per_s=(total_steps / elapsed) if elapsed > 0 else 0.0,
        provenance=provenance,
        surrogate_fidelity_caveats=tuple(export.assumptions.surrogate_fidelity_caveats),
        comms_observability=export.assumptions.comms_observability,
        config=config,
    )
    return report, export


def export_trained_policy(
    export: PolicyExport,
    store_dir: str | Path,
    *,
    version: str = DEFAULT_EXPORT_VERSION,
) -> list[tuple[str, str, Path]]:
    """Render every agent's actor to ONNX and write it to a content-addressed store.

    Returns ``(agent, digest, onnx_path)`` per agent, sorted by agent. The ONNX-Runtime
    equivalence gate runs inside
    :func:`~astro_mine.learn.export.export_policy_packages`, so a graph that diverges from its
    Torch source raises :class:`~astro_mine.learn.export.EquivalenceError` and never becomes a
    file. Every agent is exported (and therefore gated) **before** anything is written, so a
    divergence on the second agent cannot leave the first one's bytes behind.

    The ONNX toolchain lives in the optional ``[export]`` extra; its absence is reported as an
    install hint rather than an import traceback (CX-LOCAL)."""
    try:
        from astro_mine.learn.export import export_policy_packages, publish
    except ModuleNotFoundError as exc:
        if exc.name in {"onnx", "onnxruntime"}:
            raise ModuleNotFoundError(_EXPORT_INSTALL_HINT) from exc
        raise

    exported = export_policy_packages(export, version=version)
    written: list[tuple[str, str, Path]] = []
    for agent, policy in sorted(exported.items()):
        published = publish(policy, store_dir)
        written.append((agent, published.digest, published.onnx_path))
    return written














