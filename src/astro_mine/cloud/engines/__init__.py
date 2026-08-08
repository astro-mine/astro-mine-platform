"""Execution engines -- compile a JobSpec/SweepSpec/WorkflowSpec to cluster objects.

The right engine per workload shape (``cloud.md`` §2 principle 3, §3):

- **K8s Job / Indexed Job** (:mod:`.k8sjob`) for trivial one-shot containers;
- **Ray / KubeRay** (:mod:`.ray`) for tightly-coupled stateful work (RL training, actor
  rollout fleets, distributed solves);
- **Argo Workflows** (:mod:`.argo`) for DAG batch and embarrassingly-parallel fan-out.

Each engine is a **pure compiler** from a typed contract to a plain manifest ``dict`` -- no
cluster and no client needed to build (or test) one; applying is the cluster backend's job
(:mod:`astro_mine.cloud.submission.cluster`). :func:`~.selection.select_engine` routes a
single JobSpec to ``ray`` or ``k8sjob``; sweeps and workflows always compile to Argo.

Backlog: RM-P1-CLOUD-01/02 -- astro-mine-cloud#12
"""

from __future__ import annotations

from astro_mine.cloud.engines.argo import compile_sweep, compile_workflow
from astro_mine.cloud.engines.base import (
    HARNESS_COMMAND,
    Engine,
    container_spec,
    deterministic_name,
    get_engine,
    input_annotations,
    job_resources,
    queue_labels,
    register_engine,
    registered_engines,
)

# Importing the engine modules registers "k8sjob" and "ray" as a side effect.
from astro_mine.cloud.engines.k8sjob import K8sJobEngine, compile_job, resume_backoff_limit
from astro_mine.cloud.engines.ray import RayEngine, RayJobHandle, compile_rayjob
from astro_mine.cloud.engines.selection import select_engine

__all__ = [
    "HARNESS_COMMAND",
    "Engine",
    "K8sJobEngine",
    "RayEngine",
    "RayJobHandle",
    "compile_job",
    "compile_rayjob",
    "compile_sweep",
    "compile_workflow",
    "container_spec",
    "deterministic_name",
    "get_engine",
    "input_annotations",
    "job_resources",
    "queue_labels",
    "register_engine",
    "registered_engines",
    "resume_backoff_limit",
    "select_engine",
]
