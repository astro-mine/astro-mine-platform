"""The Argo Workflows engine -- DAG batch and embarrassingly-parallel fan-out.

Compiles a :class:`~astro_mine.cloud.submission.workflowspec.WorkflowSpec` to an Argo
``Workflow`` whose ``dag`` template mirrors the step dependencies, and a
:class:`~astro_mine.cloud.submission.sweepspec.SweepSpec` to a fan-out ``Workflow`` -- one
task per expanded variant, so each Argo task runs exactly the container of a reproducible
expanded JobSpec (``cloud.md`` §3, §2 principle 4). ``SweepSpec.max_parallel`` becomes the
workflow ``parallelism`` cap so a huge fan-out degrades gracefully instead of stampeding the
scheduler (§8).

Backlog: RM-P1-CLOUD-01/02 -- https://github.com/astro-mine/astro-mine-cloud/issues/12
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.cloud.engines.base import container_spec, deterministic_name, input_annotations
from astro_mine.cloud.k8s import Manifest, object_meta, sanitize_name

if TYPE_CHECKING:
    from astro_mine.cloud.submission.jobspec import JobSpec
    from astro_mine.cloud.submission.sweepspec import SweepSpec
    from astro_mine.cloud.submission.workflowspec import WorkflowSpec

__all__ = ["API_VERSION", "compile_sweep", "compile_workflow"]

API_VERSION = "argoproj.io/v1alpha1"


def _container_template(name: str, job: JobSpec) -> Manifest:
    """An Argo container template running *job*'s container, with I/O annotations."""
    template: Manifest = {"name": name, "container": container_spec(job, name=name)}
    annotations = input_annotations(job)
    if annotations:
        template["metadata"] = {"annotations": annotations}
    return template


def compile_workflow(workflow: WorkflowSpec, *, namespace: str) -> Manifest:
    """Compile a :class:`WorkflowSpec` to an Argo ``Workflow`` with a ``dag`` entrypoint."""
    tasks: list[Manifest] = []
    templates: list[Manifest] = []
    for step in workflow.steps:
        template_name = f"tmpl-{sanitize_name(step.name)}"
        templates.append(_container_template(template_name, step.job))
        task: Manifest = {"name": sanitize_name(step.name), "template": template_name}
        if step.depends_on:
            task["dependencies"] = [sanitize_name(dep) for dep in step.depends_on]
        tasks.append(task)
    spec: Manifest = {
        "entrypoint": "main",
        "templates": [{"name": "main", "dag": {"tasks": tasks}}, *templates],
    }
    meta = object_meta(workflow.name, namespace=namespace, component="workflow")
    return {"apiVersion": API_VERSION, "kind": "Workflow", "metadata": meta, "spec": spec}


def compile_sweep(sweep: SweepSpec, *, namespace: str) -> Manifest:
    """Compile a :class:`SweepSpec` to an Argo fan-out ``Workflow`` (one task per variant)."""
    tasks: list[Manifest] = []
    templates: list[Manifest] = []
    for index, variant in enumerate(sweep.expand()):
        template_name = f"variant-{index}"
        templates.append(_container_template(template_name, variant))
        tasks.append({"name": template_name, "template": template_name})
    spec: Manifest = {
        "entrypoint": "main",
        "templates": [{"name": "main", "dag": {"tasks": tasks}}, *templates],
    }
    if sweep.max_parallel is not None:
        spec["parallelism"] = sweep.max_parallel
    meta = object_meta(
        deterministic_name(sweep.base, prefix="amc-sweep"),
        namespace=namespace,
        tenant=sweep.base.tenant,
        component="sweep",
    )
    return {"apiVersion": API_VERSION, "kind": "Workflow", "metadata": meta, "spec": spec}
