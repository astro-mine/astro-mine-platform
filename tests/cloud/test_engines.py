"""Engine compilers -- JobSpec/SweepSpec/WorkflowSpec to K8s / Ray / Argo objects."""

from __future__ import annotations

import shlex

import pytest

from astro_mine.cloud.engines import (
    HARNESS_COMMAND,
    compile_sweep,
    compile_workflow,
    container_spec,
    deterministic_name,
    get_engine,
    input_annotations,
    job_resources,
    queue_labels,
    register_engine,
    registered_engines,
    resume_backoff_limit,
    select_engine,
)
from astro_mine.cloud.engines.base import workload_env
from astro_mine.cloud.engines.k8sjob import RESUME_BACKOFF_LIMIT, K8sJobEngine, compile_job
from astro_mine.cloud.engines.ray import RayJobHandle, compile_rayjob
from astro_mine.cloud.k8s import ENV_JOBSPEC, LABEL_QUEUE_NAME
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission.jobspec import CheckpointPolicy, JobSpec, ResourceRequest
from astro_mine.cloud.submission.sweepspec import SweepSpec
from astro_mine.cloud.submission.workflowspec import WorkflowSpec, WorkflowStep

IMAGE = ImageRef.parse("ghcr.io/astro-mine/x@sha256:" + "34" * 32)
JOB = JobSpec(
    image=IMAGE,
    command=["python", "-m", "run"],
    env={"FOO": "bar"},
    inputs={"in.txt": "sha256:" + "56" * 32},
    outputs=["out.txt"],
    seed=9,
    core_interface_version="0.1.0",
    resource_request=ResourceRequest(cpu="2", gpu=1),
    tenant="acme",
)
UNTENANTED = JOB.model_copy(update={"tenant": None})


# --- shared builders -----------------------------------------------------------------


def test_workload_env_includes_seed_and_core_version() -> None:
    env = workload_env(JOB)
    assert env["FOO"] == "bar"
    assert env["ASTRO_MINE_SEED"] == "9"
    assert env["ASTRO_MINE_CORE_INTERFACE_VERSION"] == "0.1.0"


def test_workload_env_carries_the_jobspec_for_the_in_pod_harness() -> None:
    """The pod reconstructs its own JobSpec -- that is how it stages content-addressed I/O."""
    env = workload_env(JOB)
    assert JobSpec.model_validate_json(env[ENV_JOBSPEC]) == JOB
    # ...and the serialized spec is *not* recursively embedded in itself.
    assert ENV_JOBSPEC not in JobSpec.model_validate_json(env[ENV_JOBSPEC]).env


def test_queue_labels_only_for_a_tenant_scoped_job() -> None:
    assert queue_labels(JOB) == {LABEL_QUEUE_NAME: "tenant-acme"}
    assert queue_labels(UNTENANTED) == {}


def test_container_spec_is_digest_pinned_and_sorted() -> None:
    c = container_spec(JOB)
    assert c["image"] == JOB.image.reference
    assert [e["name"] for e in c["env"]] == sorted(e["name"] for e in c["env"])
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_the_container_runs_the_harness_not_the_workload_command() -> None:
    """A Kubernetes `command` overrides the image ENTRYPOINT.

    Naming `job.command` here would run the workload *directly* -- with no inputs staged, no
    outputs captured and no provenance recorded, because a pod has no bind-mount to get them
    from. The harness is what runs; the workload's argv rides in the JobSpec it reads back.
    """
    c = container_spec(JOB)
    assert c["command"] == HARNESS_COMMAND
    assert c["command"] != JOB.command
    env = {e["name"]: e["value"] for e in c["env"]}
    assert JobSpec.model_validate_json(env[ENV_JOBSPEC]).command == JOB.command


def test_container_spec_can_drop_command_for_ray() -> None:
    assert "command" not in container_spec(JOB, include_command=False)


def test_job_resources_prefers_typed_then_raw_then_empty() -> None:
    assert job_resources(JOB)["requests"]["cpu"] == "2"
    raw = JOB.model_copy(update={"resource_request": None, "resources": {"cpu": "500m"}})
    assert job_resources(raw) == {"requests": {"cpu": "500m"}, "limits": {"cpu": "500m"}}
    bare = JOB.model_copy(update={"resource_request": None, "resources": {}})
    assert job_resources(bare) == {}


def test_input_annotations_are_sorted_json() -> None:
    ann = input_annotations(JOB)
    assert ann["astro-mine.org/inputs"] == '{"in.txt": "sha256:' + "56" * 32 + '"}'
    assert ann["astro-mine.org/outputs"] == '["out.txt"]'
    assert input_annotations(JobSpec(image=IMAGE)) == {}


def test_deterministic_name_is_stable_and_distinct() -> None:
    a = deterministic_name(JOB, prefix="amc-job")
    assert a == deterministic_name(JOB, prefix="amc-job")
    assert a.startswith("amc-job-")
    assert a != deterministic_name(JOB.model_copy(update={"seed": 10}), prefix="amc-job")


# --- selection & registry ------------------------------------------------------------


def test_select_engine_by_shape() -> None:
    assert select_engine(JOB) == "k8sjob"
    assert select_engine(JOB.model_copy(update={"distributed": True})) == "ray"


def test_engine_registry() -> None:
    assert set(registered_engines()) >= {"k8sjob", "ray"}
    with pytest.raises(ValueError, match="unknown engine"):
        get_engine("nope")
    with pytest.raises(ValueError, match="already registered"):
        register_engine(K8sJobEngine())
    register_engine(K8sJobEngine(), replace=True)  # replace is allowed


# --- K8s Job -------------------------------------------------------------------------


def test_compile_job_shape() -> None:
    m = compile_job(JOB, namespace="acme")
    assert m["apiVersion"] == "batch/v1"
    assert m["kind"] == "Job"
    assert m["metadata"]["namespace"] == "acme"
    assert m["spec"]["backoffLimit"] == 0
    assert m["spec"]["template"]["spec"]["restartPolicy"] == "Never"
    assert m["metadata"]["annotations"]["astro-mine.org/inputs"]


# --- Kueue admission + checkpoint-resume: JobSpec policy -> real Kubernetes behaviour ---


def test_tenant_job_is_suspended_and_labelled_for_its_local_queue() -> None:
    """A tenant-scoped Job is admitted *by Kueue*, so it is born suspended and queue-labelled."""
    m = compile_job(JOB, namespace="tenant-acme")
    assert m["metadata"]["labels"][LABEL_QUEUE_NAME] == "tenant-acme"
    assert m["spec"]["suspend"] is True
    # the queue label is Kueue's business, not a pod selector -- keep the template labels clean
    assert LABEL_QUEUE_NAME not in m["spec"]["template"]["metadata"]["labels"]


def test_untenanted_job_is_never_suspended() -> None:
    """Without a tenant there is no LocalQueue to admit it -- suspending would hang the run."""
    m = compile_job(UNTENANTED, namespace="default")
    assert "suspend" not in m["spec"]
    assert LABEL_QUEUE_NAME not in m["metadata"]["labels"]


def test_checkpointing_job_earns_a_retry_budget_that_disruption_does_not_charge() -> None:
    job = JOB.model_copy(update={"checkpoint": CheckpointPolicy(interval_seconds=30)})
    m = compile_job(job, namespace="acme")
    assert resume_backoff_limit(job) == RESUME_BACKOFF_LIMIT
    assert m["spec"]["backoffLimit"] == RESUME_BACKOFF_LIMIT
    rule = m["spec"]["podFailurePolicy"]["rules"][0]
    assert rule["action"] == "Ignore"
    assert rule["onPodConditions"] == [{"type": "DisruptionTarget", "status": "True"}]


def test_a_job_that_will_not_resume_keeps_the_no_retry_default() -> None:
    """`resume=False` means a retry could change the outcome -- so it gets none (§2 principle 4)."""
    job = JOB.model_copy(update={"checkpoint": CheckpointPolicy(interval_seconds=30, resume=False)})
    m = compile_job(job, namespace="acme")
    assert resume_backoff_limit(job) == 0
    assert m["spec"]["backoffLimit"] == 0
    assert "podFailurePolicy" not in m["spec"]


def test_explicit_backoff_limit_overrides_the_policy() -> None:
    job = JOB.model_copy(update={"checkpoint": CheckpointPolicy(interval_seconds=30)})
    assert compile_job(job, namespace="acme", backoff_limit=1)["spec"]["backoffLimit"] == 1


def test_compile_indexed_job() -> None:
    m = compile_job(JOB, namespace="acme", indexed=True, completions=8, parallelism=4)
    assert m["spec"]["completionMode"] == "Indexed"
    assert m["spec"]["completions"] == 8
    assert m["spec"]["parallelism"] == 4
    # parallelism defaults to completions when unset
    d = compile_job(JOB, namespace="acme", indexed=True, completions=8)
    assert d["spec"]["parallelism"] == 8


def test_k8s_engine_compile() -> None:
    assert get_engine("k8sjob").compile(JOB, namespace="acme")["kind"] == "Job"


# --- Ray -----------------------------------------------------------------------------


def test_compile_rayjob_shape() -> None:
    m = compile_rayjob(JOB, namespace="acme", workers=4, min_workers=1, max_workers=8)
    assert m["apiVersion"] == "ray.io/v1"
    assert m["kind"] == "RayJob"
    assert m["spec"]["shutdownAfterJobFinishes"] is True
    wg = m["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]
    assert (wg["replicas"], wg["minReplicas"], wg["maxReplicas"]) == (4, 1, 8)
    head = m["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]
    assert "command" not in head  # the entrypoint carries the command, not the container


def test_rayjob_entrypoint_is_the_harness_driver() -> None:
    """The RayJob entrypoint is the *driver* KubeRay runs on the head -- so it is the harness.

    Only the driver can stage the job's content-addressed inputs and capture its outputs; it then
    launches `job.command` itself. The entrypoint is a shell command *line*, not an argv, so it is
    built with `shlex.join` -- a naive `" ".join` mangles any argument holding a space or a quote.
    """
    m = compile_rayjob(JOB, namespace="acme")
    assert m["spec"]["entrypoint"] == shlex.join(HARNESS_COMMAND)
    assert shlex.split(m["spec"]["entrypoint"]) == HARNESS_COMMAND  # round-trips to the same argv

    head = m["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in head["env"]}
    assert JobSpec.model_validate_json(env[ENV_JOBSPEC]).command == JOB.command


def test_rayjob_carries_the_kueue_queue_label() -> None:
    m = compile_rayjob(JOB, namespace="tenant-acme")
    assert m["metadata"]["labels"][LABEL_QUEUE_NAME] == "tenant-acme"
    # ...but is not pre-suspended: Kueue only queues RayJobs when its ray.io integration is on,
    # and suspending an object nothing will admit hangs the run forever.
    assert "suspend" not in m["spec"]
    assert (
        LABEL_QUEUE_NAME
        not in compile_rayjob(UNTENANTED, namespace="default")["metadata"]["labels"]
    )


def test_rayjob_handle_from_manifest() -> None:
    m = compile_rayjob(JOB, namespace="acme")
    handle = RayJobHandle.from_manifest(m)
    assert handle.name == m["metadata"]["name"]
    assert handle.namespace == "acme"
    assert handle.entrypoint == shlex.join(HARNESS_COMMAND)


# --- Argo ----------------------------------------------------------------------------


def test_compile_workflow_dag() -> None:
    wf = WorkflowSpec(
        name="pipe",
        steps=[
            WorkflowStep(name="gen", job=JOB),
            WorkflowStep(name="run", job=JOB, depends_on=["gen"]),
        ],
    )
    m = compile_workflow(wf, namespace="acme")
    assert m["kind"] == "Workflow"
    tasks = m["spec"]["templates"][0]["dag"]["tasks"]
    run_task = next(t for t in tasks if t["name"] == "run")
    assert run_task["dependencies"] == ["gen"]
    assert len(m["spec"]["templates"]) == 3  # main + 2 step templates


def test_compile_sweep_fan_out_and_parallelism() -> None:
    sweep = SweepSpec(base=JOB, grid={"lr": [0.1, 0.2, 0.3]}, max_parallel=2)
    m = compile_sweep(sweep, namespace="acme")
    tasks = m["spec"]["templates"][0]["dag"]["tasks"]
    assert len(tasks) == 3
    assert m["spec"]["parallelism"] == 2


def test_compile_sweep_without_cap_has_no_parallelism() -> None:
    sweep = SweepSpec(base=JOB, grid={"lr": [0.1, 0.2]})
    assert "parallelism" not in compile_sweep(sweep, namespace="acme")["spec"]
