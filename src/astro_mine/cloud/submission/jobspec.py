"""The JobSpec contract -- one containerized unit of work.

A :class:`JobSpec` is the small typed contract Cloud runs: a digest-pinned image, a
command, content-addressed inputs, and the outputs to capture (``cloud.md`` §3). Every
backend consumes the *same* JobSpec -- that is what makes ``submit()`` backend-agnostic
(RM-P0-CLOUD-02). Reserved cluster fields (``resources``/``tenant``/``priority``/
``budget``) default empty so the Phase-1 backend populates them without a schema bump,
mirroring :class:`~astro_mine.cloud.artifacts.runcontext.RunContext`.

Backlog: RM-P0-CLOUD-02 -- astro-mine-cloud#2
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astro_mine.cloud._compat import validate_core_interface_version
from astro_mine.cloud.artifacts.addressing import parse_address
from astro_mine.cloud.packaging.image import ImageRef

__all__ = ["CheckpointPolicy", "JobSpec", "ResourceRequest"]

#: A MIG profile string, e.g. ``1g.10gb`` -> the ``nvidia.com/mig-1g.10gb`` extended
#: resource. ``gpu/mig.py`` owns the authoritative per-GPU catalogue; this is a shape check.
_MIG_PROFILE = re.compile(r"^[0-9]+g\.[0-9]+gb$")


class ResourceRequest(BaseModel):
    """A pod's compute request: CPU / memory / whole GPUs or a MIG slice.

    Compiles to a Kubernetes ``resources`` block via :meth:`to_k8s_resources`. A **MIG
    slice** (``mig_profile``) and **whole GPUs** (``gpu``) are mutually exclusive -- a job
    that fits on a 10 GB slice must not also strand a whole card (``cloud.md`` §7, §8). GPU
    resources are extended resources, so requests are pinned equal to limits.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu: str | None = None
    memory: str | None = None
    gpu: int = Field(default=0, ge=0)
    mig_profile: str | None = None

    @field_validator("mig_profile")
    @classmethod
    def _check_mig_profile(cls, value: str | None) -> str | None:
        if value is not None and not _MIG_PROFILE.match(value):
            raise ValueError(f"mig_profile must look like '1g.10gb', got {value!r}")
        return value

    @model_validator(mode="after")
    def _gpu_or_mig(self) -> ResourceRequest:
        if self.gpu and self.mig_profile is not None:
            raise ValueError("set gpu (whole cards) or mig_profile (a slice), not both")
        return self

    def to_k8s_resources(self) -> dict[str, dict[str, str]]:
        """Render a Kubernetes container ``resources`` block (requests + limits)."""
        requests: dict[str, str] = {}
        if self.cpu is not None:
            requests["cpu"] = self.cpu
        if self.memory is not None:
            requests["memory"] = self.memory
        limits = dict(requests)
        if self.mig_profile is not None:
            gpu_key, gpu_qty = f"nvidia.com/mig-{self.mig_profile}", "1"
        elif self.gpu:
            gpu_key, gpu_qty = "nvidia.com/gpu", str(self.gpu)
        else:
            gpu_key = None
        if gpu_key is not None:
            # Extended (GPU) resources must appear identically in requests and limits.
            requests[gpu_key] = limits[gpu_key] = gpu_qty
        out: dict[str, dict[str, str]] = {}
        if requests:
            out["requests"] = requests
        if limits:
            out["limits"] = limits
        return out


class CheckpointPolicy(BaseModel):
    """When to checkpoint a long/preemptible job so a spot eviction loses <= one interval.

    A job carrying a checkpoint policy is resumable: ``autoscale/checkpoint.py`` writes
    content-addressed checkpoints to the artifact store every ``interval_seconds`` and, on
    preemption, resumes from the last one (``cloud.md`` §8; RM-P1-CLOUD-03).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    interval_seconds: int = Field(gt=0)
    resume: bool = True


def _safe_relative_name(name: str) -> str:
    """Validate *name* is a safe relative path (no abs / no ``..`` / no backslash escape).

    Input and output names become paths under an isolated run directory; rejecting
    absolute or parent-escaping names keeps a job from reading or writing outside it.
    """
    if not name:
        raise ValueError("name must be non-empty")
    if name.startswith("/") or "\\" in name:
        raise ValueError(f"name must be a relative POSIX path: {name!r}")
    if ".." in name.split("/"):
        raise ValueError(f"name must not escape the run directory with '..': {name!r}")
    return name


class JobSpec(BaseModel):
    """A containerized unit of work run identically by every backend.

    ``image`` is the digest-pinned workload; ``command`` is the argv run inside it (or,
    for the local backend, on the workstation directly). ``inputs`` maps a run-relative
    name to a content address staged from the store; ``outputs`` lists the run-relative
    filenames captured back into the store. ``seed`` is exported as ``ASTRO_MINE_SEED``.
    """

    model_config = ConfigDict(extra="forbid")

    image: ImageRef
    command: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    seed: int | None = None
    core_interface_version: str | None = None
    # Phase-1 cluster fields (default empty so the local tier ignores them) -----------
    resource_request: ResourceRequest | None = None
    checkpoint: CheckpointPolicy | None = None
    #: Tightly-coupled/stateful work (RL, actor fleets, distributed solves) -> Ray; a
    #: trivial one-shot (the default) -> a plain K8s Job (``cloud.md`` §2 principle 3).
    distributed: bool = False
    # raw K8s resource map escape hatch, kept for forward-compat with unusual devices ---
    resources: dict[str, str] = Field(default_factory=dict)
    tenant: str | None = None
    priority: int | None = None
    budget: float | None = None

    @field_validator("inputs")
    @classmethod
    def _validate_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        for name, address in value.items():
            _safe_relative_name(name)
            parse_address(address)  # raise loudly on a non-content-address input
        return value

    @field_validator("outputs")
    @classmethod
    def _validate_outputs(cls, value: list[str]) -> list[str]:
        for name in value:
            _safe_relative_name(name)
        return value

    @field_validator("core_interface_version")
    @classmethod
    def _validate_core_interface_version(cls, value: str | None) -> str | None:
        """Admit only a Core interface version this Core can satisfy (``_compat``)."""
        return validate_core_interface_version(value)
