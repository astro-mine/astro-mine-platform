"""ResourceRequest / CheckpointPolicy -- the typed cluster fields on a JobSpec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission.jobspec import CheckpointPolicy, JobSpec, ResourceRequest

IMAGE = ImageRef.parse("ghcr.io/astro-mine/x@sha256:" + "cd" * 32)


def test_cpu_memory_only_appears_in_requests_and_limits() -> None:
    rr = ResourceRequest(cpu="2", memory="4Gi")
    assert rr.to_k8s_resources() == {
        "requests": {"cpu": "2", "memory": "4Gi"},
        "limits": {"cpu": "2", "memory": "4Gi"},
    }


def test_whole_gpu_is_an_extended_resource_in_both() -> None:
    res = ResourceRequest(gpu=2).to_k8s_resources()
    assert res["requests"]["nvidia.com/gpu"] == "2"
    assert res["limits"]["nvidia.com/gpu"] == "2"


def test_mig_profile_maps_to_the_slice_resource() -> None:
    res = ResourceRequest(mig_profile="1g.10gb").to_k8s_resources()
    assert res["requests"] == {"nvidia.com/mig-1g.10gb": "1"}
    assert res["limits"] == {"nvidia.com/mig-1g.10gb": "1"}


def test_empty_request_is_empty() -> None:
    assert ResourceRequest().to_k8s_resources() == {}


def test_gpu_and_mig_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="not both"):
        ResourceRequest(gpu=1, mig_profile="1g.10gb")


def test_malformed_mig_profile_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must look like"):
        ResourceRequest(mig_profile="half-a-card")


def test_negative_gpu_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResourceRequest(gpu=-1)


def test_checkpoint_policy_requires_positive_interval() -> None:
    assert CheckpointPolicy(interval_seconds=60).resume is True
    with pytest.raises(ValidationError):
        CheckpointPolicy(interval_seconds=0)


def test_jobspec_carries_cluster_fields_and_defaults() -> None:
    job = JobSpec(image=IMAGE)
    assert job.resource_request is None
    assert job.checkpoint is None
    assert job.distributed is False
    rich = JobSpec(
        image=IMAGE,
        resource_request=ResourceRequest(cpu="1"),
        checkpoint=CheckpointPolicy(interval_seconds=30),
        distributed=True,
    )
    assert rich.distributed is True
    assert rich.checkpoint is not None and rich.checkpoint.interval_seconds == 30
