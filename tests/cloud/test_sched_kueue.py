"""Kueue objects + admission back-pressure (fair-share, no starvation)."""

from __future__ import annotations

import pytest

from astro_mine.cloud.sched.kueue import (
    QueueAdmission,
    cluster_queue,
    local_queue,
    resource_flavor,
)


def test_resource_flavor_carries_node_labels() -> None:
    rf = resource_flavor("gpu-a100", node_labels={"accelerator": "a100"})
    assert rf["kind"] == "ResourceFlavor"
    assert rf["spec"]["nodeLabels"] == {"accelerator": "a100"}
    assert resource_flavor("plain")["spec"] == {}


def test_cluster_queue_quota_and_fair_share() -> None:
    cq = cluster_queue(
        "team-a", cohort="research", quotas={"cpu": "100", "nvidia.com/gpu": "8"}, weight=2
    )
    group = cq["spec"]["resourceGroups"][0]
    assert group["coveredResources"] == ["cpu", "nvidia.com/gpu"]
    assert cq["spec"]["cohort"] == "research"
    assert cq["spec"]["fairSharing"] == {"weight": 2}
    assert cq["spec"]["preemption"]["reclaimWithinCohort"] == "Any"


def test_cluster_queue_hard_ceiling_disables_borrowing() -> None:
    cq = cluster_queue("strict", cohort="c", quotas={"cpu": "10"}, allow_borrowing=False)
    assert cq["spec"]["preemption"]["reclaimWithinCohort"] == "Never"


def test_local_queue_points_at_cluster_queue() -> None:
    lq = local_queue("team-a", namespace="team-a", cluster_queue="team-a")
    assert lq["kind"] == "LocalQueue"
    assert lq["spec"]["clusterQueue"] == "team-a"


def test_admission_backpressure_prevents_starvation() -> None:
    admission = QueueAdmission({"noisy": {"cpu": 4.0}, "quiet": {"cpu": 4.0}})
    # the noisy tenant floods; admission stops at its own quota (back-pressure -> queued)
    assert admission.admit("noisy", {"cpu": 4.0}) is True
    assert admission.admit("noisy", {"cpu": 1.0}) is False  # over quota -> waits
    # the quiet tenant's share is untouched by the flood
    assert admission.admit("quiet", {"cpu": 4.0}) is True
    assert admission.used("noisy") == {"cpu": 4.0}


def test_release_returns_reserved_quota() -> None:
    admission = QueueAdmission({"t": {"cpu": 2.0}})
    assert admission.admit("t", {"cpu": 2.0}) is True
    assert admission.admit("t", {"cpu": 1.0}) is False
    admission.release("t", {"cpu": 2.0})
    assert admission.admit("t", {"cpu": 1.0}) is True


def test_unknown_tenant_rejected() -> None:
    with pytest.raises(KeyError, match="unknown tenant"):
        QueueAdmission({"t": {"cpu": 1.0}}).admit("ghost", {"cpu": 1.0})
