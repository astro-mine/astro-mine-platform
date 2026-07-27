"""Node-autoscaling policies -- spot-first, scale-to-zero, on-demand learner."""

from __future__ import annotations

import pytest

from astro_mine.cloud.autoscale.policy import cluster_autoscaler_nodegroup, node_pool


def _capacity_values(pool: dict) -> list[str]:  # type: ignore[type-arg]
    reqs = pool["spec"]["template"]["spec"]["requirements"]
    return next(r["values"] for r in reqs if r["key"] == "karpenter.sh/capacity-type")


def test_node_pool_is_spot_first_by_default() -> None:
    pool = node_pool("cpu")
    assert pool["kind"] == "NodePool"
    assert _capacity_values(pool) == ["spot", "on-demand"]
    assert pool["spec"]["disruption"]["consolidationPolicy"] == "WhenEmpty"  # scale-to-zero


def test_on_demand_only_for_the_irreplaceable_learner() -> None:
    assert _capacity_values(node_pool("learner", on_demand_only=True)) == ["on-demand"]


def test_gpu_pool_requires_a_gpu_and_can_disable_scale_to_zero() -> None:
    pool = node_pool("gpu", gpu=True, scale_to_zero=False)
    keys = [r["key"] for r in pool["spec"]["template"]["spec"]["requirements"]]
    assert "karpenter.k8s.aws/instance-gpu-count" in keys
    assert pool["spec"]["disruption"]["consolidationPolicy"] == "WhenEmptyOrUnderutilized"


def test_spot_first_can_be_flipped() -> None:
    assert _capacity_values(node_pool("cpu", spot_first=False)) == ["on-demand", "spot"]


def test_cluster_autoscaler_nodegroup_portable_shape() -> None:
    ng = cluster_autoscaler_nodegroup("workers", max_nodes=20)
    assert ng == {"name": "workers", "minSize": 0, "maxSize": 20, "capacityType": "spot"}
    on_demand = cluster_autoscaler_nodegroup("od", min_nodes=1, max_nodes=3, spot=False)
    assert on_demand["capacityType"] == "on-demand"


def test_cluster_autoscaler_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="min_nodes <= max_nodes"):
        cluster_autoscaler_nodegroup("bad", min_nodes=5, max_nodes=1)
