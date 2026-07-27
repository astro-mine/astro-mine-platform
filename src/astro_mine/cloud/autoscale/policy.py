"""Node-autoscaling policies -- spot-first, scale-to-zero, portable.

Compiles the node-autoscaling config that keeps compute cheap (``cloud.md`` §7, §8):
:func:`node_pool` emits a Karpenter ``NodePool`` that prefers **spot** (falling back to
on-demand), scales idle GPU pools **to zero** via empty-node consolidation, and can be pinned
**on-demand only** for the irreplaceable RL learner; :func:`cluster_autoscaler_nodegroup` is
the portable cluster-autoscaler equivalent for non-AWS substrates (``cloud.md`` §11).

Backlog: RM-P1-CLOUD-03 -- https://github.com/astro-mine/astro-mine-cloud/issues/14
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.cloud.k8s import Manifest, object_meta

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["KARPENTER_API_VERSION", "cluster_autoscaler_nodegroup", "node_pool"]

KARPENTER_API_VERSION = "karpenter.sh/v1"


def _capacity_types(*, spot_first: bool, on_demand_only: bool) -> list[str]:
    if on_demand_only:
        return ["on-demand"]
    return ["spot", "on-demand"] if spot_first else ["on-demand", "spot"]


def node_pool(
    name: str,
    *,
    instance_categories: Sequence[str] = ("c", "m", "r"),
    gpu: bool = False,
    spot_first: bool = True,
    on_demand_only: bool = False,
    scale_to_zero: bool = True,
    max_cpu: int = 1000,
) -> Manifest:
    """A Karpenter ``NodePool``: spot-first capacity, optional GPU, optional scale-to-zero.

    *on_demand_only* pins the pool to on-demand (the irreplaceable learner); *scale_to_zero*
    consolidates empty nodes away so an idle GPU pool costs nothing (``cloud.md`` §8).
    """
    requirements: list[Manifest] = [
        {
            "key": "karpenter.sh/capacity-type",
            "operator": "In",
            "values": _capacity_types(spot_first=spot_first, on_demand_only=on_demand_only),
        },
        {
            "key": "karpenter.k8s.aws/instance-category",
            "operator": "In",
            "values": list(instance_categories),
        },
    ]
    if gpu:
        requirements.append(
            {"key": "karpenter.k8s.aws/instance-gpu-count", "operator": "Gt", "values": ["0"]}
        )
    disruption = (
        {"consolidationPolicy": "WhenEmpty", "consolidateAfter": "30s"}
        if scale_to_zero
        else {"consolidationPolicy": "WhenEmptyOrUnderutilized"}
    )
    spec: Manifest = {
        "template": {"spec": {"requirements": requirements}},
        "disruption": disruption,
        "limits": {"cpu": str(max_cpu)},
    }
    return {
        "apiVersion": KARPENTER_API_VERSION,
        "kind": "NodePool",
        "metadata": object_meta(name, component="autoscale"),
        "spec": spec,
    }


def cluster_autoscaler_nodegroup(
    name: str,
    *,
    min_nodes: int = 0,
    max_nodes: int,
    spot: bool = True,
) -> Manifest:
    """A portable cluster-autoscaler node group (``min_nodes=0`` scales an idle pool to zero)."""
    if min_nodes < 0 or max_nodes < min_nodes:
        raise ValueError("require 0 <= min_nodes <= max_nodes")
    return {
        "name": name,
        "minSize": min_nodes,
        "maxSize": max_nodes,
        "capacityType": "spot" if spot else "on-demand",
    }
