"""Autoscaling & resilience -- spot-first node pools and checkpoint-to-resume.

Cost is a first-class constraint (``cloud.md`` §2 principle 5): :mod:`.policy` compiles
**spot-first** node-autoscaling policies (Karpenter / cluster-autoscaler) with scale-to-zero
idle GPU pools and on-demand only for the irreplaceable learner, and :mod:`.checkpoint`
provides **content-addressed checkpoint-to-resume** so a spot eviction mid-run loses at most
one checkpoint interval and the resumed run reproduces the uninterrupted result (``cloud.md``
§8, §2 principle 4).

Backlog: RM-P1-CLOUD-03 -- https://github.com/astro-mine/astro-mine-cloud/issues/14
"""

from __future__ import annotations

from astro_mine.cloud.autoscale.checkpoint import (
    Checkpoint,
    CheckpointStore,
    Preempted,
    run_checkpointed,
)
from astro_mine.cloud.autoscale.policy import cluster_autoscaler_nodegroup, node_pool

__all__ = [
    "Checkpoint",
    "CheckpointStore",
    "Preempted",
    "cluster_autoscaler_nodegroup",
    "node_pool",
    "run_checkpointed",
]
