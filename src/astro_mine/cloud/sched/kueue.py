"""Kueue queueing -- quotas, fair-share, and admission back-pressure.

Compiles the Kueue objects that give a shared cluster fair-share scheduling (``cloud.md`` §4,
§9): a ``ResourceFlavor`` (a node class), a ``ClusterQueue`` (nominal quotas in a ``cohort``
for fair-share borrowing), and a per-namespace ``LocalQueue``. :class:`QueueAdmission` is the
pure back-pressure model -- it admits a tenant's request only while the tenant is within quota
and otherwise makes the work *wait*, so a flood from one tenant queues instead of starving the
others (``cloud.md`` §2 principle 9, §8 "degrade, don't collapse").

Backlog: RM-P1-CLOUD-03 -- astro-mine-cloud#14
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.cloud.k8s import Manifest, object_meta, sanitize_name

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "API_VERSION",
    "QueueAdmission",
    "cluster_queue",
    "local_queue",
    "resource_flavor",
]

API_VERSION = "kueue.x-k8s.io/v1beta1"


def resource_flavor(name: str, *, node_labels: Mapping[str, str] | None = None) -> Manifest:
    """A Kueue ``ResourceFlavor`` -- a node class quotas are counted against."""
    spec: Manifest = {}
    if node_labels:
        spec["nodeLabels"] = dict(node_labels)
    return {
        "apiVersion": API_VERSION,
        "kind": "ResourceFlavor",
        "metadata": object_meta(name, component="sched"),
        "spec": spec,
    }


def cluster_queue(
    name: str,
    *,
    cohort: str,
    quotas: Mapping[str, str],
    flavor: str = "default",
    weight: int | None = None,
    allow_borrowing: bool = True,
) -> Manifest:
    """A ``ClusterQueue`` with nominal *quotas* in a *cohort* for fair-share borrowing.

    Queues sharing a *cohort* lend unused quota to one another; ``weight`` sets the tenant's
    fair-share weight. ``allow_borrowing=False`` makes the quota a hard ceiling (no lending),
    the pairing with a budget cap for a strict tenant.
    """
    resources = [
        {"name": resource, "nominalQuota": quota} for resource, quota in sorted(quotas.items())
    ]
    spec: Manifest = {
        "namespaceSelector": {},
        "cohort": sanitize_name(cohort),
        "resourceGroups": [
            {
                "coveredResources": sorted(quotas),
                "flavors": [{"name": sanitize_name(flavor), "resources": resources}],
            }
        ],
        "preemption": {
            "reclaimWithinCohort": "Any" if allow_borrowing else "Never",
            "withinClusterQueue": "LowerPriority",
        },
    }
    if weight is not None:
        spec["fairSharing"] = {"weight": weight}
    return {
        "apiVersion": API_VERSION,
        "kind": "ClusterQueue",
        "metadata": object_meta(name, component="sched"),
        "spec": spec,
    }


def local_queue(name: str, *, namespace: str, cluster_queue: str) -> Manifest:
    """A per-namespace ``LocalQueue`` pointing at a shared ``ClusterQueue``."""
    return {
        "apiVersion": API_VERSION,
        "kind": "LocalQueue",
        "metadata": object_meta(name, namespace=namespace, component="sched"),
        "spec": {"clusterQueue": sanitize_name(cluster_queue)},
    }


class QueueAdmission:
    """Pure per-tenant quota admission -- the back-pressure model behind Kueue.

    Each tenant has a resource quota; :meth:`admit` accepts a request only while the tenant
    stays within quota (and reserves it), else the request is *queued*. :meth:`release`
    returns the reservation when a job finishes. This models the guarantee that one tenant at
    its quota cannot consume another tenant's share (``cloud.md`` §9).
    """

    def __init__(self, quotas: Mapping[str, Mapping[str, float]]) -> None:
        self._quotas = {t: dict(q) for t, q in quotas.items()}
        self._used: dict[str, dict[str, float]] = {
            t: dict.fromkeys(q, 0.0) for t, q in quotas.items()
        }

    def _fits(self, tenant: str, request: Mapping[str, float]) -> bool:
        quota = self._quotas[tenant]
        used = self._used[tenant]
        return all(used.get(r, 0.0) + amount <= quota.get(r, 0.0) for r, amount in request.items())

    def admit(self, tenant: str, request: Mapping[str, float]) -> bool:
        """Reserve *request* for *tenant* if within quota; return whether it was admitted."""
        if tenant not in self._quotas:
            raise KeyError(f"unknown tenant {tenant!r}")
        if not self._fits(tenant, request):
            return False  # queued: the tenant is at quota, work waits (back-pressure)
        for resource, amount in request.items():
            self._used[tenant][resource] = self._used[tenant].get(resource, 0.0) + amount
        return True

    def release(self, tenant: str, request: Mapping[str, float]) -> None:
        """Return *tenant*'s reservation of *request* when a job completes."""
        for resource, amount in request.items():
            self._used[tenant][resource] = max(0.0, self._used[tenant].get(resource, 0.0) - amount)

    def used(self, tenant: str) -> dict[str, float]:
        """Current reserved usage for *tenant*."""
        return dict(self._used[tenant])
