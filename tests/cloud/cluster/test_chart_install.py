"""The umbrella chart really installs: the operators are up and their CRDs are servable.

``helm lint`` (the default CI job) renders templates. It does not tell you whether KubeRay, Kueue
and Argo actually come up, or whether the CRDs the engines compile against exist. That is what
this asserts, against the cluster ``platform/kind/up.sh`` stood up with a real
``helm dependency build && helm install`` (RM-P1-CLOUD-01).
"""

from __future__ import annotations

import pytest

from tests.cloud.cluster.conftest import CLUSTER_MARKS, Kubectl

pytestmark = CLUSTER_MARKS

#: The custom resources the engines compile to. If these are not servable, every compiled
#: manifest in this repo is fiction.
ENGINE_CRDS = [
    "rayjobs.ray.io",  # engines/ray.py
    "rayclusters.ray.io",
    "workflows.argoproj.io",  # engines/argo.py
    "clusterqueues.kueue.x-k8s.io",  # sched/kueue.py
    "localqueues.kueue.x-k8s.io",
    "resourceflavors.kueue.x-k8s.io",
]


@pytest.mark.parametrize("crd", ENGINE_CRDS)
def test_the_engines_custom_resources_are_established(kubectl: Kubectl, crd: str) -> None:
    kubectl("wait", "--for=condition=Established", f"crd/{crd}", "--timeout=120s")


def test_the_operator_deployments_are_available(kubectl: Kubectl) -> None:
    """Every operator the kind profile enables is *Available*, not merely created."""
    kubectl(
        "wait",
        "--for=condition=Available",
        "deployment",
        "--all",
        "-n",
        "astro-mine-system",
        "--timeout=300s",
    )


def test_a_compiled_kueue_object_is_accepted_by_the_real_api_server(kubectl: Kubectl) -> None:
    """The manifest builders in ``sched/kueue.py`` produce objects a live Kueue admits.

    Until now they were only asserted to be the dicts we expected -- never that Kubernetes agrees.
    """
    from astro_mine.cloud.k8s import to_yaml
    from astro_mine.cloud.sched.kueue import cluster_queue, resource_flavor

    kubectl.apply(
        to_yaml(
            [
                resource_flavor("chart-probe"),
                cluster_queue(
                    "chart-probe", cohort="probe", quotas={"cpu": "1"}, flavor="chart-probe"
                ),
            ]
        )
    )
    try:
        # Read it back in the version we *wrote*. Kueue 0.17 serves v1beta1 but stores v1beta2, and
        # the field is `cohort` in the former and `cohortName` in the latter -- so a bare
        # `get clusterqueue` returns the converted v1beta2 object and `spec.cohort` KeyErrors. That
        # says nothing about our manifest, which the API server accepted and converted; asking for
        # v1beta1 explicitly is what actually asserts the round-trip.
        served = kubectl.json("get", "clusterqueue.v1beta1.kueue.x-k8s.io", "chart-probe")
        assert served["spec"]["cohort"] == "probe"
    finally:
        kubectl("delete", "clusterqueue", "chart-probe", check=False)
        kubectl("delete", "resourceflavor", "chart-probe", check=False)
