"""The Phase-1 platform Helm chart -- structure of the pure-YAML files parses (RM-P1-CLOUD-01).

The full ``helm lint``/``helm template`` render runs in the CI ``helm`` job; here we assert the
declarative YAML the chart is built from is well-formed and wires the curated operator stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CHART_DIR = Path(__file__).resolve().parents[2] / "platform" / "helm" / "astro-mine-cloud"
PROFILES_DIR = Path(__file__).resolve().parents[2] / "platform" / "profiles"

# The curated operators the umbrella installs (cloud.md §4).
EXPECTED_OPERATORS = {
    "kuberay-operator",
    "argo-workflows",
    "kueue",
    "gpu-operator",
    "kube-prometheus-stack",
    "loki",
    # The admission controller. The chart emitted a Kyverno ClusterPolicy from the start but never
    # installed Kyverno, so "cosign-verified-images-only" (cloud.md §9) was a YAML file nothing
    # could enforce.
    "kyverno",
}

PROFILES = ["kind", "kind-admission", "prod"]


def _load(path: Path) -> dict:  # type: ignore[type-arg]
    return yaml.safe_load(path.read_text())


def test_chart_metadata_and_dependencies() -> None:
    chart = _load(CHART_DIR / "Chart.yaml")
    assert chart["name"] == "astro-mine-cloud"
    assert chart["apiVersion"] == "v2"
    assert chart["version"] == "0.1.0"
    deps = {d["name"] for d in chart["dependencies"]}
    assert deps == EXPECTED_OPERATORS
    # every dependency is condition-gated so a profile can toggle it
    assert all("condition" in d and "repository" in d for d in chart["dependencies"])


def test_values_expose_engine_and_admission_toggles() -> None:
    values = _load(CHART_DIR / "values.yaml")
    assert values["kuberay-operator"]["enabled"] is True
    assert values["argo-workflows"]["enabled"] is True
    assert values["kueue"]["enabled"] is True
    assert values["admission"]["enabled"] is True
    assert "cosignPublicKey" in values["admission"]


@pytest.mark.parametrize("profile", PROFILES)
def test_profiles_parse_and_cover_the_operators(profile: str) -> None:
    values = _load(PROFILES_DIR / f"{profile}.yaml")
    # each profile makes an explicit enabled decision for every operator
    assert set(values) >= EXPECTED_OPERATORS
    assert all("enabled" in values[op] for op in EXPECTED_OPERATORS)


def test_kind_profile_is_lightweight() -> None:
    values = _load(PROFILES_DIR / "kind.yaml")
    assert values["kuberay-operator"]["enabled"] is True  # engines on
    assert values["gpu-operator"]["enabled"] is False  # no GPUs in CI
    assert values["kube-prometheus-stack"]["enabled"] is False  # too heavy for a CI cluster


def test_kind_admission_profile_installs_and_scopes_the_policy() -> None:
    """The profile that finally makes the cosign policy *run* -- and keeps it from eating the
    cluster while it does."""
    values = _load(PROFILES_DIR / "kind-admission.yaml")
    assert values["kyverno"]["enabled"] is True  # the controller that enforces it
    assert values["admission"]["enabled"] is True

    # Scoped on both axes. An unscoped `*` rule over all namespaces refuses the platform's own
    # (unsigned, upstream) operator pods -- Kyverno's included -- and takes the cluster down.
    assert values["admission"]["imageReferences"] == ["kind-registry:5000/*"]
    assert values["admission"]["namespaces"] == ["astro-mine-admission"]
    # A local key pair, signed offline: there is no Rekor entry to look up.
    assert values["admission"]["ignoreTlog"] is True
    assert values["admission"]["requireAttestations"] is True


def test_the_default_admission_scope_stays_unrestricted() -> None:
    """The kind profile's scoping is a *dev* concession; the chart default must not inherit it."""
    values = _load(CHART_DIR / "values.yaml")
    assert values["admission"]["imageReferences"] == ["*"]
    assert values["admission"]["namespaces"] == []
    assert values["admission"]["ignoreTlog"] is False  # verify against the transparency log
    assert values["kyverno"]["enabled"] is False  # ...but do not enforce with no key configured


def test_required_templates_exist() -> None:
    templates = CHART_DIR / "templates"
    for name in ("_helpers.tpl", "admission-clusterpolicy.yaml", "observability.yaml", "NOTES.txt"):
        assert (templates / name).is_file(), name
    # the admission template enforces cosign + SLSA + SBOM, mirroring tenancy.admission
    policy = (templates / "admission-clusterpolicy.yaml").read_text()
    assert "verify-signature" in policy
    assert "slsa.dev/provenance" in policy
    assert "cyclonedx.org/bom" in policy
    assert "validationFailureAction: Enforce" in policy  # Audit mode would admit everything


def test_the_kind_harness_is_scripted_end_to_end() -> None:
    """The chart is only half the story: RM-P1-CLOUD-01 wants a cluster that actually comes up."""
    kind_dir = Path(__file__).resolve().parents[2] / "platform" / "kind"
    for name in (
        "cluster.yaml",
        "up.sh",
        "down.sh",
        "registry.sh",
        "minio.yaml",
        "workload.Dockerfile",
    ):
        assert (kind_dir / name).is_file(), name

    cluster = _load(kind_dir / "cluster.yaml")
    roles = [n["role"] for n in cluster["nodes"]]
    # Two workers, so the chaos test can take one away and still have somewhere to reschedule.
    assert roles.count("worker") >= 2, roles

    dockerfile = (kind_dir / "workload.Dockerfile").read_text()
    # The image's code_version and env_lockfile are both inside the RunContext content address, so
    # both must be pinned to the host's -- or the determinism gate fails for reasons unrelated to
    # the run. These two lines are the pins.
    assert "SETUPTOOLS_SCM_PRETEND_VERSION" in dockerfile
    assert "COPY pyproject.toml uv.lock" in dockerfile
    assert "ASTRO_MINE_ENV_LOCKFILE" in dockerfile
    assert "astro_mine.cloud.submission.harness" in dockerfile  # the entrypoint is the harness


def test_the_harness_registry_serves_tls() -> None:
    """astro-mine-cloud#30: the harness registry must serve TLS, or Kyverno cannot fetch a
    signature from it at all.

    Kyverno's registry client speaks HTTPS even with ``allowInsecure`` (that flag only skips cert
    *verification*, never downgrades to plaintext), so a plain-HTTP registry refuses every image
    for a transport reason -- signed or not. Guard the TLS wiring here, in the fast hermetic job,
    so it cannot silently revert to plain HTTP: the only other proof is the ~40-minute
    cluster-e2e run.
    """
    kind_dir = Path(__file__).resolve().parents[2] / "platform" / "kind"

    registry = (kind_dir / "registry.sh").read_text()
    # The registry is served with a TLS cert, not plain HTTP.
    assert "REGISTRY_HTTP_TLS_CERTIFICATE" in registry
    assert "REGISTRY_HTTP_TLS_KEY" in registry

    up = (kind_dir / "up.sh").read_text()
    # Each node's containerd trusts the registry over HTTPS via the harness CA -- not skip_verify
    # against an http:// endpoint.
    assert 'host."https://${REGISTRY}"' in up
    assert "skip_verify" not in up
    assert 'ca = "/etc/containerd/certs.d/${REGISTRY}/ca.crt"' in up
    # Kyverno is handed the same CA so its signature fetch trusts the registry's cert.
    assert "kyverno.admissionController.caCertificates.data" in up

    profile = _load(PROFILES_DIR / "kind-admission.yaml")
    # Real certificate verification -- the insecure-registry escape hatch is off, and Kyverno is
    # given the CA to verify against.
    assert profile["kyverno"]["features"]["registryClient"]["allowInsecure"] is False
    assert "caCertificates" in profile["kyverno"]["admissionController"]
