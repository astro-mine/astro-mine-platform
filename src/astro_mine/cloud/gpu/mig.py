# SPDX-License-Identifier: Apache-2.0
"""MIG profiles, time-slicing, and DCGM -- GPU sharing config.

MIG partitions one physical GPU into isolated slices so a small fit/inference job shares a
card instead of stranding it (``cloud.md`` §7, §8). This module holds the per-GPU profile
catalogue, maps a profile to its ``nvidia.com/mig-<profile>`` extended resource, validates a
requested slice against a card, and compiles the NVIDIA GPU Operator's time-slicing ConfigMap
and Helm values (MIG strategy + DCGM exporter for GPU telemetry, ``cloud.md`` §10).

Backlog: RM-P1-CLOUD-03 -- astro-mine-cloud#14
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.cloud.k8s import Manifest, object_meta

__all__ = [
    "MIG_PROFILES",
    "MigProfile",
    "gpu_operator_values",
    "mig_resource_name",
    "slices_for",
    "time_slicing_configmap",
    "validate_profile",
]


class MigProfile(BaseModel):
    """A MIG slice profile: its name, memory, and how many fit on the card."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    memory_gb: int = Field(gt=0)
    instances: int = Field(gt=0)


#: Representative MIG catalogue per GPU model. Not exhaustive -- the profiles a deployment's
#: cards actually expose; a card's real capability is confirmed by the GPU Operator.
MIG_PROFILES: dict[str, tuple[MigProfile, ...]] = {
    "a100-40gb": (
        MigProfile(name="1g.5gb", memory_gb=5, instances=7),
        MigProfile(name="2g.10gb", memory_gb=10, instances=3),
        MigProfile(name="3g.20gb", memory_gb=20, instances=2),
        MigProfile(name="7g.40gb", memory_gb=40, instances=1),
    ),
    "a100-80gb": (
        MigProfile(name="1g.10gb", memory_gb=10, instances=7),
        MigProfile(name="2g.20gb", memory_gb=20, instances=3),
        MigProfile(name="3g.40gb", memory_gb=40, instances=2),
        MigProfile(name="7g.80gb", memory_gb=80, instances=1),
    ),
    "h100-80gb": (
        MigProfile(name="1g.10gb", memory_gb=10, instances=7),
        MigProfile(name="2g.20gb", memory_gb=20, instances=3),
        MigProfile(name="3g.40gb", memory_gb=40, instances=2),
        MigProfile(name="7g.80gb", memory_gb=80, instances=1),
    ),
}


def slices_for(gpu_model: str) -> tuple[MigProfile, ...]:
    """Return the MIG profiles a *gpu_model* supports, or raise with the known models."""
    try:
        return MIG_PROFILES[gpu_model]
    except KeyError:
        known = ", ".join(sorted(MIG_PROFILES))
        raise KeyError(f"unknown GPU model {gpu_model!r}; known: {known}") from None


def validate_profile(gpu_model: str, profile_name: str) -> MigProfile:
    """Return the :class:`MigProfile` for *profile_name* on *gpu_model*, or raise."""
    for profile in slices_for(gpu_model):
        if profile.name == profile_name:
            return profile
    available = ", ".join(p.name for p in slices_for(gpu_model))
    raise ValueError(f"{gpu_model!r} has no MIG profile {profile_name!r}; available: {available}")


def mig_resource_name(profile_name: str) -> str:
    """The extended-resource key for a MIG slice, e.g. ``nvidia.com/mig-1g.10gb``."""
    return f"nvidia.com/mig-{profile_name}"


def time_slicing_configmap(name: str, *, namespace: str, replicas: int) -> Manifest:
    """A GPU Operator time-slicing ConfigMap advertising *replicas* virtual GPUs per card.

    Time-slicing (unlike MIG) shares a card by oversubscription, not hardware isolation --
    the right lever for many tiny inference jobs that tolerate interleaving (``cloud.md`` §8).
    """
    if replicas < 1:
        raise ValueError("time-slicing replicas must be >= 1")
    config = (
        "version: v1\n"
        "sharing:\n"
        "  timeSlicing:\n"
        "    resources:\n"
        "    - name: nvidia.com/gpu\n"
        f"      replicas: {replicas}\n"
    )
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": object_meta(name, namespace=namespace, component="gpu"),
        "data": {"any": config},
    }


def gpu_operator_values(*, mig_strategy: str = "mixed", enable_dcgm: bool = True) -> Manifest:
    """Helm values for the NVIDIA GPU Operator: MIG strategy + DCGM telemetry.

    ``mig_strategy`` is ``mixed`` (heterogeneous slice profiles per node) or ``single``;
    DCGM exports GPU metrics for the utilisation dashboards (``cloud.md`` §10).
    """
    if mig_strategy not in {"mixed", "single", "none"}:
        raise ValueError("mig_strategy must be 'mixed', 'single', or 'none'")
    return {
        "migManager": {"enabled": mig_strategy != "none"},
        "mig": {"strategy": mig_strategy},
        "dcgmExporter": {"enabled": enable_dcgm},
    }
