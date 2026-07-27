"""MIG profiles, time-slicing, and GPU Operator config."""

from __future__ import annotations

import pytest

from astro_mine.cloud.gpu.mig import (
    MIG_PROFILES,
    gpu_operator_values,
    mig_resource_name,
    slices_for,
    time_slicing_configmap,
    validate_profile,
)


def test_a100_partitions_into_seven_slices() -> None:
    smallest = min(slices_for("a100-40gb"), key=lambda p: p.memory_gb)
    assert smallest.name == "1g.5gb"
    assert smallest.instances == 7


def test_slices_for_unknown_model_raises() -> None:
    with pytest.raises(KeyError, match="unknown GPU model"):
        slices_for("gpu-9000")


def test_validate_profile_accepts_known_and_rejects_unknown() -> None:
    profile = validate_profile("a100-80gb", "1g.10gb")
    assert profile.memory_gb == 10
    with pytest.raises(ValueError, match="no MIG profile"):
        validate_profile("a100-80gb", "9g.99gb")


def test_mig_resource_name() -> None:
    assert mig_resource_name("1g.10gb") == "nvidia.com/mig-1g.10gb"


def test_time_slicing_configmap() -> None:
    cm = time_slicing_configmap("ts", namespace="gpu", replicas=4)
    assert cm["kind"] == "ConfigMap"
    assert "replicas: 4" in cm["data"]["any"]
    with pytest.raises(ValueError, match=">= 1"):
        time_slicing_configmap("ts", namespace="gpu", replicas=0)


def test_gpu_operator_values() -> None:
    values = gpu_operator_values(mig_strategy="mixed")
    assert values["mig"]["strategy"] == "mixed"
    assert values["dcgmExporter"]["enabled"] is True
    assert gpu_operator_values(mig_strategy="none")["migManager"]["enabled"] is False
    with pytest.raises(ValueError, match="mig_strategy"):
        gpu_operator_values(mig_strategy="bogus")


def test_catalogue_covers_the_common_cards() -> None:
    assert {"a100-40gb", "a100-80gb", "h100-80gb"} <= set(MIG_PROFILES)
