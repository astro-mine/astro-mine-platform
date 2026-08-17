# SPDX-License-Identifier: Apache-2.0
"""GPU scheduling & sharing -- MIG partitioning, time-slicing, DCGM telemetry.

Small GPU jobs must not strand whole cards (``cloud.md`` §7, §8). :mod:`.mig` holds the
per-GPU **MIG** profile catalogue (e.g. an A100 -> 7x10 GB slices), validates a requested
slice against a card, and compiles the NVIDIA GPU Operator config for MIG, **time-slicing**,
and the **DCGM** exporter that feeds GPU-utilisation dashboards (``cloud.md`` §10).

Backlog: RM-P1-CLOUD-03 -- astro-mine-cloud#14
"""

from __future__ import annotations

from astro_mine.cloud.gpu.mig import (
    MIG_PROFILES,
    MigProfile,
    gpu_operator_values,
    mig_resource_name,
    slices_for,
    time_slicing_configmap,
    validate_profile,
)

__all__ = [
    "MIG_PROFILES",
    "MigProfile",
    "gpu_operator_values",
    "mig_resource_name",
    "slices_for",
    "time_slicing_configmap",
    "validate_profile",
]
