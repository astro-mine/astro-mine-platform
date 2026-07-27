"""Shared Kubernetes-manifest helpers -- labels, naming, env, YAML."""

from __future__ import annotations

import pytest
import yaml

from astro_mine.cloud.k8s import (
    LABEL_COMPONENT,
    LABEL_MANAGED_BY,
    LABEL_RUN,
    LABEL_TENANT,
    MANAGED_BY,
    env_var_list,
    labels,
    object_meta,
    sanitize_name,
    to_yaml,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Tenant-A", "tenant-a"),
        ("  spaces here  ", "spaces-here"),
        ("weird__name!!", "weird-name"),
        ("-edges-", "edges"),
        ("x" * 80, "x" * 63),
    ],
)
def test_sanitize_name(raw: str, expected: str) -> None:
    assert sanitize_name(raw) == expected


def test_sanitize_name_rejects_unnameable() -> None:
    with pytest.raises(ValueError, match="no RFC-1123-safe"):
        sanitize_name("!!!")


def test_labels_carry_the_standard_set() -> None:
    out = labels(component="workload", tenant="Tenant-A", run="Run 1", extra={"k": "v"})
    assert out[LABEL_MANAGED_BY] == MANAGED_BY
    assert out[LABEL_COMPONENT] == "workload"
    assert out[LABEL_TENANT] == "tenant-a"
    assert out[LABEL_RUN] == "run-1"
    assert out["k"] == "v"


def test_labels_omit_tenant_and_run_when_absent() -> None:
    out = labels(component="workload")
    assert LABEL_TENANT not in out
    assert LABEL_RUN not in out


def test_object_meta_sanitizes_and_attaches() -> None:
    meta = object_meta(
        "My Job", namespace="NS-1", tenant="t", component="workload", annotations={"a": "b"}
    )
    assert meta["name"] == "my-job"
    assert meta["namespace"] == "ns-1"
    assert meta["annotations"] == {"a": "b"}
    assert meta["labels"][LABEL_TENANT] == "t"


def test_env_var_list_is_sorted() -> None:
    assert env_var_list({"B": "2", "A": "1"}) == [
        {"name": "A", "value": "1"},
        {"name": "B", "value": "2"},
    ]


def test_to_yaml_single_and_stream_round_trip() -> None:
    one = {"kind": "Job", "metadata": {"name": "x"}}
    assert yaml.safe_load(to_yaml(one)) == one
    two = [one, {"kind": "Pod"}]
    assert list(yaml.safe_load_all(to_yaml(two))) == two
