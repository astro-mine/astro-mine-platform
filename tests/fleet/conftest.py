"""Shared SADF fixtures for the Fleet test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

# A minimal, valid SADF v0.1 document (frames empty → root_frame may be any name).
VALID_SADF = """\
sadf_version: "0.1"
asset:
  identity:
    id: test.rover
    name: Test Rover
    version: "0.1.0"
    kind: rover
  core_interface_versions:
    sadf: "0.1.0"
  root_frame: base
"""

# Missing required identity fields (name/version/kind) and root_frame.
INVALID_SADF = """\
sadf_version: "0.1"
asset:
  identity:
    id: broken
"""


@pytest.fixture
def valid_file(tmp_path: Path) -> Path:
    path = tmp_path / "asset.sadf.yaml"
    path.write_text(VALID_SADF, encoding="utf-8")
    return path


@pytest.fixture
def invalid_file(tmp_path: Path) -> Path:
    path = tmp_path / "broken.sadf.yaml"
    path.write_text(INVALID_SADF, encoding="utf-8")
    return path
